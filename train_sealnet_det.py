import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
import os
import argparse
from tensorboardX import SummaryWriter
import time
from utils.model_library import *
from utils.dataloaders.data_loader_train_det import ImageFolderTrainDet
from utils.dataloaders.transforms_det import ShapeTransform
from PIL import ImageFile
import warnings

parser = argparse.ArgumentParser(description='trains a CNN to find seals in satellite imagery')
parser.add_argument('--training_dir', type=str, help='base directory to recursively search for images in')
parser.add_argument('--model_architecture', type=str, help='model architecture, must be a member of models '
                                                           'dictionary')
parser.add_argument('--hyperparameter_set', type=str, help='combination of hyperparameters used, must be a member of '
                                                           'hyperparameters dictionary')
parser.add_argument('--cv_weights', nargs='?', type=str, default='NO', help='weights for weighted-cross validation, '
                                                                            'must be a member of cv_weights dictionary')
parser.add_argument('--output_name', type=str, help='name of output file from training, this name will also be used in '
                                                    'subsequent steps of the pipeline')
parser.add_argument('--pipeline', type=str, help='name of the detection pipeline where the model will be saved')
parser.add_argument('--dest_folder', type=str, default='saved_models', help='folder where the model will be saved')

args = parser.parse_args()

# check for invalid inputs
if args.model_architecture not in model_archs:
    raise Exception("Invalid architecture -- see supported architectures:  {}".format(list(model_archs.keys())))

if args.training_dir not in training_sets:
    raise Exception("Training set is not defined in ./utils/model_library.py")

if args.hyperparameter_set not in hyperparameters:
    raise Exception("Hyperparameter combination is not defined in ./utils/model_library.py")

if args.pipeline not in model_defs:
    raise Exception('Pipeline is not defined in ./utils/model_library.py')

if args.cv_weights not in cv_weights:
    raise Exception("Cross-validation are not defined in ./utils/model_library.py")

# image transforms seem to cause truncated images, so we need this
ImageFile.LOAD_TRUNCATED_IMAGES = True

# we get an RGB warning, but the loader properly converts to RGB -after- this
warnings.filterwarnings('ignore', module='PIL')

# Data augmentation and normalization for training
# Just normalization for validation
arch_input_size = model_archs[args.model_architecture]['input_size']


data_transforms = {
    'training': {'shape_transform': ShapeTransform(arch_input_size, train=True),
                 'int_transform': transforms.Compose([
                     transforms.ColorJitter(brightness=np.random.choice([0, 1]) * 0.05,
                                            contrast=np.random.choice([0, 1]) * 0.05),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])},
    'validation': {'shape_transform': ShapeTransform(arch_input_size, train=False),
                   'int_transform': transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])},
}

# define data dir and image size
data_dir = "./training_sets/{}".format(args.training_dir)
img_size = training_sets[args.training_dir]['scale_bands'][0]


# save image datasets
image_datasets = {x: ImageFolderTrainDet(root=os.path.join(data_dir, x),
                                         shape_transform=data_transforms[x]['shape_transform'],
                                         int_transform=data_transforms[x]['int_transform'])
                  for x in ['training', 'validation']}


# helper function to get the (x,y) of max values
def get_xy_locs(array, count):
    if count == 0:
        return np.array([])
    cols = array.shape[1]
    flat = array.flatten()
    return np.array([[x // cols, x % cols] for x in flat.argsort()[-count:]])


# helper function to get a loss based on the square mean euclidean distance between predicted seal centroids and
# ground-truth seal centroids
def get_euc_loss(pred_locs, gt_locs):
    loss = 0
    num_pairs = min(len(pred_locs), len(gt_locs))
    n = num_pairs
    pairs = []

    for i in range(len(pred_locs)):
        for j in range(len(gt_locs)):
            pairs.append([i, j, np.linalg.norm(pred_locs[i] - gt_locs[j])])

    pairs = sorted(pairs, key=lambda x: x[2])
    while num_pairs > 0:
        i, j = pairs[0][:2]
        loss += pairs[0][2]
        pairs = [pair for pair in pairs if pair[0] != i and pair[1] != j]
        num_pairs -= 1

    return np.sqrt(loss / max(1, n))


# Force minibatches to have an equal representation amongst classes during training with a weighted sampler
def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


# For unbalanced dataset we create a weighted sampler
weights = make_weights_for_balanced_classes(image_datasets['training'].imgs, len(image_datasets['training'].classes))
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))


# change batch size ot match number of GPU's being used?
dataloaders = {"training": torch.utils.data.DataLoader(image_datasets["training"],
                                                       batch_size=
                                                       hyperparameters[args.hyperparameter_set]['batch_size_train'],
                                                       sampler=sampler, num_workers=
                                                       hyperparameters[args.hyperparameter_set]['num_workers_train']),
               "validation": torch.utils.data.DataLoader(image_datasets["validation"],
                                                         batch_size=
                                                         hyperparameters[args.hyperparameter_set]['batch_size_val'],
                                                         num_workers=
                                                         hyperparameters[args.hyperparameter_set]['num_workers_val'])
               }
dataset_sizes = {x: len(image_datasets[x]) for x in ['training', 'validation']}

use_gpu = torch.cuda.is_available()


def train_model(model, criterion1, criterion2, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # create summary writer for tensorboardX
    writer = SummaryWriter()
    # keep track of training iterations
    global_step = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            if phase == 'training':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for iter, data in enumerate(dataloaders[phase]):
                # get the inputs
                inputs, _, counts, locations = data
                counts.type(torch.int)

                # get location indices
                locs = [get_xy_locs(loc, int(counts[idx])) for idx, loc in enumerate(locations.numpy())]

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    counts = Variable(counts.cuda())
                    locations = Variable(locations.cuda())
                else:
                    inputs, counts, locations = Variable(inputs), Variable(counts), Variable(locations)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                pred_cnt, pred_loc = model(inputs)

                # flatten locations and predicted locations
                locations = locations.view(-1)
                pred_loc_flat = F.sigmoid(pred_loc).view(-1)

                # get euclidean loss
                pred_loc = pred_loc.cpu().detach()

                # find predicted location
                pred_loc = [get_xy_locs(loc, max(0, int(pred_cnt[idx]))) for idx, loc in enumerate(pred_loc.numpy())]

                # get euclidean loss
                euc_loss = sum([get_euc_loss(pred_loc[idx], locs[idx]) for idx in range(len(locs))]) / 25

                # get counting loss
                pred_cnt = pred_cnt.cuda()

                cnt_loss = criterion1(pred_cnt, counts)
                hm_loss = criterion2(pred_loc_flat, locations) * max(1, euc_loss)
                if iter % 200 == 0:
                    print('\n {} training iterations'.format(iter))
                    print('   Hubber loss: {}'.format(cnt_loss.item()))
                    print('   Euclidean loss: {}'.format(euc_loss))
                    print('   BCE loss: {}'.format(hm_loss.item()))
                    print('   total loss: {}'.format(hm_loss.item() + cnt_loss.item()))

                # backward + optimize only if in training phase
                if phase == 'training':
                    cnt_loss.backward(retain_graph=True)
                    hm_loss.backward()
                    optimizer.step()
                    global_step += 1

                # statistics
                running_loss += (hm_loss.item() + cnt_loss.item()) * inputs.size(1)

            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == 'validation':
                writer.add_scalar('validation_loss', epoch_loss, global_step=global_step)

            else:
                writer.add_scalar('training_loss', epoch_loss, global_step=global_step)
                writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], global_step=global_step)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'validation':
                time_elapsed = time.time() - since
                print('training time: {}h {:.0f}m {:.0f}s\n'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60,
                                                                    time_elapsed % 60))

    time_elapsed = time.time() - since
    print('Training complete in {}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))

    # save the model, keeping haulout and single seal models in separate folders
    torch.save(model.state_dict(), './{}/{}/{}/{}.tar'.format(args.dest_folder, args.pipeline,
                                                              args.output_name, args.output_name))

    return model


def main():
    model_ft = model_defs[args.pipeline][args.model_architecture]

    # get weight
    cv_weight = cv_weights[args.cv_weights](1)

    # define criterion
    criterion = loss_functions[args.pipeline](cv_weight)
    criterion2 = nn.BCELoss()

    if use_gpu:
        # i think we can set parallel GPU usage here. will test
        # http://pytorch.org/docs/master/nn.html
        # http://pytorch.org/docs/master/nn.html#dataparallel-layers-multi-gpu-distributed
        # The batch size should be larger than the number of GPUs used.
        # It should also be an integer multiple of the number of GPUs so that
        # each chunk is the same size (so that each GPU processes the same number of samples).
        # model_ft = nn.DataParallel(model_ft).cuda()
        model_ft = model_ft.cuda()
        criterion = criterion.cuda()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=hyperparameters[args.hyperparameter_set]['learning_rate'])

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=hyperparameters[args.hyperparameter_set]['step_size']
                                           , gamma=hyperparameters[args.hyperparameter_set]['gamma'])

    # start training
    train_model(model_ft, criterion, criterion2, optimizer_ft, exp_lr_scheduler,
                num_epochs=hyperparameters[args.hyperparameter_set]['epochs'])


if __name__ == '__main__':
    main()




