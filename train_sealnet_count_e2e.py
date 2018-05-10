import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import os
import argparse
from tensorboardX import SummaryWriter
import time
from utils.model_library import *
from utils.dataloaders.data_loader_train_det import ImageFolderTrainDet
from utils.dataloaders.transforms_det import ShapeTransform
from utils.custom_architectures.count_ception import ModelCountception
from utils.custom_architectures.nasnet_scalable_count import NASNetALarge
from utils.custom_architectures.nasnet_scalable_e2e import NASNetAe2e
from PIL import ImageFile
import warnings

parser = argparse.ArgumentParser(description='trains a CNN to find seals in satellite imagery')
parser.add_argument('--training_dir', type=str, help='base directory to recursively search for images in')
parser.add_argument('--model_architecture', type=str, help='model architecture, must be a member of models '
                                                           'dictionary')
parser.add_argument('--hyperparameter_set', type=str, help='combination of hyperparameters used, must be a member of '
                                                           'hyperparameters dictionary')
parser.add_argument('--cv_weights', type=str, help='weights for weighted-cross validation, must be a member of '
                                                   'cv_weights dictionary')
parser.add_argument('--output_name', type=str, help='name of output file from training, this name will also be used in '
                                                    'subsequent steps of the pipeline')
args = parser.parse_args()

# check for invalid inputs
if args.model_architecture not in model_archs:
    raise Exception("Unsupported architecture")

if args.training_dir not in training_sets:
    raise Exception("Invalid training set")

if args.cv_weights not in cv_weights:
    raise Exception("Invalid cross-validation weights")

if args.hyperparameter_set not in hyperparameters:
    raise Exception("Invalid hyperparameter combination")

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


def train_model(model,  optimizer, scheduler, criterion_class, num_epochs=2, criterion_count=None):
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
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels, counts = data

                # create tensorboard variables
                counts.type(torch.FloatTensor)

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    counts = Variable(counts.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels, counts = Variable(inputs), Variable(labels), Variable(counts)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                out_class, out_count = model(inputs)
                preds = torch.max(out_class.data, 1)[1]
                out_count = torch.Tensor([ele for ele in out_count])
                out_count = Variable(out_count, requires_grad=True)
                if use_gpu:
                    out_count = out_count.cuda()
                    out_class = out_class.cuda()

                class_loss = criterion_class(out_class, labels)
                count_loss = criterion_count(out_count, counts)

                loss = class_loss + count_loss

                # backward + optimize only if in training phase
                if phase == 'training':
                    loss.backward()
                    optimizer.step()
                    global_step += 1

                # statistics
                running_loss += loss.item() * inputs.size(1)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            if phase == 'validation':
                writer.add_scalar('validation_loss', epoch_loss, global_step=global_step)
                writer.add_scalar('validation_accuracy', epoch_acc, global_step=global_step)

            else:
                writer.add_scalar('training_loss', epoch_loss, global_step=global_step)
                writer.add_scalar('training_accuracy', epoch_acc, global_step=global_step)
                writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], global_step=global_step)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'validation':
                time_elapsed = time.time() - since
                print('training time: {}h {:.0f}m {:.0f}s\n'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60,
                                                                    time_elapsed % 60))

    time_elapsed = time.time() - since
    print('Training complete in {}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))

    # save the model, keeping haulout and single seal models in separate folders
    if model_archs[args.model_architecture]['haulout']:
        torch.save(model.state_dict(), 'saved_models/haulout/{}/{}.tar'.format(args.output_name, args.output_name))
    else:
        torch.save(model.state_dict(), 'saved_models/single_seal/{}/{}.tar'.format(args.output_name, args.output_name))

    return model


def main():

    if args.model_architecture == 'CountCeption':
        model = ModelCountception()

    elif args.model_architecture == 'NasnetACount':
        model = NASNetALarge(in_channels_0=48, out_channels_0=24, out_channels_1=32, out_channels_2=64,
                             out_channels_3=128, num_classes=11)
        # load weights from classification
        model.load_state_dict(
            torch.load("./saved_models/haulout/model5/model5.tar"))
        # replace last linear
        model.last_linear = nn.Linear(128*12, 1)

    elif args.model_architecture == 'NasnetAe2e':
        model = NASNetAe2e(in_channels_0=48, out_channels_0=24, out_channels_1=32, out_channels_2=64,
                           out_channels_3=128, num_classes=9)

    # define criterion for counting
    criterion_class = nn.CrossEntropyLoss(weight=torch.FloatTensor(cv_weights[args.cv_weights]))
    criterion_count = nn.MSELoss()

    if use_gpu:
        # i think we can set parallel GPU usage here. will test
        # http://pytorch.org/docs/master/nn.html
        # http://pytorch.org/docs/master/nn.html#dataparallel-layers-multi-gpu-distributed
        # The batch size should be larger than the number of GPUs used.
        # It should also be an integer multiple of the number of GPUs so that
        # each chunk is the same size (so that each GPU processes the same number of samples).
        # model_ft = nn.DataParallel(model_ft).cuda()
        model = model.cuda()
        criterion_class = criterion_class.cuda()
        criterion_count = criterion_count.cuda()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=hyperparameters[args.hyperparameter_set]['learning_rate'])

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=hyperparameters[args.hyperparameter_set]['step_size']
                                           , gamma=hyperparameters[args.hyperparameter_set]['gamma'])

    # start training
    train_model(model, optimizer_ft, exp_lr_scheduler, criterion_class=criterion_class,
                criterion_count=criterion_count,
                num_epochs=hyperparameters[args.hyperparameter_set]['epochs'])


if __name__ == '__main__':
    main()




