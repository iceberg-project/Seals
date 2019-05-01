"""
Train sealnet
==========================================================

CNN training script for ICEBERG seals use case.

Author: Bento Goncalves
License: MIT
Copyright: 2018-2019
"""

import argparse
import datetime
import os
import shutil
import time
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms

from utils.dataloaders.data_loader_train_det import ImageFolderTrainDet
from utils.dataloaders.transforms_det import ShapeTransform
from utils.model_library import *

parser = argparse.ArgumentParser(description='trains a CNN to find seals in satellite imagery')
parser.add_argument('--training_dir', type=str, help='base directory to recursively search for images in')
parser.add_argument('--model_architecture', type=str, help='model architecture, must be a member of models '
                                                           'dictionary')
parser.add_argument('--hyperparameter_set', type=str, help='combination of hyperparameters used, must be a member of '
                                                           'hyperparameters dictionary')
parser.add_argument('--output_name', type=str, help='name of output file from training, this name will also be used in '
                                                    'subsequent steps of the pipeline')
parser.add_argument('--models_folder', type=str, default='saved_models', help='folder where the model will be saved')

args = parser.parse_args()

# define pipeline
pipeline = model_archs[args.model_architecture]['pipeline']

# check for invalid inputs
if args.model_architecture not in model_archs:
    raise Exception("Invalid architecture -- see supported architectures:  {}".format(list(model_archs.keys())))

if args.training_dir not in training_sets:
    raise Exception("Training set is not defined in ./utils/model_library.py")

if args.hyperparameter_set not in hyperparameters:
    raise Exception("Hyperparameter combination is not defined in ./utils/model_library.py")

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
                                         int_transform=data_transforms[x]['int_transform'],
                                         training_set=args.training_dir,
                                         shuffle=x == 'training')
                  for x in ['training', 'validation']}


def get_xy_locs(array, count, min_dist=3):
    """
    Helper function to locate a predefined number of intensity peaks on a heatmap. Uses a combination of dilation
    filters to find cells in the image that remain unchanged between the original image and dilated filters -- which
    should be intensity peaks. The function returns a numpy array with the (x,y) locations of the 'n' highest intensity
    peaks, with a minimum distance of 'm' between peaks.

    :param array:np.array array of points where we desire to get intensity peaks
    :param count: number of intensity peaks
    :param min_dist: minimum distance between intensity peaks
    :return: array with (x,y) of each intensity peak
    """
    if count == 0:
        return np.array([])
    cols = array.shape[1]
    # dilate (2 dilations are more robust than one)
    dil_array = cv2.dilate(array, np.ones([3, 3], dtype=np.uint8))
    dil_array2 = cv2.dilate(array, np.ones([5, 5], dtype=np.uint8))
    # check indices that do not change with dilations
    array = array * (cv2.compare(array, dil_array, 2) == 255) * (cv2.compare(array, dil_array2, 2) == 255)
    # flatten array, get rid of zeros and sort it
    flat = array.flatten()
    flat_order = (-flat).argsort()
    # find first zero and remove tail
    flat_order = flat_order[:next((idx for idx, ele in enumerate(flat_order) if not flat[ele]), None)]
    # check if detections are too close
    to_remove = []
    for idx, ele in enumerate(flat_order):
        if idx in to_remove:
            continue
        for idx2 in range(idx + 1, len(flat_order)):
            if np.linalg.norm(np.array([flat_order[idx] // cols, flat_order[idx] % cols]) -
                              np.array([flat_order[idx2] // cols, flat_order[idx2] % cols])) < min_dist:
                to_remove.append(idx2)
    flat_order = np.delete(flat_order, to_remove)
    # return x peaks
    return np.array([(x // cols, x % cols) for x in flat_order[:count]])


# Force minibatches to have an equal representation amongst classes during training with a weighted sampler
def make_weights_for_balanced_classes(images, nclasses):
    """
    Generates weights to get balanced classes during training. To be used with weighted random samplers.

    :param images: list of training images in training set.
    :param nclasses: number of classes on training set.
    :return: list of weights for each training image.
    """
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

sigmoid = torch.nn.Sigmoid()


def save_checkpoint(state, is_best_loss, is_best_f1, is_best_recall, is_best_precision):
    """
    Saves model checkpoints during training. At the end of each validation cycle, model checkpoints are stored if they
    surpass previous best scores at a number of validation metrics (i.e. loss, F-1 score, precision and recall).
    Alongside the model state from the latest epoch, one model state is kept for each validation metric.

    :param state: pytorch model state, with parameter values for a giv
    :param is_best_loss: boolean for whether the current state beats the lowest loss
    :param is_best_f1: boolean for whether the current state beats the highest F-1 score
    :param is_best_recall: boolean for whether the current state beats the highest recall
    :param is_best_precision: boolean for whether the current state beats the highest precision
    :return:
    """
    filename = './{}/{}/{}/{}'.format(args.models_folder, pipeline, args.output_name, args.output_name)
    torch.save(state, filename + '.tar')
    if is_best_loss:
        shutil.copyfile(filename + '.tar', filename + '_best_loss.tar')
    if is_best_f1:
        shutil.copyfile(filename + '.tar', filename + '_best_f1.tar')
    if is_best_recall:
        shutil.copyfile(filename + '.tar', filename + '_best_recall.tar')
    if is_best_precision:
        shutil.copyfile(filename + '.tar', filename + '_best_precision.tar')


def train_model(model, criterion1, criterion2, criterion3, optimizer, scheduler, bce_weight, num_epochs=25):
    """
    Helper function to train CNNs. Trains detection models using heatmaps, where the output heatmap has the same
    dimensions of the input image. Heatmap detection may be assisted with a regression branch to provide counts and/or
    an occupancy branch to decide if a patch is worth counting.

    :param model: pytorch model
    :param criterion1: loss function
    :param criterion2: additional loss function
    :param criterion3: additional loss function
    :param optimizer: optimizer
    :param scheduler: training scheduler to deal with weight decay during training
    :param bce_weight: weights for binary cross-entropy loss, ensure balanced weighting of underrepresented labels
    :param num_epochs: number of training epochs
    :return:
    """
    # keep track of running time
    since = time.time()

    # create summary writer for tensorboardX
    writer = SummaryWriter(log_dir='./tensorboard_logs/{}_{}'.format(args.output_name, str(datetime.datetime.now())))

    # keep track of training iterations
    global_step = 0

    # keep track of best accuracy
    best_loss = 1000000000
    best_f1 = 0
    best_recall = 0
    best_precision = 0

    # loss dictionary
    loss_dict = {'count': lambda x: criterion1(x, counts),
                 'heatmap': lambda x: criterion2(x.view(-1), locations.view(-1)),
                 'occupancy': lambda x: criterion3(x, occ)}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            print('\n{} \n'.format(phase))
            if phase == 'training':
                if epoch % 20 == 0:
                    scheduler.step()
                model.train(True)  # Set model to training mode

                running_loss = {'count': 0,
                                'heatmap': 0,
                                'occupancy': 0}

                # Iterate over data.
                for iter, data in enumerate(dataloaders[phase]):

                    # get the inputs
                    inputs, _, counts, locations = data

                    # get occupancy
                    occ = torch.Tensor([cnt > 0 for cnt in counts]).cuda()

                    # get batch weights
                    batch_weights_loc = ((locations.view(-1) * bce_weight[0]) + 1).cuda()
                    batch_weights_occ = (occ * bce_weight[1] + 1).cuda()

                    # wrap them in Variable
                    if use_gpu:
                        inputs = Variable(inputs.cuda())
                        counts = Variable(counts.cuda())
                        locations = Variable(locations.cuda())
                        occ = Variable(occ.cuda())
                    else:
                        inputs, counts, occ, locations = Variable(inputs), Variable(counts), Variable(occ), \
                                                         Variable(locations)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward pass to get prediction dictionary
                    out_dict = model(inputs)

                    # load bce weights
                    criterion2.weight = batch_weights_loc
                    criterion3.weight = batch_weights_occ

                    # get losses
                    batch_loss = {}
                    for key in out_dict:
                        batch_loss[key] = loss_dict[key](out_dict[key])
                        running_loss[key] += batch_loss[key].item() * len(occ)

                    # add losses up
                    loss = 0
                    for ele in batch_loss:
                        if out_dict[ele].requires_grad:
                            loss += batch_loss[ele]

                    loss.backward()

                    # clip gradients
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                    # step with optimizer
                    optimizer.step()

                    global_step += 1

            else:
                model.train(False)  # Set model to evaluate mode
                with torch.no_grad():
                    running_loss = {'count': 0,
                                    'heatmap': 0,
                                    'occupancy': 0}
                    false_negatives = 0
                    false_positives = 0
                    true_positives = 0
                    false_negatives_fixed = 0
                    false_positives_fixed = 0
                    true_positives_fixed = 0

                    # Iterate over data.
                    for iter, data in enumerate(dataloaders[phase]):
                        # get the inputs
                        inputs, _, counts, locations = data

                        # get precision and recall
                        ground_truth_xy = [get_xy_locs(loc, int(counts[idx])) for idx, loc in
                                           enumerate(locations.numpy())]

                        # get occupancy
                        occ = torch.Tensor([cnt > 0 for cnt in counts]).cuda()

                        # get batch weights
                        batch_weights_loc = ((locations.view(-1) * bce_weight[0]) + 1).cuda()
                        batch_weights_occ = (occ * bce_weight[1] + 1).cuda()

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
                        out_dict = model(inputs)

                        # get predicted locations, precision and recall
                        pred_xy = [get_xy_locs(loc, int(round(
                            out_dict['count'][idx].item()))) for idx, loc in
                                   enumerate(out_dict['heatmap'].cpu().numpy())]

                        if 'occupancy' in out_dict:
                            fixed_cnt = out_dict['count'] * torch.Tensor([ele > 0.5 for ele in
                                                                          sigmoid(out_dict['occupancy'])]).cuda()
                            pred_xy_fixed = [get_xy_locs(loc, int(round(fixed_cnt[idx].item()))) for idx, loc in
                                             enumerate(out_dict['heatmap'].cpu().numpy())]
                        else:
                            pred_xy_fixed = pred_xy

                        for idx, ele in enumerate(ground_truth_xy):
                            n_matches = 0
                            n_matches_fixed = 0
                            if len(ele) == 0:
                                false_positives += len(pred_xy[idx])
                                false_positives_fixed += len(pred_xy_fixed[idx])
                            else:
                                matched_gt = set([])
                                matched_pred = set([])
                                matched_gt_fixed = set([])
                                matched_fixed = set([])

                                for gt_idx, pnt in enumerate(ele):
                                    for pred_idx_fixed, pnt2 in enumerate(pred_xy_fixed[idx]):
                                        if gt_idx in matched_gt_fixed:
                                            continue
                                        if pred_idx_fixed not in matched_fixed and np.linalg.norm(pnt - pnt2) < 3:
                                            n_matches_fixed += 1
                                            matched_fixed.add(pred_idx_fixed)
                                            matched_gt_fixed.add(gt_idx)

                                    for pred_idx, pnt2 in enumerate(pred_xy[idx]):
                                        if gt_idx in matched_gt:
                                            continue
                                        if pred_idx not in matched_pred and np.linalg.norm(pnt - pnt2) < 3:
                                            n_matches += 1
                                            matched_pred.add(pred_idx)
                                            matched_gt.add(gt_idx)

                                true_positives += n_matches
                                false_positives += len(pred_xy[idx]) - n_matches
                                false_negatives += len(ele) - n_matches
                                true_positives_fixed += n_matches_fixed
                                false_positives_fixed += len(pred_xy_fixed[idx]) - n_matches_fixed
                                false_negatives_fixed += len(ele) - n_matches_fixed

                        # load bce weights
                        criterion2.weight = batch_weights_loc
                        criterion3.weight = batch_weights_occ

                        # get losses
                        batch_loss = {}
                        for key in out_dict:
                            batch_loss[key] = loss_dict[key](out_dict[key])
                            running_loss[key] += batch_loss[key].item() * len(occ)

            # get epoch losses
            epoch_loss = {'heatmap': 0,
                          'count': 0,
                          'occupancy': 0}

            for loss in running_loss:
                epoch_loss[loss] = running_loss[loss] / dataset_sizes[phase]

            if phase == 'validation':
                epoch_precision = true_positives / max(1, true_positives + false_positives)
                epoch_recall = true_positives / max(1, true_positives + false_negatives)
                epoch_f1 = epoch_precision * epoch_recall
                epoch_precision_fxd = true_positives_fixed / max(1, true_positives_fixed + false_positives_fixed)
                epoch_recall_fxd = true_positives_fixed / max(1, true_positives_fixed + false_negatives_fixed)
                epoch_f1_fxd = epoch_precision_fxd * epoch_recall_fxd
                for loss in epoch_loss:
                    writer.add_scalar('validation_loss_{}'.format(loss), epoch_loss[loss], global_step=global_step)
                writer.add_scalar('validation_precision', epoch_precision, global_step=global_step)
                writer.add_scalar('validation_recall', epoch_recall, global_step=global_step)
                writer.add_scalar('validation_f1', epoch_f1, global_step=global_step)
                if 'occupancy' in out_dict:
                    writer.add_scalar('validation_precision_fixed', epoch_precision_fxd, global_step=global_step)
                    writer.add_scalar('validation_recall_fixed', epoch_recall_fxd, global_step=global_step)
                    writer.add_scalar('validation_f1_fixed', epoch_f1_fxd, global_step=global_step)
                total_loss = sum(epoch_loss.values())
                is_best_loss = total_loss < best_loss
                epoch_f1 = max(epoch_f1, epoch_f1_fxd)
                is_best_f1 = epoch_f1 > best_f1
                is_best_precision = epoch_precision > best_precision
                is_best_recall = epoch_recall > best_recall
                best_loss = min(total_loss, best_loss)
                best_f1 = max(epoch_f1, best_f1)
                save_checkpoint(model.state_dict(), is_best_loss, is_best_f1, is_best_recall, is_best_precision)

            else:
                for loss in epoch_loss:
                    writer.add_scalar('training_loss_{}'.format(loss), epoch_loss[loss], global_step=global_step)
                writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], global_step=global_step)

            for loss in epoch_loss:
                print('{} loss: {}'.format(loss, epoch_loss[loss]))

            if phase == 'validation':
                time_elapsed = time.time() - since
                print('training time: {}h {:.0f}m {:.0f}s\n'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60,
                                                                    time_elapsed % 60))
    time_elapsed = time.time() - since
    print('Training complete in {}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))


def main():
    # set seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = model_defs[pipeline][args.model_architecture]

    # define criterion
    criterion = nn.SmoothL1Loss()
    criterion2 = nn.BCEWithLogitsLoss()
    criterion3 = nn.BCEWithLogitsLoss()

    # find BCE weight
    bce_weights = [arch_input_size ** 2 * (86514 / 232502), 11 / 2]

    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()
        criterion2 = criterion2.cuda()
        criterion3 = criterion3.cuda()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=hyperparameters[args.hyperparameter_set]['learning_rate'])

    # Decay LR by a factor of 0.5 every 20 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=hyperparameters[args.hyperparameter_set]['step_size']
                                           , gamma=hyperparameters[args.hyperparameter_set]['gamma'])

    # start training
    train_model(model, criterion, criterion2, criterion3, optimizer_ft, exp_lr_scheduler,
                num_epochs=hyperparameters[args.hyperparameter_set]['epochs'], bce_weight=bce_weights)


if __name__ == '__main__':
    main()
