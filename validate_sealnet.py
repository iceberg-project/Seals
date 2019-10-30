"""
Train sealnet
==========================================================

CNN validation script for ICEBERG seals use case.

Author: Bento Goncalves
License: MIT
Copyright: 2018-2019
"""

import argparse
import os
import time
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import ImageFile
from torch.autograd import Variable
from torchvision import transforms

from utils.dataloaders.data_loader_train_det import ImageFolderTrainDet
from utils.dataloaders.transforms_det import ShapeTransform
from utils.model_library import *
from utils.getxy_max import getxy_max

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
    'validation': {'shape_transform': ShapeTransform(arch_input_size, train=False),
                   'int_transform': transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize([0.5], [0.225])])},
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
                  for x in ['validation']}

# change batch size ot match number of GPU's being used?
dataloaders = {"validation": torch.utils.data.DataLoader(image_datasets["validation"],
                                                         batch_size=
                                                         hyperparameters[args.hyperparameter_set]['batch_size_val'],
                                                         num_workers=
                                                         hyperparameters[args.hyperparameter_set]['num_workers_val'])
               }
dataset_sizes = {'validation': len(image_datasets['validation'])}

use_gpu = torch.cuda.is_available()

sigmoid = torch.nn.Sigmoid()


def validate_model(model, criterion1, criterion2, criterion3, bce_weight):
    """
    Helper function to train CNNs. Trains detection models using heatmaps, where the output heatmap has the same
    dimensions of the input image. Heatmap detection may be assisted with a regression branch to provide counts and/or
    an occupancy branch to decide if a patch is worth counting.

    :param model: pytorch model
    :param criterion1: loss function
    :param criterion2: additional loss function
    :param criterion3: additional loss function
    :param bce_weight: weights for binary cross-entropy loss, ensure balanced weighting of underrepresented labels
    :return: val_stats: pandas DataFrame with validation stats
    """
    # keep track of running time
    since = time.time()

    # loss dictionary
    loss_dict = {'count': lambda x: criterion1(x, counts),
                 'heatmap': lambda x: criterion2(x.view(-1), locations.view(-1) * 10),
                 'occupancy': lambda x: criterion3(x, occ)}

    # run validation loop
    model.train(False)  
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
        for iter, data in enumerate(dataloaders['validation']):
            # get the inputs
            inputs, _, counts, locations = data

            # get precision and recall
            ground_truth_xy = [getxy_max(loc, int(counts[idx])) for idx, loc in enumerate(locations.numpy())]

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

            # forward
            out_dict = model(inputs)

            # get predicted locations, precision and recall
            pred_xy = [getxy_max(loc, int(round(
                out_dict['count'][idx].item()))) for idx, loc in
                        enumerate(out_dict['heatmap'].cpu().numpy())]

            if 'occupancy' in out_dict:
                fixed_cnt = out_dict['count'] * torch.Tensor([ele > 0.5 for ele in
                                                                sigmoid(out_dict['occupancy'])]).cuda()
                pred_xy_fixed = [getxy_max(loc, int(round(fixed_cnt[idx].item()))) for idx, loc in
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
                        pnt = np.array(pnt)
                        for pred_idx_fixed, pnt2 in enumerate(pred_xy_fixed[idx]):
                            pnt2 = np.array(pnt2)
                            if gt_idx in matched_gt_fixed:
                                continue
                            if pred_idx_fixed not in matched_fixed and np.linalg.norm(pnt - pnt2) < 3:
                                n_matches_fixed += 1
                                matched_fixed.add(pred_idx_fixed)
                                matched_gt_fixed.add(gt_idx)

                        for pred_idx, pnt2 in enumerate(pred_xy[idx]):
                            pnt2 = np.array(pnt2)
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
        epoch_loss[loss] = running_loss[loss] / dataset_sizes['validation']

    epoch_precision = true_positives / max(1, true_positives + false_positives)
    epoch_recall = true_positives / max(1, true_positives + false_negatives)
    epoch_f1 = epoch_precision * epoch_recall
    epoch_precision_fxd = true_positives_fixed / max(1, true_positives_fixed + false_positives_fixed)
    epoch_recall_fxd = true_positives_fixed / max(1, true_positives_fixed + false_negatives_fixed)
    epoch_f1_fxd = epoch_precision_fxd * epoch_recall_fxd
    total_loss = sum(epoch_loss.values())
    fixed = 0
    if epoch_f1_fxd > epoch_f1:
        fixed = 1
        epoch_f1 = epoch_f1_fxd
        epoch_recall = epoch_recall_fxd
        epoch_precision = epoch_precision_fxd

    for loss in epoch_loss:
        print('{} loss: {}'.format(loss, epoch_loss[loss]))

    # save validation stats
    val_stats = pd.DataFrame({'f1': epoch_f1,
                              'loss': total_loss,
                              'recall': epoch_recall,
                              'precision': epoch_precision,
                              'fixed': fixed},
                              index = [0])
    time_elapsed = time.time() - since

    print('Validation complete in {}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))

    return val_stats


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
        model = nn.DataParallel(model)

    # load checkpoint
    model.load_state_dict(torch.load("./{}/{}/{}/{}_best_f1.tar".format(args.models_folder, pipeline, args.output_name,
                                                                        args.output_name)))

    # start validation
    val_stats = validate_model(model, criterion, criterion2, criterion3, bce_weight=bce_weights)
    val_stats.to_csv("./{}/{}/{}/{}_stats.csv".format(args.models_folder, pipeline, args.output_name,
                                                      args.output_name))


if __name__ == '__main__':
    main()
