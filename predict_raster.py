"""
Predict sealnet
==========================================================

CNN deployment script for ICEBERG seals use case. Splits a raster into tiles and predicts seal counts and locations in
tiles with 'predict_sealnet.py'.

Author: Bento Goncalves
License: MIT
Copyright: 2018-2019
"""

import argparse
import os
import shutil
import warnings

import torch

from predict_sealnet import predict_patch
from tile_raster import tile_raster
from utils.model_library import *

warnings.filterwarnings('ignore', module='PIL')

parser = argparse.ArgumentParser(description='validates a CNN at the haul out level')
parser.add_argument('--input_image', type=str, help='base directory to recursively search for validation images in')
parser.add_argument('--model_architecture', type=str, help='model architecture for seal detection')
parser.add_argument('--hyperparameter_set', type=str, help='combination of hyperparameters used for CNNs, must be a '
                                                           'member of hyperparameters dictionary')
parser.add_argument('--training_set', type=str, help='training set where models were trained')
parser.add_argument('--stride', type=float, default=1.0,
                    help='stride for tiling (e.g. 1 = adjacent tiles, 0.5 = 50% overlap)')
parser.add_argument('--test_folder', type=str, default='to_classify', help='folder where the model will be saved')
parser.add_argument('--save_heatmaps', type=int, default=0, help='whether or not heatmaps are saved')


def main():
    # unroll arguments
    args = parser.parse_args()
    pipeline = model_archs[args.model_architecture]['pipeline']
    input_image = args.input_image
    output_folder = args.test_folder
    stride = args.stride
    scales = [model_archs[args.model_architecture]['input_size']]

    # check for pre-existing tiles and subtiles
    if os.path.exists('{}/tiles'.format(args.test_folder)):
        shutil.rmtree('{}/tiles'.format(args.test_folder))

    # tile raster into patches
    tile_raster(input_image, output_folder, scales, stride)

    # predict tiles
    model = model_defs[pipeline][args.model_architecture]

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()
    model.eval()

    # load saved model weights from training
    model_name = args.model_architecture + '_ts-' + args.training_set.split('_')[-1]
    model.load_state_dict(
        torch.load("./saved_models/{}/{}/{}.tar".format(pipeline, model_name, model_name)))

    predict_patch(model=model, input_size=model_archs[args.model_architecture]['input_size'],
                  batch_size=hyperparameters[args.hyperparameter_set]['batch_size_test'],
                  test_dir=args.test_folder,
                  output_dir='{}/{}'.format(args.test_folder, os.path.basename(input_image)[:-4]),
                  num_workers=hyperparameters[args.hyperparameter_set]['num_workers_train'],
                  save_heatmaps=args.save_heatmaps)


if __name__ == "__main__":
    main()
