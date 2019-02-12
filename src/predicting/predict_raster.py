"""
Predict sealnet
==========================================================

CNN deployment script for ICEBERG seals use case. Splits a raster into tiles and predicts seal counts and locations in
tiles with 'predict_sealnet.py'.

Author: Bento Goncalves
License: MIT
Copyright: 2018-2019
"""

import torch
import warnings
import argparse
import os
import shutil
from utils.model_library import *
from predict_sealnet import predict_patch
warnings.filterwarnings('ignore', module='PIL')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='validates a CNN at the haul out level')
    parser.add_argument('--input_image', type=str, help='base directory to recursively search for validation images in')
    parser.add_argument('--model_architecture', type=str, help='model architecture for seal detection')
    parser.add_argument('--hyperparameter_set', type=str, help='combination of hyperparameters used for CNNs, must be a '
                                                           'member of hyperparameters dictionary')
    parser.add_argument('--training_set', type=str, help='training set where models were trained')
    parser.add_argument('--test_folder', type=str, default='to_classify', help='folder where the model will be saved')
    parser.add_argument('--tile', type=bool, default=False, help='Input image needs to be tiled')
    parser.add_argument('--model_path', type=str, help='folder where the model is tarball is')
    parser.add_argument('--output_folder', type=str, help='folder where results will be saved')
    # unroll arguments
    args = parser.parse_args()
    pipeline = model_archs[args.model_architecture]['pipeline']
    input_image = args.input_image
    output_folder = args.test_folder
    scales = [model_archs[args.model_architecture]['input_size']]
    
    # check if the image needs to be tiled.
    if args.tile:
        
        from tile_raster import tile_raster

        # check for pre-existing tiles and subtiles
        if os.path.exists('%s/tiles' % (args.test_folder)):
            shutil.rmtree('%s/tiles' % (args.test_folder))

        # tile raster into patches
        tile_raster(input_image, output_folder, scales)

    # predict tiles
    model = model_defs[pipeline][args.model_architecture]

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()
    model.eval()

    # load saved model weights from training
    model_name = args.model_architecture + '_ts-' + args.training_set.split('_')[-1]
    model.load_state_dict(
        torch.load("%s/%s.tar" % (args.model_path, model_name)))

    predict_patch(input_image= args.input_image, model=model,
                  input_size=model_archs[args.model_architecture]['input_size'],
                  batch_size=hyperparameters[args.hyperparameter_set]['batch_size_test'],
                  test_dir=args.test_folder,
                  output_dir='%s' % (args.output_folder),
                  num_workers=hyperparameters[args.hyperparameter_set]['num_workers_train'])
