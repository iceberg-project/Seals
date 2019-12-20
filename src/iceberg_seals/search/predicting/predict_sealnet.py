"""
Predict sealnet
==========================================================

CNN prediction script for ICEBERG seals use case. Predicts seal locations and counts for tiles in input folder.

Author: Bento Goncalves
License: MIT
Copyright: 2018-2019
"""

import argparse
import os
import shutil
import time
import warnings

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
# image transforms seem to cause truncated images, so we need this
from PIL import ImageFile
from fiona.crs import from_epsg
from shapely.geometry.geo import box, Point
from torchvision import transforms

from ..utils.dataloaders.data_loader_test import ImageFolderTest
from ..utils.model_library import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings('ignore', module='PIL')


def parse_args():
    parser = argparse.ArgumentParser(description='validates a CNN at the haul out level')
    parser.add_argument('--test_dir', type=str, help='base directory to recursively search for validation images in')
    parser.add_argument('--model_architecture', type=str, help='model architecture, must be a member of models '
                                                               'dictionary')
    parser.add_argument('--hyperparameter_set', type=str, help='combination of hyperparameters used, must be a member '
                                                               'of hyperparameters dictionary')
    parser.add_argument('--model_name', type=str, help='name of input model file from training, this name will also be '
                                                       'used in subsequent steps of the pipeline')
    parser.add_argument('--models_folder', type=str, default='saved_models', help='folder where the model tar file is'
                                                                                  'saved')
    parser.add_argument('--output_dir', type=str, help='folder where output shapefiles will be stored')
    parser.add_argument('--save_heatmaps', type=int, help='boolean for saving heatmaps', default=0)
    return parser.parse_args()


# helper function to get the (x,y) of max values
def get_xy_locs(array, count, min_dist=3):
    """
    Helper function to locate a predefined number of intensity peaks on a heatmap. Uses abination of dilation
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
    # flatten array, get rid of zeros and sort i
    flat = array.flatten()
    flat_order = (-flat).argsort()
    # find first zero and remove tail
    flat_order = flat_order[:next((idx for idx, ele in enumerate(flat_order) if not flat[ele]), None)]
    # check if detections are too close
    to_remove = []
    #for idx, ele in enumerate(flat_order):
    #    if idx in to_remove:
    #        continue
    #    for idx2 in range(idx + 1, len(flat_order)):
    #        if np.linalg.norm(np.array([flat_order[idx] // cols, flat_order[idx] % cols]) -
    #                          np.array([flat_order[idx2] // cols, flat_order[idx2] % cols])) < min_dist:
    #            to_remove.append(idx2)
    flat_order = np.delete(flat_order, to_remove)
    # return x peaks
    return [(x // cols, x % cols) for x in flat_order[:count]]


def predict_patch(model, output_dir, test_dir, batch_size=2, input_size=299, threshold=0, num_workers=1,
                  duplicate_tolerance=1.5, remove_tiles=False, save_heatmaps=True):
    """
    Patch prediction function. Outputs shapefiles for counts and locations.

    :param model: pytorch model
    :param test_dir: directory with input tiles
    :param output_dir: output directory name
    :param batch_size: number of images per mini-batch
    :param input_size: size of input images
    :param threshold: threshold for occupancy
    :param num_workers: number of workers on dataloader
    :param duplicate_tolerance: proximity at which two points are considered duplicates, used for NMS
    :param remove_tiles: Remove the tiles folder from the filesystem.
    :param save_heatmaps: Saves heatmaps from output
    :return:
    """

    # create output folder for heatmaps if needed
    if save_heatmaps:
        os.makedirs(output_dir + '/heatmaps', exist_ok=True)

    # crop and normalize images
    data_transforms = transforms.Compose([
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.225])
    ])

    # load dataset
    dataset = ImageFolderTest(test_dir, data_transforms)

    # separate into batches with dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    # check for GPU support
    use_gpu = torch.cuda.is_available()

    # store predictions and filenames
    predicted_locs = []
    intensity = []
    fnames = []

    # set training flag to False
    model.train(False)

    # keep track of running time
    since = time.time()
    sigmoid = torch.nn.Sigmoid()

    with torch.no_grad():

        for data in dataloader:

            # get the inputs
            inputs, filenames = data

            # gpu support
            if use_gpu:
                inputs = inputs.cuda()

            # do a forward pass to get predictions
            # detection models

            out_dict = model(inputs)
            counts = out_dict['count']
            if 'occupancy' in out_dict:
                counts = counts * torch.Tensor([ele > threshold for ele in sigmoid(out_dict['occupancy'])]).cuda()
            counts = counts.round().int()
            pred_cnt_batch = [count.item() for count in counts]
            
            # find seals if count > 0
            if sum(pred_cnt_batch) > 0:
                
                locs = out_dict['heatmap'].cpu().detach()
                locs = locs.numpy()

                # save heatmap
                if save_heatmaps:
                    for idx, loc in enumerate(locs):
                        if pred_cnt_batch[idx] > 0:
                            loc = (loc - np.min(loc)) / (np.max(loc) - np.min(loc))
                            loc = np.vstack([np.zeros([1, input_size, input_size]), loc.reshape(1, input_size, input_size),
                                            np.zeros([1, input_size, input_size])]) * 255
                            cv2.imwrite(output_dir + '/heatmaps/' + filenames[idx], loc.transpose(1, 2, 0))

                # find predicted location
                points = []
                for idx, loc in enumerate(locs):
                    if pred_cnt_batch[idx] > 0:
                        loc = (loc - np.min(loc)) / (np.max(loc) - np.min(loc))
                        curr_points = get_xy_locs(loc, pred_cnt_batch[idx])
                        for batch in curr_points:
                            intensity.append(loc[0, batch[0], batch[1]]) 
                            predicted_locs.append(batch)
                            fnames.append(filenames[idx])
                

    pred_locations = {'x': [pnt[0] for pnt in predicted_locs],
                      'y': [pnt[1] for pnt in predicted_locs],
                      'filenames': [fname for fname in fnames],
                      'intensity': [val for val in intensity]}

    #for idx, batch in enumerate(predicted_locs):
    #    for pnt in batch:
    #        pred_locations['x'].append(pnt[0])
    #        pred_locations['y'].append(pnt[1])
    #        pred_locations['filenames'].append(fnames[idx])
    #        pred_locations['intensity'].append(intensity[idx])
    #        pnt_idx += 1

    pred_locations = pd.DataFrame(pred_locations)

    # save shapefile for counts / classes
    shapefile_path = '%s/predicted_shapefiles/' % output_dir
    if not os.path.exists(shapefile_path):
        os.makedirs(shapefile_path)

    # load affine matrix
    affine_matrix = rasterio.Affine(*[ele for ele in pd.read_csv(
        '%s/affine_matrix.csv' % (test_dir))['transform']])

    # create geopandas DataFrames to store counts per patch and seal locations
    output_shpfile_locs = gpd.GeoDataFrame()

    # setup projection for output
    output_shpfile_locs.crs = from_epsg(3031)

    if len(pred_locations) > 0:
        
        # order locations by heatmap intensity
        pred_locations = pred_locations.iloc[np.argsort(pred_locations['intensity'] * -1)]

        # add locations
        for _, row in pred_locations.iterrows():
            fname = row['filenames']
            intensity = row['intensity']
            up, left, down, right = [int(ele) for ele in fname.split('_')[-5: -1]]
            x, y = [up + row['y'], left + row['x']]
            x_loc, y_loc = [x, y] * affine_matrix

            # keep point if there isn't any duplicate with higher intensity
            keep = True
            for _, row2 in output_shpfile_locs.iterrows():
                if row2['geometry'].distance(Point(x_loc, y_loc)) < duplicate_tolerance:
                    keep = False
                    break
            if keep:
                output_shpfile_locs = output_shpfile_locs.append(pd.Series({'geometry': Point(x_loc, y_loc),
                                                                            'x': x, 'y': y,
                                                                            'filename': fname,
                                                                            'intensity': intensity}),
                                                                 ignore_index=True)

        # add scene name
        output_shpfile_locs = output_shpfile_locs.join(pd.DataFrame({
            'scene': [os.path.basename(test_dir)] * len(output_shpfile_locs)}))

        # save shapefile
        output_shpfile_locs.to_file(shapefile_path + 'locations.shp'.format(
            os.path.basename(output_dir)))

    # remove tiles
    if remove_tiles:
        shutil.rmtree('{}/tiles'.format(test_dir))

    time_elapsed = time.time() - since
    print('Testing complete in %dh %dm %ds' % (time_elapsed // 3600,
                                               time_elapsed // 60,
                                               time_elapsed % 60))
    print('Total predicted in %s: ' % os.path.basename(test_dir), len(output_shpfile_locs))


def main():
    # load arguments
    args = parse_args()

    # check for invalid inputs
    assert args.model_architecture in model_archs, "Invalid architecture -- see supported" \
                                                   " architectures: %s" % list(model_archs.keys())

    assert args.hyperparameter_set in hyperparameters, "Hyperparameter combination is not defined in " \
                                                       "./utils/model_library.py"

    # find pipeline
    pipeline = model_archs[args.model_architecture]['pipeline']

    # create model instance
    model = model_defs[pipeline][args.model_architecture]

    # check for GPU support and set model to evaluation mode
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()
    model.eval()

    # load saved model weights from pt_train.py
    model.load_state_dict(torch.load("./{}/{}/{}/{}.tar".format(args.models_folder, pipeline, args.model_name,
                                                                args.model_name)))

    # run validation to get confusion matrix
    predict_patch(model=model, input_size=model_archs[args.model_architecture]['input_size'],
                  test_dir=args.test_dir, output_dir=args.output_dir,
                  batch_size=hyperparameters[args.hyperparameter_set]['batch_size_test'],
                  num_workers=hyperparameters[args.hyperparameter_set]['num_workers_train'],
                  remove_tiles=True)


if __name__ == '__main__':
    main()
