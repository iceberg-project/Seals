import torch
import pandas as pd
import geopandas as gpd
import warnings
import argparse
from shapely.geometry.geo import box, Point
from fiona.crs import from_epsg
import cv2
import os
import shutil
import numpy as np
from utils.model_library import *
import rasterio
from predict_sealnet import predict_patch

# image transforms seem to cause truncated images, so we need this
from PIL import Image

warnings.filterwarnings('ignore', module='PIL')

parser = argparse.ArgumentParser(description='validates a CNN at the haul out level')
parser.add_argument('--input_image', type=str, help='base directory to recursively search for validation images in')
parser.add_argument('--count_architecture', type=str, nargs='?', help='model architecture for counting, must be a '
                                                                      'member of models dictionary')
parser.add_argument('--det_architecture', type=str, nargs='?', help='model architecture for detecting, must be a '
                                                                    'member of models dictionary')
parser.add_argument('--hyperparameter_set_count', type=str, help='combination of hyperparameters used for the '
                                                                 'counting model, must be a member of '
                                                                 'hyperparameters dictionary')
parser.add_argument('--training_dir', type=str, help='directory where models were trained')
parser.add_argument('--dest_folder', type=str, default='to_classify', help='folder where the model will be saved')


def main():
    # unroll arguments
    args = parser.parse_args()
    input_image = args.input_image
    output_folder = args.dest_folder
    scales = [model_archs[args.det_architecture]['input_size']] * 3
    output_folder = './{}/tiles/images/'.format(output_folder)

    # check for pre-existing tiles and subtiles
    if os.path.exists('./{}/tiles'.format(args.dest_folder)):
        shutil.rmtree('./{}/tiles'.format(args.dest_folder))

    if os.path.exists('./{}/sub-tiles'.format(args.dest_folder)):
        shutil.rmtree('./{}/sub-tiles'.format(args.dest_folder))

    print('\nPredicting with {}:'.format(os.path.basename(args.dest_folder)))

    # create geopandas DataFrame to store classes and counts per patch
    output_shpfile = gpd.GeoDataFrame()

    # setup projection for output
    output_shpfile.crs = from_epsg(3031)

    # get affine matrix for raster file
    with rasterio.open(input_image) as src:
        affine_matrix = src.transform
        if affine_matrix == rasterio.Affine(1, 0, 0, 0, 1, 0):
            affine_matrix = rasterio.Affine(1, 0, 0, 0, -1, 0)

    # generate empty rows
    fnames = [ele for ele in os.listdir('./{}/tiles/images/'.format(args.dest_folder))]
    for fname in fnames:
        up, left, down, right = [int(ele) for ele in fname.split('_')[-5: -1]]
        coords = [point * affine_matrix for point in [[down, left], [down, right], [up, left], [up, right]]]
        output_shpfile = output_shpfile.append(pd.Series({'label': 'open-water',
                                                          'geometry': box(minx=min([point[0] for point in coords]),
                                                                          miny=min([point[1] for point in coords]),
                                                                          maxx=max([point[0] for point in coords]),
                                                                          maxy=max([point[1] for point in coords])),
                                                          'count': 0}, name=fname))

    # find class names
    class_names = sorted([subdir for subdir in os.listdir('./training_sets/{}/training'.format(args.training_dir))])

    # count inside subpatches with counting CNN or detection CNN,
    if args.count_architecture is not None:
        model = model_defs['Pipeline1.1'][args.count_architecture]

    else:
        model = model_defs['Pipeline1.2'][args.det_architecture]

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()
    model.eval()

    # load saved model weights from training
    if args.count_architecture is not None:
        model_name = args.count_architecture + '_ts-' + args.training_dir.split('_')[-1]
        model.load_state_dict(
            torch.load("./saved_models_stable/Pipeline1.1/{}/{}.tar".format(model_name, model_name)))

        counts = predict_patch(model=model, input_size=model_archs[args.count_architecture]['input_size'],
                               pipeline='Pipeline1.1',
                               batch_size=hyperparameters[args.hyperparameter_set_count]['batch_size_test'],
                               test_dir='./' + args.dest_folder,
                               out_file='{}_count'.format(os.path.basename(input_image)[:-4]),
                               dest_folder='./' + args.dest_folder,
                               num_workers=hyperparameters[args.hyperparameter_set_count]['num_workers_train'],
                               class_names=class_names)
    else:
        model_name = args.det_architecture + '_ts-' + args.training_dir.split('_')[-1]
        model.load_state_dict(
            torch.load("./saved_models_stable/Pipeline1.2/{}/{}.tar".format(model_name, model_name)))

        counts, locations = predict_patch(model=model, input_size=model_archs[args.det_architecture]['input_size'],
                                          pipeline='Pipeline1.2',
                                          batch_size=
                                          hyperparameters[args.hyperparameter_set_count]['batch_size_test'],
                                          test_dir='./' + args.dest_folder,
                                          out_file='{}_count'.format(os.path.basename(input_image)[:-4]),
                                          dest_folder='./' + args.dest_folder,
                                          num_workers=
                                          hyperparameters[args.hyperparameter_set_count]['num_workers_train'],
                                          class_names=class_names)
        locations.to_csv('locations.csv')

    print('    Total predicted in {}: '.format(os.path.basename(input_image)), sum(counts['predictions']))

    for row in counts.iterrows():
        fname = row[1]['filenames']
        output_shpfile.loc[fname, 'count'] += row[1]['predictions']

    # save shapefile for counts / classes
    shapefile_path = './{}/predicted_shapefiles/{}/'.format(args.dest_folder,
                                                            os.path.basename(input_image)[:-4])
    os.makedirs(shapefile_path)
    output_shpfile.to_file(shapefile_path + '{}_prediction.shp'.format(os.path.basename(args.dest_folder)))

    # save shapefile for individual seal locations
    if args.det_architecture is not None and len(locations) > 0:
        # create geopandas DataFrame to store classes and counts per patch
        output_shpfile_locs = gpd.GeoDataFrame()

        # setup projection for output
        output_shpfile_locs.crs = from_epsg(3031)

        # add locations
        for row in locations.iterrows():
            fname = row[1]['filenames']
            up, left, down, right = [int(ele) for ele in fname.split('_')[-5: -1]]
            x, y = [up + row[1]['y'], left + row[1]['x']]
            x_loc, y_loc = [x, y] * affine_matrix
            output_shpfile_locs = output_shpfile_locs.append(pd.Series({'geometry': Point(x_loc, y_loc),
                                                                        'x': x, 'y': y}),
                                                             ignore_index=True)

        # add scene name
        output_shpfile_locs = output_shpfile_locs.join(pd.DataFrame({
            'scene': [os.path.basename(input_image)] * len(output_shpfile_locs)}))

        # save shapefile
        output_shpfile_locs.to_file(shapefile_path + '{}_locations.shp'.format(
            os.path.basename(args.dest_folder)))

    # clean up tiles and sub-tiles
    if os.path.exists('./{}/tiles'.format(args.dest_folder)):
        shutil.rmtree('./{}/tiles'.format(args.dest_folder))


if __name__ == "__main__":
    main()

