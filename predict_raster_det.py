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
from tile_raster import tile_raster

# image transforms seem to cause truncated images, so we need this
from PIL import Image

warnings.filterwarnings('ignore', module='PIL')


parser = argparse.ArgumentParser(description='validates a CNN at the haul out level')
parser.add_argument('--input_image', type=str, help='base directory to recursively search for validation images in')
parser.add_argument('--class_architecture', type=str, help='model architecture for classification, must be a member of '
                                                           'models dictionary')
parser.add_argument('--count_architecture', type=str, nargs='?',  help='model architecture for counting, must be a '
                                                                       'member of models dictionary')
parser.add_argument('--det_architecture', type=str, nargs='?', help='model architecture for detecting, must be a '
                                                                    'member of models dictionary')
parser.add_argument('--hyperparameter_set_class', type=str, help='combination of hyperparameters used for the '
                                                                 'classification model, must be a member of '
                                                                 'hyperparameters dictionary')
parser.add_argument('--hyperparameter_set_count', type=str, help='combination of hyperparameters used for the '
                                                                 'counting model, must be a member of '
                                                                 'hyperparameters dictionary')
parser.add_argument('--training_dir', type=str, help='directory where models were trained')
parser.add_argument('--dest_folder', type=str, default='to_classify', help='folder where the model will be saved')
parser.add_argument('--pos_classes', type=str, default='crabeater_weddell', help='classes that we wish to count')
parser.add_argument('--skip_class', type=str, default=False, help='option to skip classifying step')


# helper function to divide patch into sub-patches for counting
def get_subpatches(patch, count_size, pad_int=0):
    """
    Subdivides a patch into sub-patches and counts within sub-patches, padding it to match counting CNN dimensions.
    Returns a total count of seals inside that patch

    :param patch: (np.array) patch image converted to np.array with dtype=uint8
    :param count_size: (int) input size for counting CNN
    :param pad_int: (int) intensity value for
    :param count_args: argparse object with model arguments to define count_cnn
    :return:  total count of seals in patch
    """

    # create padding
    pad = 0

    # extract patch_size
    patch_size = patch.shape[0]

    # check if dimensions are the same
    if patch_size == count_size:
        return [patch]

    # find minimum amount of padding required and pad image if necessary
    if (patch_size / count_size) % 1 != 0:
        pad = int(np.ceil((np.ceil(patch_size / count_size) * count_size - patch_size)))
        # pad image
        patch = np.dstack([np.pad(patch[:, :, i], pad_width=pad, mode='constant', constant_values=pad_int)
                           for i in range(patch.shape[2])])
        # update patch size
        patch_size = patch.shape[0] + 1

    # extract subpatches
    sub_patches = [patch[i:i+count_size, j:j+count_size, :] for i in range(pad, patch_size - count_size, count_size)
                   for j in range(pad, patch_size - count_size, count_size)]

    return sub_patches


def main():
    # unroll arguments
    args = parser.parse_args()
    input_image = args.input_image
    output_folder = args.dest_folder
    scales = [model_archs[args.class_architecture]['input_size']] * 3
    output_folder = './{}/tiles/images/'.format(output_folder)

    # check for pre-existing tiles and subtiles
    if os.path.exists('./{}/tiles'.format(args.dest_folder)):
        shutil.rmtree('./{}/tiles'.format(args.dest_folder))

    if os.path.exists('./{}/sub-tiles'.format(args.dest_folder)):
        shutil.rmtree('./{}/sub-tiles'.format(args.dest_folder))

    # tile raster into patches
    tile_raster(input_image, output_folder, scales)

    print('\nPredicting with {}:'.format(os.path.basename(args.dest_folder)))

    # create geopandas DataFrame to store classes and counts per patch
    output_shpfile = gpd.GeoDataFrame()

    # setup projection for output
    output_shpfile.crs = from_epsg(3031)

    # get affine matrix for raster file
    with rasterio.open(input_image) as src:
        affine_matrix = src.transform

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

    # treat all patches as positive if skipping classification
    if args.skip_class == '1':
        pos_patches = fnames

    else:
        # check if patches were already classified with class architecture
        preds_path = './{}/ClassPredictions/{}/{}/class_predictions.csv'.format(args.dest_folder.split('/')[0],
                                                                                args.class_architecture,
                                                                                os.path.basename(args.input_image))
        if os.path.exists(preds_path):
            predictions = pd.read_csv(preds_path)

        else:
            # find patches with seals with classification CNN
            num_classes = training_sets[args.training_dir]['num_classes']
            model = model_defs['Pipeline1'][args.class_architecture](num_classes)

            use_gpu = torch.cuda.is_available()
            if use_gpu:
                model.cuda()
            model.eval()

            # load saved model weights from pt_train.py
            model_name = args.class_architecture + '_ts-' + args.training_dir.split('_')[-1]
            model.load_state_dict(torch.load("./saved_models_stable/Pipeline1/{}/{}.tar".format(model_name,
                                                                                                model_name)))
            # classify patches
            predictions = predict_patch(model=model, input_size=model_archs[args.class_architecture]['input_size'],
                                        pipeline='Pipeline1',
                                        batch_size=hyperparameters[args.hyperparameter_set_class]['batch_size_test'],
                                        test_dir='./' + args.dest_folder,
                                        out_file='{}_class'.format(os.path.basename(input_image)[:-4]),
                                        dest_folder='./' + args.dest_folder,
                                        num_workers=hyperparameters[args.hyperparameter_set_class]['num_workers_train'],
                                        class_names=class_names)

            os.makedirs('/'.join(preds_path.split('/')[:-1]))
            predictions.to_csv(preds_path)

        # add entries for predictions in GeoDataFrame
        for row in predictions.iterrows():
            fname = row[1]['filenames']
            output_shpfile.loc[fname, 'label'] = row[1]['predictions']

        # get the subset of positive patches filenames
        positive_classes = args.pos_classes.split('_')
        pos_patches = predictions.loc[predictions['predictions'].isin(positive_classes), 'filenames']

    # check if there are positive patches
    if len(pos_patches) > 0:

        if args.count_architecture is not None:
            count_size = model_archs[args.count_architecture]['input_size']
        else:
            count_size = model_archs[args.det_architecture]['input_size']

        os.makedirs('./{}/subtiles/images'.format(args.dest_folder))

        # find number of subpatches in patch
        n_sub = int(np.ceil(scales[0] / count_size))
        # loop over positive patches creating subpatches
        for fname in pos_patches:
            # get polygon for tiles
            up, left, down, right = [int(ele) for ele in fname.split('_')[-5: -1]]
            # extract subpatches
            subpatches = get_subpatches(patch=np.array(Image.open('./{}/tiles/images/{}'.format(args.dest_folder,
                                                                                                fname))),
                                        count_size=count_size)
            for idx, patch in enumerate(subpatches):
                # get polygon for subtile
                sub_up = up + ((idx % n_sub) * count_size)
                sub_down = sub_up - count_size
                sub_left = left + ((idx // n_sub) * count_size)
                sub_right = sub_left + count_size
                cv2.imwrite('./{}/subtiles/images/{}-{}-sub_{}_{}_{}_{}_.jpg'.format(args.dest_folder, idx,
                                                                                   fname.split('.')[0], sub_up,
                                                                                   sub_left, sub_down, sub_right),
                            patch)

        # remove tiles to count only subtiles
        shutil.rmtree('./{}/tiles'.format(args.dest_folder))

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
            fname = row[1]['filenames'].split('-')[1] + '.jpg'
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
                x, y = [up + row[1]['y'], left + row[1]['x']] * affine_matrix
                output_shpfile_locs = output_shpfile_locs.append(pd.Series({'geometry': Point(x, y)}),
                                                                 ignore_index=True)

            # save shapefile
            output_shpfile_locs.to_file(shapefile_path + '{}_locations.shp'.format(
                os.path.basename(args.dest_folder)))

    # clean up tiles and sub-tiles
    if os.path.exists('./{}/tiles'.format(args.dest_folder)):
        shutil.rmtree('./{}/tiles'.format(args.dest_folder))

    if os.path.exists('./{}/subtiles'.format(args.dest_folder)):
        shutil.rmtree('./{}/subtiles'.format(args.dest_folder))


if __name__ == "__main__":
    main()
