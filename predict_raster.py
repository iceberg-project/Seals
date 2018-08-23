import torch
import pandas as pd
import geopandas as gpd
import warnings
import argparse
from shapely.geometry.geo import box
import affine
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
parser.add_argument('--count_architecture', type=str, help='model architecture for counting, must be a member of '
                                                           'models dictionary')
parser.add_argument('--hyperparameter_set_class', type=str, help='combination of hyperparameters used for the '
                                                                 'classification model, must be a member of '
                                                                 'hyperparameters dictionary')
parser.add_argument('--hyperparameter_set_count', type=str, help='combination of hyperparameters used for the '
                                                                 'counting model, must be a member of '
                                                                 'hyperparameters dictionary')
parser.add_argument('--training_dir', type=str, help='directory where models were trained')
parser.add_argument('--dest_folder', type=str, default='to_classify', help='folder where the model will be saved')
parser.add_argument('--pos_classes', type=str, default='crabeater_weddell', help='classes that we wish to count')
parser.add_argument('--skip_to_count', type=str, default=False, help='option to skip classifying and just count')


# helper function to divide patch into sub-patches for counting
def get_subpatches(patch, count_size, pad_int=255):
    """
    Subdivides a patch into sub-patches and counts within sub-patches, padding it to match counting CNN dimensions.
    Returns a total count of seals inside that patch

    :param patch: (np.array) patch image converted to np.array with dtype=uint8
    :param count_size: (int) input size for counting CNN
    :param pad_int: (int) intensity value for
    :param count_args: argparse object with model arguments to define count_cnn
    :return:  total count of seals in patch
    """
    
    # extract patch_size
    patch_size = patch.shape[0]

    # check if dimensions are the same
    if patch_size == count_size:
        return patch

    # find minimum amount of padding required and pad image if necessary
    if (patch_size / count_size) % 1 != 0:
        pad = int(np.ceil((np.ceil(patch_size / count_size) * count_size - patch_size) / 2))
        # pad image
        patch = np.dstack([np.pad(patch[:, :, i], pad_width=pad, mode='constant', constant_values=pad_int)
                           for i in range(patch.shape[2])])
        # update patch size
        patch_size += pad * 2

    # extract subpatches
    sub_patches = [patch[i:i+count_size, j:j+count_size, :] for i in range(0, patch_size - count_size, count_size)
                   for j in range(0, patch_size - count_size, count_size)]

    return sub_patches


def main():
    # unroll arguments
    args = parser.parse_args()
    input_image = args.input_image
    output_folder = args.dest_folder
    scales = [model_archs[args.class_architecture]['input_size']] * 3
    output_folder = './{}/tiles/images/'.format(output_folder)

    # check for pre-existing tiles and subties
    if os.path.exists('./{}/tiles'.format(args.dest_folder)):
        shutil.rmtree('./{}/tiles'.format(args.dest_folder))

    if os.path.exists('./{}/sub-tiles'.format(args.dest_folder)):
        shutil.rmtree('./{}/sub-tiles'.format(args.dest_folder))

    # tile raster into patches
    tile_raster(input_image, output_folder, scales)

    print('\nPredicting with {}:'.format(os.path.basename(args.dest_folder)))

    # create geopandas DataFrame for storing output
    output_shpfile = gpd.GeoDataFrame()
    # setup projection for output
    output_shpfile.crs = from_epsg(3031)

    # get affine matrix for raster file
    with rasterio.open(input_image) as src:
        affine_matrix = affine.Affine.from_gdal(*src.transform)

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
    if args.skip_to_count == '1':
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
            model.load_state_dict(torch.load("./saved_models_stable/Pipeline1/{}/{}.tar".format(args.class_architecture,
                                                                                                args.class_architecture)))
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

        if not os.path.exists('./{}/sub-tiles/images'.format(args.dest_folder)):
            os.makedirs('./{}/sub-tiles/images'.format(args.dest_folder))

        # loop over positive patches creating subpatches
        for fname in pos_patches:
            subpatches = get_subpatches(patch=np.array(Image.open('./{}/tiles/images/{}'.format(args.dest_folder, fname))),
                                        count_size=model_archs[args.count_architecture]['input_size'])
            for idx, patch in enumerate(subpatches):
                if np.min(patch) > 200 or np.max(patch) < 55:
                    continue
                cv2.imwrite('./{}/sub-tiles/images/{}-{}'.format(args.dest_folder, idx, fname), patch)

        # remove tiles to count only sub-tiles
        shutil.rmtree('./{}/tiles'.format(args.dest_folder))

        # count inside subpatches with counting CNN
        model = model_defs['Pipeline1.1'][args.count_architecture]

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            model.cuda()
        model.eval()

        # load saved model weights from pt_train.py
        model.load_state_dict(torch.load("./saved_models_stable/Pipeline1.1/{}/{}.tar".format(args.count_architecture,
                                                                                              args.count_architecture)))

        counts = predict_patch(model=model, input_size=model_archs[args.count_architecture]['input_size'],
                               pipeline='Pipeline1.1',
                               batch_size=hyperparameters[args.hyperparameter_set_count]['batch_size_test'],
                               test_dir='./' + args.dest_folder,
                               out_file='{}_count'.format(os.path.basename(input_image)[:-4]),
                               dest_folder='./' + args.dest_folder,
                               num_workers=hyperparameters[args.hyperparameter_set_count]['num_workers_train'],
                               class_names=class_names)
        print('    Total predicted in {}: '.format(os.path.basename(input_image)), sum(counts['predictions']))

        for row in counts.iterrows():
            fname = row[1]['filenames'].split('-')[1]
            output_shpfile.loc[fname, 'count'] += row[1]['predictions']

    # save shapefile
    shapefile_path = './{}/predicted_shapefiles/{}/'.format(args.dest_folder, os.path.basename(input_image)[:-4])
    os.makedirs(shapefile_path)
    output_shpfile.to_file(shapefile_path + '{}_prediction.shp'.format(os.path.basename(args.dest_folder)))


if __name__ == "__main__":
    main()
