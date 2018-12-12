import pandas as pd
import numpy as np
import os
import cv2
import time
import random
import argparse
import rasterio
from sklearn.utils import shuffle

parser = argparse.ArgumentParser(description='creates training sets to train and validate sealnet instances')
parser.add_argument('--rasters_dir', type=str, help='root directory where rasters are located')
parser.add_argument('--scale_bands', type=str, help='for multi-scale models, string with size of scale bands separated'
                                                    'by underscores')
parser.add_argument('--out_folder', type=str, help='directory where training set will be saved to')
parser.add_argument('--labels', type=str, help='class names, separated by underscores')
parser.add_argument('--det_classes', type=str, help='classes that will be targeted at detection, '
                                                    'separated by underscores')
parser.add_argument('--shape_file', type=str, help='path to shape file with seal points')
parser.add_argument('--georef', type=str, default='1', help="whether or not the raster is georeferenced")
parser.add_argument('--remap', type=str, nargs='?', help='label remapping')
parser.add_argument('--rgb', type=str, default=False, help='whether or not training set is RGB (vs. grayscale)')

args = parser.parse_args()


def get_patches(out_folder: str, raster_dir: str, shape_file: str, lon: str, lat: str, patch_sizes: list,
                remap: dict, rgb: bool) -> object:
    """
    Generates multi-band patches at different scales around vector points to use as a training set.

    Input:
        out_folder: folder name for the created dataset
        raster_dir : directory with raster images (.tif) we wish to extract patches from.
        shape_file : path to .csv shape_file with training points with latitude, longitude, classification label and
            source raster layer.
        lon : column in vector_df containing longitude component.
        lat : column in vector_df containing latitude component.
        patch_sizes : list with pyramid dimensions in multi-band output, first element is used as base dimension,
            subsequent elements must be larger than the previous element and will be re-sized to match patch_sizes[0]
        labels : list with training set labels

    Output:
        folder with extracted multi-band training images separated in subfolders by intended label
    """
    # check for invalid inputs
    assert sum([patch_sizes[idx] > patch_sizes[idx+1] for idx in range(len(patch_sizes) - 1)]) == 0,\
        "Patch sizes with non-increasing dimensions"

    # extract detection classes, the training set will store count and (x,y) within tile for objects of those classes
    det_classes = args.det_classes.split('_')

    # save seal locations inside images
    detections = pd.DataFrame()

    # read csv exported from seal points shape file as a pandas DataFrame
    df = pd.read_csv(shape_file)

    # create training set directory
    if not os.path.exists("./training_sets/"):
        os.makedirs("./training_sets/")

    if not os.path.exists("./training_sets/{}".format(out_folder)):
        os.makedirs("./training_sets/{}".format(out_folder))

    for folder in ['training', 'validation']:
        if not os.path.exists("./training_sets/{}/{}".format(out_folder, folder)):
            os.makedirs("./training_sets/{}/{}".format(out_folder, folder))
        for lbl in pd.unique([ele for ele in remap.values()]):
            subdir = "./training_sets/{}/{}/{}".format(out_folder, folder, lbl)
            if not os.path.exists(subdir):
                os.makedirs(subdir)

    rasters = []
    print("Checking input folder for invalid files:\n\n")
    for path, _, files in os.walk(raster_dir):
        for filename in files:
            filename_lower = filename.lower()
            # only add raster files wth annotated points
            if not filename_lower.endswith('.tif'):
                print('  {} is not a valid scene.'.format(filename))
                continue
            if filename not in pd.unique(df['scene']):
                print('  {} is not an annotated scene.'.format(filename))
                continue
            rasters.append(os.path.join(path, filename))

    # shuffle rasters and remove potential duplicates
    rasters = shuffle(rasters)

    # keep track of how many points were processed
    num_imgs = 0
    since = time.time()
    print("\nCreating dataset:\n")
    for cnt, rs in enumerate(rasters):
        # get distance to determine which points will fall inside the tile
        tile_center = patch_sizes[0] // 2

        # amount of padding
        pad = patch_sizes[-1] // 2
        # extract image data and affine transforms
        with rasterio.open(rs) as src:
            if rgb:
                rgb_bands = src.read()[0:3, :, :].astype(np.uint8)
                #rgb_bands = np.vstack([np.pad(band, pad_width=pad, mode='constant', constant_values=0) for band in
                #                       bands])
            else:
                band = src.read()[0, :, :].astype(np.uint8)
                #band = np.pad(band, pad_width=pad, mode='constant', constant_values=0)
            affine_transforms = src.transform

        # get coordinates from affine matrix
        if args.georef == '1':
            width, _, x0, _, height, y0 = affine_transforms[:6]

        else:
            width, height = 1, -1
            x0, y0 = 0, 0

        # filter points to include points inside current raster, sort them based on coordinates and fix index range
        df_rs = df.loc[df['scene'] == os.path.basename(rs)]
        df_rs = df_rs.sort_values(by=[lon, lat])
        df_rs.index = range(len(df_rs.index))

        # iterate through the points
        for row, p in enumerate(df_rs.iterrows()):
            x = int((p[1][lon] - x0) / width) #+ pad
            y = int((p[1][lat] - y0) / height) #+ pad
            upper_left = [x - int(patch_sizes[0]/2), y - int(patch_sizes[0]/2)]
            bands = []
            # extract patches at different scales
            for idx, scale in enumerate(patch_sizes):
                try:
                    if rgb:
                        patch = rgb_bands[idx, y - int(scale / 2): y + int((scale + 1) / 2),
                                          x - int(scale / 2): x + int((scale + 1) / 2)]
                        patch = cv2.resize(patch, (patch_sizes[0], patch_sizes[0]))
                        bands.append(patch)
                    else:
                        patch = band[y - int(scale/2): y + int((scale+1)/2), x - int(scale/2): x + int((scale+1)/2)]
                        patch = cv2.resize(patch, (patch_sizes[0], patch_sizes[0]))
                        bands.append(patch)
                except:
                    continue

            # check if we have a valid image
            if len(bands) == len(patch_sizes):
                # combine bands into an image
                bands = np.dstack(bands)
                # get remapped label
                label = remap[p[1]['label']]
                # save patch image to correct subfolder based on label
                filename = "./training_sets/{}/{}/{}/{}.jpg".format(out_folder, p[1]['dataset'], label,
                                                                    p[1]['shapeid'])
                cv2.imwrite(filename, bands)
                # store counts and detections
                locs = ""
                # add a detection in the center of the tile if class is in det_classes
                if label in det_classes:
                    locs += "{}_{}".format(tile_center, tile_center)

                    # look down the DataFrame for detections that also fall inside the tile
                    inside = True
                    search_idx = row + 1
                    while inside:
                        # check if idx is still inside DataFrame
                        if search_idx > (len(df_rs) - 1):
                            break
                        # get det_x, det_y
                        #det_x = (int((df_rs.loc[search_idx, lon] - x0) / width) + pad) - upper_left[0]
                        #det_y = (int((df_rs.loc[search_idx, lat] - y0) / height) + pad) - upper_left[1]
                        det_x = (int((df_rs.loc[search_idx, lon] - x0) / width)) - upper_left[0]
                        det_y = (int((df_rs.loc[search_idx, lat] - y0) / height)) - upper_left[1]

                        # check if it falls inside patch
                        if 0 <= det_x < patch_sizes[0]:
                            # check label
                            if 0 <= det_y < patch_sizes[0]:
                                if remap[df_rs.loc[search_idx, 'label']] in det_classes:
                                    # search y direction
                                    locs += "_{}_{}".format(det_x, det_y)
                            search_idx += 1
                        else:
                            inside = False

                    # look up the DataFrame for detections that also fall inside the tile
                    inside = True
                    search_idx = row - 1
                    while inside:
                        # check if idx is still inside DataFrame
                        if search_idx < 0:
                            break
                        # get det_x, det_y
                        #det_x = (int((df_rs.loc[search_idx, lon] - x0) / width) + pad) - upper_left[0]
                        #det_y = (int((df_rs.loc[search_idx, lat] - y0) / height) + pad) - upper_left[1]
                        det_x = (int((df_rs.loc[search_idx, lon] - x0) / width)) - upper_left[0]
                        det_y = (int((df_rs.loc[search_idx, lat] - y0) / height)) - upper_left[1]

                        # check if it falls inside patch
                        if 0 <= det_x < patch_sizes[0]:
                            # check label
                            if 0 <= det_y < patch_sizes[0]:
                                if remap[df_rs.loc[search_idx, 'label']] in det_classes:
                                    # search y direction
                                    locs += "_{}_{}".format(det_x, det_y)
                            search_idx -= 1
                        else:
                            inside = False
                # add detections
                new_row = pd.Series({'file_name': os.path.basename(filename), 'locations': locs})
                new_row.name = p[1]['shapeid']
                detections = detections.append(new_row)
                num_imgs += 1

        if rgb:
            del rgb_bands
        else:
            del band
        print("\n  Processed {} out of {} rasters".format(cnt + 1, len(rasters)))

    time_elapsed = time.time() - since
    print("\n\n{} training images created in {:.0f}m {:.0f}s".format(num_imgs, time_elapsed // 60, time_elapsed % 60))
    detections = detections.sort_index()
    detections.to_csv('./training_sets/{}/detections.csv'.format(out_folder))


def main():
    # set random seed to get same order of samples in both vanilla and multiscale training sets
    random.seed(4)

    raster_dir = args.rasters_dir
    patch_sizes = [int(ele) for ele in args.scale_bands.split('_')]
    out_folder = args.out_folder
    labels = args.labels.split('_')
    shape_file = args.shape_file
    rgb = args.rgb == '1'

    if args.remap is not None:
        remap = {ele[0]: ele[1] for ele in [entry.split('->') for entry in args.remap.split('_')]}
    else:
        remap = {ele: ele for ele in labels}

    # create vanilla and multi-scale training sets
    print('\nCreating {}:\n'.format(out_folder))
    get_patches(out_folder=out_folder, raster_dir=raster_dir, shape_file=shape_file, lat='y', lon='x',
                patch_sizes=patch_sizes, remap=remap, rgb=rgb)


if __name__ == '__main__':
    main()
