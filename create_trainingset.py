import pandas as pd
import numpy as np
import os
import cv2
import time
import random
from osgeo import gdal
from sklearn.utils import shuffle


def get_patches(out_folder: str, raster_dir: str, vector_df: str, lon: str, lat: str, patch_sizes: list,
                labels: list) -> object:
    """
    Generates multi-band patches at different scales around vector points to use as a training set.

    Input:
        out_folder: folder name for the created dataset
        raster_dir : directory with raster images (.tif) we wish to extract patches from.
        vector_df : path to pandas data.frame with training points with latitude, longitude, classification label and
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

    # read pandas data.frame
    df = pd.read_csv(vector_df)

    # shuffle rows
    df = shuffle(df)

    # create training set directory
    if not os.path.exists("./{}".format(out_folder)):
        os.makedirs("./{}".format(out_folder))

    for folder in ['training', 'validation']:
        if not os.path.exists("./{}/{}".format(out_folder, folder)):
            os.makedirs("./{}/{}".format(out_folder, folder))
        for lbl in labels:
            subdir = "./{}/{}/{}".format(out_folder, folder, lbl)
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


    # shuffle rasters
    rasters = shuffle(rasters)

    # keep track of how many points were processed
    num_imgs = 0
    since = time.time()
    print("\nCreating dataset:\n")
    for count, rs in enumerate(rasters):
        gdata = gdal.Open(rs)

        # get upper left corners and pixel height and width from raster
        x0, w, _, y0, _, h = gdata.GetGeoTransform()

        # extract raster values and save as numpy array
        band = gdata.GetRasterBand(1)
        data = band.ReadAsArray().astype(np.uint8)

        # free up memory
        del gdata

        # add padding to prevent out of range indices close to borders
        data = np.pad(data, pad_width=patch_sizes[-1] // 2, mode='constant', constant_values=0)

        # filter points to include points inside current raster
        df_rs = df.loc[df['scene'] == os.path.basename(rs)]

        # iterate through the points
        for p in df_rs.iterrows():
            x = int((p[1][lon] - x0) / w) + patch_sizes[-1] // 2
            y = int((p[1][lat] - y0) / h) + patch_sizes[-1] // 2
            bands = []
            # extract patches at different scales
            for scale in patch_sizes:
                try:
                    patch = data[y - int(scale/2): y + int((scale+1)/2), x - int(scale/2): x + int((scale+1)/2)]
                    patch = cv2.resize(patch, (patch_sizes[0], patch_sizes[0]))
                    bands.append(patch)
                except:
                    continue

            # check if we have a valid image
            if len(bands) == len(patch_sizes):
                # combine bands into an image
                bands = np.dstack(bands)
                # save patch image to correct subfolder based on label
                file = "./{}/{}/{}/{}.jpg".format(out_folder, p[1]['dataset'], p[1]['label'], num_imgs)
                cv2.imwrite(file, bands)
                num_imgs += 1

        del data, band
        print("\n  Processed {} out of {} rasters".format(count + 1, len(rasters)))

    time_elapsed = time.time() - since
    print("\n\n{} training images created in {:.0f}m {:.0f}s".format(num_imgs, time_elapsed // 60, time_elapsed % 60))
    return None


# set random seed to get same order of samples in both vanilla and multiscale training sets
random.seed(4)

# create vanilla and multiscale training sets
get_patches(out_folder="training_set", raster_dir="/home/bento/imagery", vector_df="temp-nodes.csv", lat='y', lon='x',
            patch_sizes=[450, 450, 450], labels=["crabeater", "weddell", "other", "pack-ice", "emperor", "open-water",
                                                 "ice-sheet", "marching-emperor", "rock", "crack", "glacier"])

get_patches(out_folder="training_set_multiscale", raster_dir="/home/bento/imagery", vector_df="temp-nodes.csv",
            lat='y', lon='x', patch_sizes=[450, 1350, 4000], labels=["crabeater", "weddell", "other", "pack-ice",
                                                                     "open-water", "ice-sheet", "marching-emperor",
                                                                     "emperor", "rock", "crack", "glacier"])
