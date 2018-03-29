import pandas as pd
import numpy as np
import os
from osgeo import gdal
import cv2


def get_patches(raster_dir: str, vector_df: str, lon: str, lat: str, patch_sizes: list, labels: list) -> object:
    """
    Generates multi-band patches at different scales around vector points to use as a training set.

    Input:
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
    print(pd.unique(df['scene']))

    # create training set directory
    if not os.path.exists("./training_set"):
        os.makedirs("./training_set")

    for folder in ['training', 'validation']:
        if not os.path.exists("./training_set/{}".format(folder)):
            os.makedirs("./training_set/{}".format(folder))
        for lbl in labels:
            subdir = "./training_set/{}/{}".format(folder, lbl)
            if not os.path.exists(subdir):
                os.makedirs(subdir)

    rasters = []
    for path, _, files in os.walk(raster_dir):
        for filename in files:
            filename_lower = filename.lower()
            # only add raster files wth annotated points
            if not filename_lower.endswith('.tif'):
                print('{} is not a valid scene.'.format(filename))
                continue
            if filename not in pd.unique(df['scene']):
                print('{} is not an annotated scene.'.format(filename))
                continue
            rasters.append(os.path.join(path, filename))

    # keep track of how many points were processed
    num_imgs = 0
    for rs in rasters:
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
                file = "./training_set/{}/{}/{}.jpg".format(p[1]['dataset'], p[1]['label'], num_imgs)
                cv2.imwrite(file, bands)
                num_imgs += 1

        del data, band
    return None


get_patches(raster_dir="/home/bento/imagery", vector_df="temp-nodes.csv", lat='y', lon='x',
            patch_sizes=[450, 450, 450], labels=["crabeater", "weddell", "other", "pack-ice", "emperor"])