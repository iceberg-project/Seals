import pandas as pd
import numpy as np
import os
from osgeo import gdal, ogr, osr
import cv2


def get_patches(raster_dir: str, vector_df: str, lon: str, lat: str, patch_sizes: list, labels: list,
                train_prop: float=0.9) -> object:
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
        train_prop : proportion of images added to training set (float between 0 and 1)

    Output:
        folder with extracted multi-band training images separated in subfolders by intended label
    """
    # check for invalid inputs
    assert 1 >= train_prop >= 0, "Proportion of images going to training set is not a valid probability"
    assert sum([patch_sizes[idx] >= patch_sizes[idx+1] for idx in range(len(patch_sizes) - 1)]) == 0,\
        "Patch sizes with non-increasing dimensions"

    # read pandas data.frame
    df = pd.read_csv(vector_df)

    for folder in ['training', 'validation']:
        if not os.path.exists("./{}".format(folder)):
            os.makedirs("./{}".format(folder))
        for lbl in labels:
            subdir = "./{}/{}".format(folder, lbl)
            if not os.path.exists(subdir):
                os.makedirs(subdir)

    rasters = []
    for path, _, files in os.walk(raster_dir):
        for filename in files:
            filename_lower = filename.lower()
            if not filename_lower.endswith('.tif'):
                print('{} is not a valid scene.'.format(filename_lower))
                continue
            rasters.append(os.path.join(path, filename))

    # keep track of how many points were processed
    num_imgs = 0
    for rs in rasters:
        gdata = gdal.Open('{}'.format(rs))
        ulx, xres, _, uly, _, yres = gdata.GetGeoTransform()
        lrx = ulx + (gdata.RasterXSize * xres)
        lry = uly + (gdata.RasterYSize * yres)
        band = gdata.GetRasterBand(1)
        #del gdata
        no_data = band.GetNoDataValue()

        w, h = xres, yres

        # change upper left corner into latitude and longitude

        # source projection
        source = osr.SpatialReference()
        source.ImportFromWkt(gdata.GetProjection())

        # target projection
        target = osr.SpatialReference()
        target.ImportFromEPSG(4326)

        # Create the transform - this can be used repeatedly
        transform = osr.CoordinateTransformation(source, target)

        # transform x0 and y0
        lat_lon_ul = transform.TransformPoint(ulx, uly)
        lat_lon_lr = transform.TransformPoint(lrx, lry)

        x0, y0 = lat_lon_ul[0], lat_lon_ul[1]
        xc, yc = lat_lon_lr[0], lat_lon_lr[1]
        data = band.ReadAsArray().astype(np.int8)
        #del band
        #data = np.pad(data, pad_width=patch_sizes[-1], mode='constant', constant_values=0)
        #TODO: add padding to raster data based on the last element patch_sizes

        # iterate through the points
        for p in df.iterrows():
            x = int((p[1][lon] - x0) * data.shape[0] / (xc - x0))
            y = int((p[1][lat] - y0) * data.shape[1] / (yc - y0))
            print(x)
            print(y)
            bands = []
            for scale in patch_sizes:
                try:
                    if data[y, x] != no_data:
                        #TODO: save patch image to correct subfolder based on label
                        patch = data[y - int(scale/2): y + int(scale/2), x - int(scale/2): x + int(scale/2)]
                        patch = cv2.resize(patch, (scale[0], scale[0]))
                        bands.append(patch)
                except:
                    pass
            #TODO: merge bands into a numpy array and write to folder
            bands = np.array(bands)
            print(bands.shape)
            file = "./{}/{}/{}.jpg".format(np.random.choice(['training', 'validation'], p=[train_prop, 1-train_prop]),
                                           p[1]['label'], num_imgs)
            cv2.imwrite(file, bands)
            num_imgs += 1

        del data, band
    return None


get_patches(raster_dir="/home/bento/imagery", vector_df="temp-nodes.csv", lat='y', lon='x',
            patch_sizes=[299, 600, 1200], labels=["crabeater", "weddell", "other", "pack-ice", "emperor"])