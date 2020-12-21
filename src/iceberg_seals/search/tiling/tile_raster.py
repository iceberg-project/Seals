"""
Tile raster
==========================================================

Tiling script for ICEBERG seals use case. Tiles rasters into predefined size patches. Patches are named according to
the raster indices that define their boundaries. Optional arguments allow padding and multiple scale bands. Also saves
a .csv for the raster's affine matrix -- used on 'predict_sealnet.py' to go from raster index to projected 'x' and
'y' of predicted seals.

Author: Bento Goncalves
License: MIT
Copyright: 2018-2019
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from itertools import product
from functools import partial
from multiprocessing import Pool, cpu_count
import os
import argparse
import cv2
import time


def parse_args():
    parser = argparse.ArgumentParser(description='divides a raster image into files')

    parser.add_argument('--input_image', '-i', type=str, required=True,
                        help='full path to raster file we wish to tile out')
    parser.add_argument('--output_folder', '-o', type=str, required=True,
                        help='folder where tiles will be stored')
    parser.add_argument('--bands', '-b', required=False, type=str, default='0',
                        help='sting with bands seperated by commas. defaults to 0 for the panchromatic band')
    parser.add_argument('--stride', '-s', type=float, default=1.0, required=False,
                        help='distance between tiles as a multiple of patch_size. defaults to 1.0, i.e. adjacent '
                             'tiles without overlap')
    parser.add_argument('--patch_size', '-p', type=int, default=224, required=False,
                        help='side dimensions for each patch. patches are required to be squares.')
    parser.add_argument('--geotiff', '-g', type=int, default=0, required=False,
                        help='boolean for whether to keep geographical information.')

    return parser.parse_args()


# helper function to write a windowed tile
def write_tile(scn, bands, patch_size, output_folder, offset, geotiff):
    # read window
    window = Window(row_off=offset[0], col_off=offset[1], width=patch_size, height=patch_size)

    # get band indexes
    indexes = [scn.indexes[ele] for ele in bands]
    patch = scn.read(indexes, window=window)
    filename = "%s/tile_%d_%d_%d_%d_.tif" % (output_folder, offset[0], offset[1],
                                             offset[0] + patch_size, offset[1] + patch_size)
    if geotiff:
        with rasterio.open(filename,
                           mode='w',
                           driver='GTiff',
                           width=patch_size,
                           height=patch_size,
                           transform=scn.window_transform(window),
                           crs=scn.crs,
                           count=len(bands),
                           compress='lzw',
                           dtype=rasterio.uint8) as dst:
            dst.write(patch)
    else:
        patch = np.transpose(patch, [1, 2, 0])
        cv2.imwrite(filename, patch)


def tile_raster(input_image, output_folder, bands, patch_size, stride=1, geotiff=0):
    """
    Function to tile a raster into tiles of predefined size and bands.
    :param input_image: full path to input raster
    :param output_folder: path to output folder
    :param bands: selected bands, defaults to panchromatic
    :param patch_size: x and y dimensions for each tile
    :param stride: distance between consecutive tiles as a product of patch size
    :return: None
    """

    # time it
    tic = time.time()

    # create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # read entire scene
    src_scn = rasterio.open(input_image)

    # save affine matrix
    affine_matrix = pd.DataFrame({'transform': src_scn.transform[:6]})
    affine_matrix.to_csv('%s/affine_matrix.csv' % output_folder)

    # get number of rows and columns in scene
    nrows, ncols = src_scn.shape

    # create iterator with offsets for small windows
    offsets = product(range(0, nrows, int(patch_size * stride)),
                      range(0, ncols, int(patch_size * stride)))
    n = 0

    # create multiprocessing pool and tile in parallel
    for offset in offsets:
        n += 1
        write_tile(src_scn, bands, patch_size, output_folder, offset, geotiff)

    elapsed = time.time() - tic
    print('\n%d tiles created in %d minutes and %.2f seconds' % (n, int(elapsed // 60), elapsed % 60))


def main():
    args = parse_args()

    # unroll arguments
    input_image = args.input_image
    output_folder = args.output_folder
    bands = [int(ele) for ele in args.bands.split(',')]
    patch_size = args.patch_size
    stride = args.stride
    geotiff = args.geotiff

    # tile image
    tile_raster(input_image, output_folder, bands, patch_size, stride, geotiff)


if __name__ == '__main__':
    main()
