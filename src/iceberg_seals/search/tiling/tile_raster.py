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
import os
import argparse
import cv2
import time


def parse_args():
    parser = argparse.ArgumentParser(description='divides a raster image into files')
    parser.add_argument('--scale_bands', type=str, required=True,
                        help='for multi-scale models, string with size of \
                              scale bands separated by spaces')
    parser.add_argument('--input_image', type=str, required=True,
                        help='full path to raster file we wish to tile out')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='folder where tiles will be stored')
    parser.add_argument('--pad', type=str, default=False,
                        help='flag for padding the image, required for \
                              multiscale')

    return parser.parse_args()


def tile_raster(input_image, output_folder, scales, pad_img=False):
    # time it
    tic = time.time()

    # create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # read image
    with rasterio.open(input_image) as src:
        band = np.array(src.read()[0, :, :], dtype=np.uint8)
        # save affine matrix
        affine_matrix = pd.DataFrame({'transform': src.transform[:6]})
        affine_matrix.to_csv('%s/affine_matrix.csv' % output_folder)

    # add tiles subfolder
    output_folder = '%s/tiles/%s/' % (output_folder, os.path.basename(input_image))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # pad image
    pad = 0

    if pad_img:
        pad = scales[-1] // 2
        band = np.pad(band, pad_width=pad, mode='constant', constant_values=0)

    # extract tile size and raster size
    tile_size = scales[0]
    raster_width = band.shape[0]
    raster_height = band.shape[1]

    # tile out image
    count = 0
    for x in range(tile_size // 2 + pad, raster_width - pad, tile_size):
        for y in range(tile_size // 2 + pad, raster_height - pad, tile_size):
            scale_bands = []
            # find corners for polygon
            up = y - scales[0] // 2
            left = x - scales[0] // 2
            down = y + scales[0] // 2
            right = x + scales[0] // 2
            for scale in scales:
                curr_scale = band[x - scale // 2: x + scale // 2,  y - scale // 2: y + scale // 2]
                curr_scale = cv2.resize(curr_scale, (scales[0], scales[0]))
                scale_bands.append(curr_scale)
            # remove black corners
            if np.max(scale_bands[0]) == 0:
                continue
            # combine scales and save tile
            scale_bands = np.dstack(scale_bands)
            # save it with polygon coordinates
            filename = "%s/tile_%d_%d_%d_%d_.jpg" % (output_folder, up, left, down, right)
            cv2.imwrite(filename, scale_bands)
            count += 1
    toc = time.time()
    elapsed = toc - tic
    print('\n%d tiles created in %d minutes and %.2f seconds' % (count, int(elapsed // 60), elapsed % 60))


def main():
    args = parse_args()

    # unroll arguments
    input_image = args.input_image
    scales = [int(ele) for ele in args.scale_bands.split('_')]
    output_folder = args.output_folder
    tile_raster(input_image, output_folder, scales, True)


if __name__ == '__main__':
    main()




