import os
import argparse
import time
import cv2
import rasterio
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Divides a raster image into \
                                     files')
    parser.add_argument('--scale_bands', type=str, help='For multi-scale \
                        models, string with size of scale bands separated \
                        by spaces')
    parser.add_argument('--input_image', type=str, help='Full path to raster \
                        file we wish to tile out')
    parser.add_argument('--output_folder', type=str, help='Folder where tiles \
                        will be stored')
    parser.add_argument('--pad', type=str, default=False, help='Flag for \
                        padding the image, required for multiscale')

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
                curr_scale = band[x - scale // 2: x + scale // 2,
                                  y - scale // 2: y + scale // 2]
                curr_scale = cv2.resize(curr_scale, (scales[0], scales[0]))
                scale_bands.append(curr_scale)
            # remove black corners
            if np.max(scale_bands[0]) == 0:
                continue
            # combine scales and save tile
            scale_bands = np.dstack(scale_bands)
            # save it with polygon coordinates
            filename = "{}/tile_{}_{}_{}_{}_.jpg".format(output_folder, up,
                                                         left, down, right)
            cv2.imwrite(filename, scale_bands)
            count += 1
    toc = time.time()
    elapsed = toc - tic
    print('\n{} tiles created in {} minutes and {:.2f} seconds'.
          format(count, int(elapsed // 60), elapsed % 60))


def main():
    args = parse_args()

    # unroll arguments
    input_image = args.input_image
    scales = [int(ele) for ele in args.scale_bands.split('_')]
    output_folder = args.output_folder + '/tiles'


    tile_raster(input_image, output_folder, scales, True)


if __name__ == '__main__':
    main()
