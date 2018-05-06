import numpy as np
import rasterio
import os
import argparse
import cv2

parser = argparse.ArgumentParser(description='divides a raster image into files')
parser.add_argument('--scale_bands', type=str, help='for multi-scale models, string with size of scale bands separated'
                                                    'by spaces')
parser.add_argument('--input_image', type=str, help='full path to raster file we wish to tile out')
parser.add_argument('--output_folder', type=str, help='folder where tiles will be stored')

args = parser.parse_args()


# shadows of arguments before arg parser
input_image = args.input_image
output_folder = args.output_folder
scales = [int(ele) for ele in args.scale_bands.split('_')]
output_folder = './{}/{}/{}/tiles'.format(output_folder, os.path.basename(input_image), args.scale_bands)


def main():
    # create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # read image
    with rasterio.open(input_image) as src:
        band = src.read()[0, :, :]


    # pad image
    pad = scales[-1] // 2
    band = np.pad(band, pad_width=pad, mode='constant', constant_values=0)

    # extract tile size and raster size
    tile_size = scales[0]
    raster_width = band.shape[0]
    raster_height = band.shape[1]

    # tile out image
    for x in range(tile_size // 2 + pad, raster_width - pad, tile_size):
        for y in range(tile_size // 2 + pad, raster_height - pad, tile_size):
            scale_bands = []
            for scale in scales:
                curr_scale = band[x - scale // 2: x + scale // 2, y - scale // 2: y + scale // 2]
                curr_scale = cv2.resize(curr_scale, (scales[0], scales[0]))
                scale_bands.append(curr_scale)
            # check if the tile is not at the corners
            if np.argmax(scale_bands[0]) == 0:
                continue
            # combine scales and save tile
            scale_bands = np.dstack(scale_bands)
            filename = "{}/tile_{}_{}_.jpg".format(output_folder, x - pad, y - pad)
            cv2.imwrite(filename, scale_bands)


if __name__ == '__main__':
    main()




