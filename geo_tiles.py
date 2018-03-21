#!/usr/bin/env python3

import os
import argparse
import rasterio
import rasterio.windows
import glob

def tile_raster(input_path,output_dir,width,height,overlap=0):
    """
    Tiles a .tif into smaller tif files with dimensions set
    by width and height (in pixels), with an overlap between
    tiles (also in pixels). The same overlap is applied
    horizontally and vertically.
    Tiles on the left and bottom borders will be padded with
    no data to ensure they are the requested size.
    Image tiles are output into an appropriately named sub-directory of
    'output', with the bounding box (in the CRS of the original tiff)
     appended to each tiles name.
     Also outputs a csv file providing the affine transformation to
     go from pixel coordinates in a tile back to the original CRS.

    :param input_path: .tif file to be tiled
    :param output_dir:
    :param width: integer pixels
    :param height: integer pixels
    :param overlap: integer pixels
    :return: None
    """
    path = input_path
    output_directory = output_dir
    target_height = height
    target_width = width
    with rasterio.open(path) as src:
        # get base name of input file for use in tile names
        src_name = os.path.splitext(os.path.basename(path))[0]

        # input tif dimensions
        src_height = src.shape[0]
        src_width = src.shape[1]

        # create new sub-directory to store tiles
        tile_dir = os.path.join(output_directory,src_name)
        if not os.path.exists(tile_dir):
            os.makedirs(tile_dir)
        tile_scheme_path = os.path.join(tile_dir,'tile_scheme.csv')

        # Lazy open original tiff and read each window (tile) to write
        with open(tile_scheme_path,'w') as tile_scheme:
            for i in range(0, src_height - overlap, target_height - overlap):
                for j in range(0, src_width - overlap, target_width - overlap):

                    # Create tile window
                    tile_window = rasterio.windows.Window(i,j,target_width,target_height)
                    window_transform = rasterio.windows.transform(tile_window, src.transform)
                    window_bounds = src.window_bounds(tile_window)

                    # Generate tile name and affine transform
                    tile_name = os.path.join("{}_{}_{}_{}_{}.jpg".format(src_name,*window_bounds))
                    tile_path = os.path.join(tile_dir,tile_name)
                    comma_sep_affine = "{},{}".format(tile_name,",".join(str(x) for x in window_transform.to_gdal()))
                    tile_scheme.write(comma_sep_affine + "\n")

                    # Update tile meta-data
                    kwargs = src.meta.copy()
                    kwargs.update({
                        'height': tile_window.height,
                        'width': tile_window.width,
                        'transform': window_transform})

                    # Write tile
                    with rasterio.open(tile_path, 'w', **kwargs) as dst:
                        dst.write(src.read(window=tile_window))


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input",help="Input directory")
    parser.add_argument("-o","--output",help="Output directory")
    parser.add_argument("-d","--dimensions",nargs=2,help="Tile height and width",type=int)
    parser.add_argument("-l","--overlap", help="Tile overlap",type=int)
    args = parser.parse_args()
    images = glob.glob(os.path.join(args.input,"*.tif"))
    for image in images:
        tile_raster(image,output_dir=args.output,width=args.dimensions[1],height=args.dimensions[0],overlap=args.overlap)


