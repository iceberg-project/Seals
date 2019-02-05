"""
Marge shapefiles
==========================================================

Output shapefile merger for ICEBERG seals use case. Merges all '.dbf' files inside an input folder into a single
shapefile.

Author: Bento Goncalves
License: MIT
Copyright: 2018-2019
"""

import argparse
import os
import shutil

import geopandas as gpd
from fiona.crs import from_epsg


def parse_args():
    parser = argparse.ArgumentParser(description='merges shapefiles')
    parser.add_argument('--input_dir', type=str, help='base directory to recursively search for shapefiles')
    parser.add_argument('--output_dir', type=str, help='folder where merged shapefiles will be stored')
    return parser.parse_args()


def main():
    args = parse_args()
    merged = gpd.GeoDataFrame()
    merged.crs = from_epsg(3031)

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    for path, subfolder, file in os.walk('{}'.format(args.input_dir)):
        for filename in file:
            if filename.split('_')[-1] == 'locations.dbf':
                if len(gpd.read_file('{}/{}'.format(path, filename))) > 0:
                    merged = merged.append(gpd.read_file('{}/{}'.format(path, filename)), ignore_index=True)

    merged.to_file('{}/merged_locations.shp'.format(args.output_dir))


if __name__ == '__main__':
    main()
