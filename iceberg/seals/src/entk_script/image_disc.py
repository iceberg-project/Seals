"""
Image Discovery Kernel
==========================================================

This script takes as input a path and returns a dataframe
with all the images and their size.

Author: Ioannis Paraskevakos
License: MIT
Copyright: 2018-2019
"""
from glob import glob
import argparse
import os
import math
import pandas as pd


def image_discovery(path, filename='list.csv', filesize=False):
    """
    This function creates a dataframe with image names and size from a path.

    :Arguments:
        :path: Images path, str
        :filename: The filename of the CSV file containing the dataframe.
                   Default Value: list.csv
        :filesize: Whether or not the image sizes should be inluded to the
                   dataframe. Default value: False
    """

    filepaths = glob(path + '/*.tif')
    if filesize:
        dataset_df = pd.DataFrame(columns=['Filename', 'Size'])
        for filepath in filepaths:
            filesize = int(math.ceil(os.path.getsize(filepath)/1024/1024))
            dataset_df.loc[len(dataset_df)] = [filepath, filesize]
    else:
        dataset_df = pd.DataFrame(columns=['Filename'])
        for filepath in filepaths:
            dataset_df.loc[len(dataset_df)] = [filepath]

    dataset_df.to_csv(filename, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to a remote resource where data \
                        are')
    parser.add_argument('--filename', type=str, default='list.csv',
                        help='Name of the output CSV file')
    parser.add_argument('--filesize', help='Include the filesize to the \
                        output CSV', action='store_true')
    args = parser.parse_args()

    image_discovery(args.path, args.filename, args.filesize)
