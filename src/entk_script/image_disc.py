from glob import glob
import argparse
import os
import pandas as pd

def image_discovery(path,filename='list.csv',filesize=False):

    filepaths = glob(path+'/*.tif')
    if filesize:
        dataset_df = pd.DataFrame(columns=['Filename','Size'])
        for filepath in filepaths:
            dataset_df.loc[len(dataset_df)] = [filepath,os.path.getsize(filepath)]
    else:
        dataset_df = pd.DataFrame(columns=['Filename'])
        for filepath in filepaths:
            dataset_df.loc[len(dataset_df)] = [filepath]

    dataset_df.to_csv(filename,index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path',help='Path to a remote resource where data are')
    parser.add_argument('--filename', type=str, default='list.csv',help='Name of the output CSV file')
    parser.add_argument('--filesize', help='Include the filesize to the output CSV', action='store_true')
    args = parser.parse_args()

    image_discovery(args.path,args.filename,args.filesize)