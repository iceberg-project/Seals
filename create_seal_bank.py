import pandas as pd
import numpy as np
import os
import cv2
import time
import random
import argparse
import rasterio
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from utils.model_library import *
from PIL import Image

parser = argparse.ArgumentParser(description='creates training sets to train and validate sealnet instances')
parser.add_argument('--out_folder', type=str, help='directory where seal bank will be saved to')
parser.add_argument('--training_dir', type=str, help='directory where training set is located')
parser.add_argument('--label', type=str, help='class name to search for seals')

# read arguments
args = parser.parse_args()

# check for invalid arguments
seal_classes = ['weddell', 'crabeater']
if args.label not in seal_classes:
    raise Exception('Invalid label choice, must be one of {}'.format(seal_classes))

if args.training_dir not in training_sets:
    raise Exception("Training set is not defined in ./utils/model_library.py")

# create folder to store seal bank
dir_path = args.out_folder + '/{}'.format(args.label)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)


# helper function to get distance between two points
def dist2d(point1, point2):
    return np.sqrt(np.power(point1[0] - point2[0], 2) + np.power(point1[1] - point2[1], 2))


# function to create seal bank
def create_seal_bank(seal_points, detections, label, out, seal_radius=5):
    """
    Loops through a training set cropping individual seals and saving minimum distance to the next seal if there are
    two or more seals in the haul out

    :param detections: (pd.DataFrame) -- data frame with img filenames and seal locations within img
    :param seal_points: (pd.DataFrame) -- data frame with img filenames and img classes
    :param label: (str) -- class label of target seal images
    :param seal_radius: (int) -- radius of buffer around individual seals
    :param out: (str) -- name of the seal bank folder (saved as a subfolder of ./training_sets/seal_banks)
    :return: a folder with cropped individual seals and csv file with mean distance and sd of distance
    """

    # store distance between nearest neighbors
    dists = []

    # filter detections to include only rows with seals
    idcs = [ele[0] for ele in seal_points.loc[seal_points['label'] == label, 'shapeid'].items()]
    detections = detections.loc[idcs]

    # loop over rows cropping individual seals
    print('\nGenerating {} seal bank:'.format(label))
    for idx, row in enumerate(detections.iterrows()):
        if idx % 100 == 0:
            print('  extracted {} out of {} {} seal images'.format(idx, label, len(detections)))
        # read images
        filepath = './training_sets/{}/training/{}/{}'.format(args.training_dir, args.label, row[1]['file_name'])
        if not os.path.exists(filepath):
            filepath = './training_sets/{}/validation/{}/{}'.format(args.training_dir, args.label, row[1]['file_name'])
        img = np.array(Image.open(filepath))

        # do histogram equalization
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)

        # crop image
        img = img[225 - seal_radius: 225 + seal_radius,
                  225 - seal_radius: 225 + seal_radius, :]

        # write image
        cv2.imwrite('{}/{}/{}'.format(out, label, row[1]['file_name']), img)

        # add nearest neighbor distances
        locs = row[1]['locations'].split('_')
        locs = [[int(locs[i]), int(locs[i+1])] for i in range(0, len(locs) - 1)]
        if len(locs) > 1:
            # keep track of distances and which points where already surveyed
            cur_dists = []
            explored = []
            if len(explored) == len(locs):
                break
            for i in range(len(locs)):
                min_dist = [10000, 50]
                for j in range(i+1, len(locs)):
                    cur_dist = dist2d(locs[i], locs[j])
                    if cur_dist < min_dist[0]:
                        min_dist = [cur_dist, j]
                explored.extend([i, min_dist[1]])
                if min_dist[0] != 10000:
                    cur_dists.append(min_dist[0])

            # store distances
            dists.extend(cur_dists)

    # extract mean distance and sd for distances
    distances = pd.DataFrame({'distance': [ele for ele in dists]})
    distances.to_csv('{}/{}/{}_distances.csv'.format(out, label, label))


def main():
    # run function
    seal_points = pd.read_csv('seal_points_espg3031.csv')
    detetections = pd.read_csv('./training_sets/{}/detections.csv'.format(args.training_dir), index_col=0)
    create_seal_bank(seal_points, detetections, args.label, args.out_folder)


if __name__ == '__main__':
    main()