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


parser = argparse.ArgumentParser(description='creates training sets to train and validate sealnet instances')
parser.add_argument('--out_folder', type=str, help='directory where training set will be saved to')
parser.add_argument('--label', type=str, help='class name to search for seals')
parser.add_argument('--background', type=str, help='type of background')


def create_synthetic_ts():
    pass