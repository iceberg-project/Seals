import torch
import pandas as pd
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import torch.nn as nn
import time
import warnings
import argparse
from model_library import *
from nasnet_scalable import NASNetALarge

# image transforms seem to cause truncated images, so we need this
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings('ignore', module='PIL')

parser = argparse.ArgumentParser(description='validates a CNN at the scene level')
parser.add_argument('--validation_dir', type=str, help='base directory to recursively search for scenes')
parser.add_argument('--model_architecture', type=str, help='model architecture, must be a member of models '
                                                           'dictionary')
parser.add_argument('--hyperparameter_set', type=str, help='combination of hyperparameters used, must be a member of '
                                                           'hyperparameters dictionary')
parser.add_argument('--input_name', type=str, help='name of input model file from training, this name will also be used'
                                                   'in subsequent steps of the pipeline')
args = parser.parse_args()