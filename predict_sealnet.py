import torch
from torch.autograd import Variable
from torchvision import transforms, models
from utils.dataloaders.data_loader_test import ImageFolderTest
import os
from utils.model_library import *
import torch.nn as nn
import pandas as pd
from PIL import ImageFile
import argparse
import affine

parser = argparse.ArgumentParser(description='predict new images using a previously trained model')
parser.add_argument('--training_dir', type=str, help='base directory to recursively search for images in')
parser.add_argument('--model_architecture', type=str, help='model architecture, must be a member of models '
                                                           'dictionary')
parser.add_argument('--hyperparameter_set', type=str, help='combination of hyperparameters used, must be a member of '
                                                           'hyperparameters dictionary')
parser.add_argument('--output_name', type=str, help='name of output file from training, this name will also be used in '
                                                    'subsequent steps of the pipeline')
parser.add_argument('--pipeline', type=str, help='name of the detection pipeline where the model will be saved')

parser.add_argument('--data_dir', type=str, help='directory with tiles to be classified')
parser.add_argument('--affine_transforms', type=str, help='name of .csv file with affine data transforms')
parser.add_argument('--scene', type=str, help='name of raster file where data_dir is located')

args = parser.parse_args()

# check for invalid inputs
if args.model_architecture not in model_archs:
    raise Exception("Invalid architecture -- see supported architectures:  {}".format(list(model_archs.keys())))

if args.training_dir not in training_sets:
    raise Exception("Training set is not defined in ./utils/model_library.py")

if args.hyperparameter_set not in hyperparameters:
    raise Exception("Hyperparameter combination is not defined in ./utils/model_library.py")

if args.pipeline not in model_defs:
    raise Exception('Pipeline is not defined in ./utils/model_library.py')

# load data transforms
affine_transforms = pd.read_csv(args.affine_transforms).to_dict('list')

# get affine transformations
affine_transforms = {key: affine.Affine(*affine_transforms[key]) for key in affine_transforms}

ImageFile.LOAD_TRUNCATED_IMAGES = True

# normalize input images
arch_input_size = model_archs[args.model_architecture]['input_size']
data_transforms = transforms.Compose([
        transforms.CenterCrop(arch_input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# create dataloader instance
dataset = ImageFolderTest(args.data_dir, data_transforms)
batch_size = hyperparameters[args.hyperparameter_set]['batch_size_test']
num_workers = hyperparameters[args.hyperparameter_set]['num_workers_train']
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


img_exts = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
class_names = sorted([subdir for subdir in os.listdir('./training_sets/{}/training'.format(args.training_dir))])
data_dir = args.data_dir

use_gpu = torch.cuda.is_available()

# create tile catalog to go from classification to x,y
classified = pd.DataFrame()

# add tiled images to the catalog
idx = 0
for path, _, files in os.walk(data_dir):
    for filename in files:
        # extract x and y from filename and run affine transformation
        x, y = affine_transforms[args.scene] * (int(filename.split('_')[-3]), int(filename.split('_')[-2]))
        filename_lower = filename.lower()
        if not (any(filename_lower.endswith(ext) for ext in img_exts)):
            print('{} is not a valid image file.'.format(filename))
            continue

        row = {'scene': args.scene, 'label': None, 'x': x, 'y': y, 'tile': filename}
        classified = classified.append(row, ignore_index=True)
        idx += 1


# classify images with CNN
def main():
    # load number of classes
    num_classes = training_sets[args.training_dir]['num_classes']

    # create model
    model_ft = model_defs[args.pipeline][args.model_architecture]

    # check for GPU support and set model to evaluation mode
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model_ft.cuda()
    model_ft.eval()

    # load saved model weights from pt_train.py, check if it is a haulout model or a single seal model
    if model_archs[args.model_architecture]['haulout']:
        model_ft.load_state_dict(torch.load("./saved_models/haulout/{}/{}.tar".format(args.model_name, args.model_name)))
    else:
        model_ft.load_state_dict(torch.load("./saved_models/single_seal/{}/{}.tar".format(args.model_name,
                                                                                          args.model_name)))
    if use_gpu:
        model_ft.cuda()

    # classify images in dataloader
    for data in dataloader:
        # get the inputs
        inputs, _, file_names = data

        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        # do a forward pass to get predictions
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs.data, 1)
        for idx, label in enumerate([int(ele) for ele in preds]):
            classified.loc[classified['tile'] == file_names[idx], 'label'] = class_names[label]

    # save output to .csv file
    if model_archs[args.model_architecture]['haulout']:
        classified.to_csv('./saved_models/haulout/{}/{}_scene_val_tmp.csv'.format(args.model_name, args.model_name),
                          index=False)
    else:
        classified.to_csv('./saved_models/single_seal/{}/{}_scene_val_tmp.csv'.format(args.model_name, args.model_name),
                          index=False)


if __name__ == '__main__':
    main()
