import torch
from torch.autograd import Variable
from torchvision import transforms, models
import os
from model_library import *
import torch.nn as nn
import pandas as pd
from custom_architectures.nasnet_scalable import NASNetALarge
from PIL import Image
from PIL import ImageFile
import argparse
import affine

parser = argparse.ArgumentParser(description='predict new images using a previously trained model')
parser.add_argument('--training_dir', type=str, help='base directory to search for classification labels')
parser.add_argument('--model_architecture', type=str, help='model architecture, must be a member of models '
                                                           'dictionary')
parser.add_argument('--model_name', type=str, help='name of input model file from training, this name will also be used'
                                                   'in subsequent steps of the pipeline')
parser.add_argument('--data_dir', type=str, help='directory with tiles to be classified')
parser.add_argument('--affine_transforms', type=str, help='name of .csv file with affine data transforms')
parser.add_argument('--scene', type=str, help='name of raster file where data_dir is located')

args = parser.parse_args()

# check for invalid inputs
if args.model_architecture not in model_archs:
    raise Exception("Unsupported architecture")

if args.training_dir not in training_sets:
    raise Exception("Invalid training set")

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

img_exts = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
class_names = sorted([subdir for subdir in os.listdir('./training_sets/{}/training'.format(args.training_dir))])
data_dir = args.data_dir

use_gpu = torch.cuda.is_available()


def main():
    # create a pandas DataFrame to save classified images
    classified = pd.DataFrame()
    num_classes = training_sets[args.training_dir]['num_classes']

    # create model instance
    if args.model_architecture == "Resnet18":
        model_ft = models.resnet18(pretrained=False, num_classes=num_classes)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    else:
        model_ft = NASNetALarge(in_channels_0=48, out_channels_0=24, out_channels_1=32, out_channels_2=64,
                                out_channels_3=128, num_classes=num_classes)

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
    # recursively iterate over all files in the base folder and classify them
    for path, _, files in os.walk(data_dir):
        for filename in files:
            # extract x and y from filename and run affine transformation
            x, y = affine_transforms[args.scene] * (int(filename.split('_')[-3]), int(filename.split('_')[-2]))
            filename_lower = filename.lower()
            if not (any(filename_lower.endswith(ext) for ext in img_exts)):
                print('{} is not a valid image file.'.format(filename))
                continue
            try:
                f = os.path.join(path, filename)
                image = Image.open(f)
                img_tensor = data_transforms(image)
                img_tensor.unsqueeze_(0)
                if use_gpu:
                    img_variable = Variable(img_tensor.cuda())
                else:
                    img_variable = Variable(img_tensor)
                output = model_ft(img_variable)
                _, pred = torch.max(output.data, 1)
                # add prediction to output
                row = {'scene': args.scene, 'label': class_names[pred], 'x': x, 'y': y, 'tile': filename}
                classified = classified.append(row, ignore_index=True)
            except Exception as e:
                print('{} is invalid: {}'.format(f, e))
                continue
    if model_archs[args.model_architecture]['haulout']:
        classified.to_csv('./saved_models/haulout/{}/{}_scene_val_tmp.csv'.format(args.model_name, args.model_name))
    else:
        classified.to_csv('./saved_models/single_seal/{}/{}_scene_val_tmp.csv'.format(args.model_name, args.model_name))


if __name__ == '__main__':
    main()
