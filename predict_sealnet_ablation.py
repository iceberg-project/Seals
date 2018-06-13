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
parser.add_argument('--model_name', type=str, help='name of output file from training, this name will also be used in '
                                                   'subsequent steps of the pipeline')
parser.add_argument('--pipeline', type=str, help='name of the detection pipeline where the model will be saved')
parser.add_argument('--positive_classes', type=str, help='name of classes that that should be kept with _ in between'
                                                         'class labels')
parser.add_argument('--dest_folder', type=str, default='saved_models', help='folder where the model will be saved')


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


ImageFile.LOAD_TRUNCATED_IMAGES = True

# load positive classes
pos_classes = args.positive_classes.split('_')

# normalize input images
arch_input_size = model_archs[args.model_architecture]['input_size']
data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = './training_sets/{}/validation/'.format(args.training_dir)

# create dataloader instance
dataset = ImageFolderTest(data_dir, data_transforms)
batch_size = hyperparameters[args.hyperparameter_set]['batch_size_test']
num_workers = hyperparameters[args.hyperparameter_set]['num_workers_train']
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


img_exts = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
class_names = sorted([subdir for subdir in os.listdir('./training_sets/{}/training'.format(args.training_dir))])

use_gpu = torch.cuda.is_available()


# classify images with CNN
def main():
    # keep patch classifications
    classified = pd.DataFrame()

    # load number of classes
    num_classes = training_sets[args.training_dir]['num_classes']

    # create model
    model_ft = model_defs[args.pipeline][args.model_architecture](num_classes)

    # check for GPU support and set model to evaluation mode
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model_ft.cuda()

    # load features
    model_ft.load_state_dict(torch.load("./{}/{}/{}/{}.tar".format(args.dest_folder, args.pipeline, args.model_name,
                                                                   args.model_name)))

    # change to evaluation mode
    model_ft.eval()

    # classify images in dataloader
    for data in dataloader:
        # get the inputs
        inputs, file_names = data

        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        # do a forward pass to get predictions
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs.data, 1)
        for idx, label in enumerate([int(ele) for ele in preds]):
            if class_names[label] in pos_classes:
                classified = classified.append({'file': file_names[idx]}, ignore_index=True)

    # save output to .csv file

    classified.to_csv('./{}/{}/classified_patches.csv'.format(args.dest_folder, args.pipeline))


if __name__ == '__main__':
    main()
