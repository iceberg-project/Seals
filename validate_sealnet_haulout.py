import torch
import pandas as pd
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import torch.nn as nn
import time
import warnings
import argparse
from model_library import *
from custom_architectures.nasnet_scalable import NASNetALarge

# image transforms seem to cause truncated images, so we need this
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings('ignore', module='PIL')

parser = argparse.ArgumentParser(description='validates a CNN at the haul out level')
parser.add_argument('--training_dir', type=str, help='base directory to recursively search for validation images in')
parser.add_argument('--model_architecture', type=str, help='model architecture, must be a member of models '
                                                           'dictionary')
parser.add_argument('--hyperparameter_set', type=str, help='combination of hyperparameters used, must be a member of '
                                                           'hyperparameters dictionary')
parser.add_argument('--model_name', type=str, help='name of input model file from training, this name will also be used'
                                                   'in subsequent steps of the pipeline')
args = parser.parse_args()

# check for invalid inputs
if args.model_architecture not in model_archs:
    raise Exception("Unsupported architecture")

if args.training_dir not in training_sets:
    raise Exception("Invalid training set")

if args.hyperparameter_set not in hyperparameters:
    raise Exception("Invalid hyperparameter combination")


def validate_model(model, val_dir, out_file, batch_size=8, input_size=299,  to_csv=True, num_workers=1):
    """
    Generates a confusion matrix from a PyTorch model and validation images

    :param model: pyTorch model (already trained)
    :param val_dir: str -- directory with validation images
    :param batch_size: int -- number of images per batch
    :param input_size: int -- size of input images
    :param to_csv : whether or not pandas dataframe gets saved as a .csv table
    :return: pd.data.frame -- data frame with predictions and labels by validation batch
    """

    # crop and normalize images
    data_transforms = transforms.Compose([
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load dataset
    dataset = datasets.ImageFolder('./training_sets/{}/validation'.format(val_dir), data_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    class_names = dataset.classes

    # check for GPU support
    use_gpu = torch.cuda.is_available()

    # create pandas data.frame for confusion matrix
    conf_matrix = pd.DataFrame(columns=['predicted', 'ground_truth'])

    # set training flag to False
    model.train(False)

    # keep track of correct answers to get accuracy
    running_corrects = 0

    # keep track of running time
    since = time.time()

    for data in dataloader:
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # do a forward pass to get predictions
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        # keep track of correct answers to get accuracy
        running_corrects += torch.sum(preds == labels.data).item()

        # add current predictions to conf_matrix
        conf_matrix_batch = pd.DataFrame(data=[[class_names[int(ele)] for ele in preds],
                                               [class_names[int(ele)] for ele in labels]])
        conf_matrix_batch = conf_matrix_batch.transpose()
        conf_matrix_batch.columns = ['predicted', 'ground_truth']
        conf_matrix = conf_matrix.append(conf_matrix_batch)

    time_elapsed = time.time() - since

    # print output
    print('Validation complete in {}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, time_elapsed // 60, time_elapsed % 60))
    print('Validation Acc: {:4f}'.format(running_corrects / len(dataset)))
    
    if to_csv:
        conf_matrix.to_csv('./saved_models/haulout/{}/{}_haul_val.csv'.format(out_file, out_file))

    return conf_matrix


def main():
    # create model instance
    num_classes = training_sets[args.training_dir]['num_classes']
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

    # load saved model weights from pt_train.py
    model_ft.load_state_dict(torch.load("./saved_models/haulout/{}/{}.tar".format(args.model_name, args.model_name)))

    # run validation to get confusion matrix
    conf_matrix = validate_model(model=model_ft, input_size=model_archs[args.model_architecture]['input_size'],
                                 batch_size=hyperparameters[args.hyperparameter_set]['batch_size_test'],
                                 val_dir=args.training_dir, out_file=args.model_name,
                                 num_workers=hyperparameters[args.hyperparameter_set]['num_workers_train'])


if __name__ == '__main__':
    main()
