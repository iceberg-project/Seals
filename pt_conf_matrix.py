from sealnet_nas_scalable import *
import torch
import pandas as pd
import datetime
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
import warnings

# image transforms seem to cause truncated images, so we need this
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings('ignore', module='PIL')


def get_conf_matrix(model, val_dir, batch_size=8, input_size=299, to_csv=True):
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
    dataset = datasets.ImageFolder(val_dir, data_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1)

    # extract class names
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
        running_corrects += torch.sum(preds == labels.data)

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
        now = datetime.datetime.now()
        conf_matrix.to_csv('{}_{}_{}_{}_{}_conf_matrix.csv'.format(now.day, now.month, now.year, now.hour, now.minute))

    return conf_matrix


def main():
    # loading the pretrained model and adding new classes to it
    # create model instance
    model = NASNetALarge(in_channels_0=48, out_channels_0=24, out_channels_1=32,
                         out_channels_2=64, out_channels_3=128)
    # load saved model weights from pt_train.py
    model.load_state_dict(torch.load("./nn_model.pth.tar"))

    # check for GPU support and set model to evaluation mode
    model.eval()
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()

    # run validation to get confusion matrix
    conf_matrix = get_conf_matrix(model=model, val_dir='./training_set_multiscale/validation')


if __name__ == '__main__':
    main()
