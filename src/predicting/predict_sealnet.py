import torch
import pandas as pd
from torchvision import datasets, transforms, models
import numpy as np
from utils.dataloaders.data_loader_test import ImageFolderTest
import time
import warnings
import argparse
from utils.model_library import *

# image transforms seem to cause truncated images, so we need this
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings('ignore', module='PIL')


def parse_args():
    parser = argparse.ArgumentParser(description='validates a CNN at the haul out level')
    parser.add_argument('--test_dir', type=str, help='base directory to recursively search for validation images in')
    parser.add_argument('--model_architecture', type=str, help='model architecture, must be a member of models '
                                                               'dictionary')
    parser.add_argument('--hyperparameter_set', type=str, help='combination of hyperparameters used, must be a member '
                                                               'of hyperparameters dictionary')
    parser.add_argument('--model_name', type=str, help='name of input model file from training, this name will also be '
                                                       'used in subsequent steps of the pipeline')
    parser.add_argument('--pipeline', type=str, help='name of the detection pipeline where the model is loaded from')
    parser.add_argument('--ablation', type=int, default=0, help='boolean for whether or not this validation run will'
                                                                'work on the ablation dataset. runs on regular training'
                                                                'sets by default')
    parser.add_argument('--dest_folder', type=str, default='saved_models', help='folder where the model will be saved')
    return parser.parse_args()


# helper function to get the (x,y) of max values
def get_xy_locs(array, count):
    if count == 0:
        return np.array([])
    cols = array.shape[1]
    flat = array.flatten()
    return np.array([[x // cols, x % cols] for x in flat.argsort()[-count:]])


def predict_patch(model, dest_folder, test_dir, out_file, pipeline, batch_size=2, input_size=299,
                  class_names='', num_workers=1):
    """
    Predicts class or number of seals in an image

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
    dataset = ImageFolderTest(test_dir, data_transforms)

    # separate into batches with dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    # check for GPU support
    use_gpu = torch.cuda.is_available()

    # store predictions and filenames
    predicted_cnts = []
    predicted_locs = []
    predictions = []
    fnames = []

    # set training flag to False
    model.train(False)

    # keep track of running time
    since = time.time()

    for data in dataloader:

        # get the inputs
        inputs, filenames = data

        # gpu support
        if use_gpu:
            inputs = inputs.cuda()

        # do a forward pass to get predictions
        # detection models
        if pipeline == 'Pipeline1.2':
            cnts, locs = model(inputs)
            # if statement prevents iterations over 0-d tensors
            if cnts.size() != torch.Size([0]):
                pred_cnt_batch = [max(0, round(float(ele))) for ele in cnts]
                locs = locs.cpu().detach()
                # find predicted location
                locs = [get_xy_locs(loc, max(0, int(cnts[idx]))) for idx, loc in enumerate(locs.numpy())]
                # save batch predictions
                predicted_cnts.extend(pred_cnt_batch)
            predicted_locs.extend(locs)
        # counting and classification models
        else:
            outputs = model(inputs)

            # save batch predictions
            if pipeline == "Pipeline1":
                _, preds = torch.max(outputs.data, 1)
                # predicted classes
                preds_batch = [class_names[int(ele)] for ele in preds]

            else:
                # predicted counts
                preds_batch = [max(0, round(float(ele))) for ele in outputs]
            # add batch predictions
            predictions.extend(preds_batch)
        # add filename
        fnames.extend(filenames)
    time_elapsed = time.time() - since

    # print output
    print('Testing complete in {}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, time_elapsed // 60, time_elapsed % 60))

    # save output to .csv
    # detection output
    if pipeline == 'Pipeline1.2':
        # locations
        pred_locations = {'x': [],
                          'y': [],
                          'filenames': []}
        for idx, batch in enumerate(predicted_locs):
            for pnt in batch:
                pred_locations['x'].append(pnt[0])
                pred_locations['y'].append(pnt[1])
                pred_locations['filenames'].append(fnames[idx])
        pred_locations = pd.DataFrame(pred_locations)
        # counts
        predictions = pd.DataFrame({'predictions': [ele for ele in predicted_cnts],
                                    'filenames': [ele for ele in fnames]})
        return predictions, pred_locations

    # classification and counting output
    else:
        predictions = pd.DataFrame({'predictions': [ele for ele in predictions],
                                    'filenames': [ele for ele in fnames]})
        predictions.to_csv('./{}/{}_predictions.csv'.format(dest_folder, out_file), index=False)
        return predictions


def main():
    # load arguments
    args = parse_args()

    # check for invalid inputs
    if args.model_architecture not in model_archs:
        raise Exception("Invalid architecture -- see supported architectures:  {}".format(list(model_archs.keys())))

    if args.hyperparameter_set not in hyperparameters:
        raise Exception("Hyperparameter combination is not defined in ./utils/model_library.py")

    if args.pipeline not in model_defs:
        raise Exception('Pipeline is not defined in ./utils/model_library.py')
    # create model instance
    if args.pipeline == 'Pipeline1':
        num_classes = training_sets[args.training_dir]['num_classes']
        model_ft = model_defs[args.pipeline][args.model_architecture](num_classes)

    else:
        model_ft = model_defs[args.pipeline][args.model_architecture]

    # check for GPU support and set model to evaluation mode
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model_ft.cuda()
    model_ft.eval()

    # load saved model weights from pt_train.py
    model_ft.load_state_dict(torch.load("./{}/{}/{}/{}.tar".format(args.dest_folder, args.pipeline, args.model_name,
                                                                   args.model_name)))
    # extract class_names

    # run validation to get confusion matrix
    predict_patch(model=model_ft, input_size=model_archs[args.model_architecture]['input_size'],
                  pipeline=args.pipeline, batch_size=hyperparameters[args.hyperparameter_set]['batch_size_test'],
                  test_dir=args.test_dir, out_file=args.model_name, dest_folder=args.dest_folder,
                  num_workers=hyperparameters[args.hyperparameter_set]['num_workers_train'])


if __name__ == '__main__':
    main()
