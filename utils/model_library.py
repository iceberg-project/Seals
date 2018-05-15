# Script to store model architectures and hyperparameter combinations
__all__ = ['model_archs', 'training_sets', 'hyperparameters', 'cv_weights', 'model_defs', 'model_dataloaders']


from utils.custom_architectures import *
from utils.dataloaders import *
from torchvision import models, datasets

# architecture definitions with input size and whether the model is used at the haulout level or single seal level
model_archs = {'NasnetA': {'input_size': 299},
               'Resnet18': {'input_size': 224},
               'WideResnetA': {'input_size': 28},
               'CountCeption': {'input_size': 32},
               'NasnetACount': {'input_size': 299},
               'NasnetAe2e': {'input_size': 299}}

# model definitions
model_defs = {'Pipeline1': {'NasnetA': lambda num_classes: NASNetA(in_channels_0=48, out_channels_0=24,
                                                                   out_channels_1=32, out_channels_2=64,
                                                                   out_channels_3=128, num_classes=num_classes),
                            'Resnet18': lambda num_classes: models.resnet18(pretrained=False, num_classes=num_classes)}}

# model dataloaders
model_dataloaders = {'Pipeline1': lambda dataset, transforms: datasets.ImageFolder(dataset, transforms),
                     'Pipeline1.1': lambda dataset, transforms: ImageFolderTrainDet(dataset, transforms)}


# training sets with number of classes and size of scale bands
training_sets = {'training_set_vanilla': {'num_classes': 11, 'scale_bands': [450, 450, 450]},
                 'training_set_multiscale_A': {'num_classes': 11, 'scale_bands': [450, 1350, 4000]},
                 'training_set_vanilla_count': {'num_classes': 9, 'scale_bands': [450, 450, 450]},
                 'training_set_multiscale_A_count': {'num_classes': 9, 'scale_bands': [450, 1350, 4000]}}

# hyperparameter sets
hyperparameters = {'A': {'learning_rate': 1E-3, 'batch_size_train': 64, 'batch_size_val': 8, 'batch_size_test': 64,
                         'step_size': 1, 'gamma': 0.95, 'epochs': 5, 'num_workers_train': 8, 'num_workers_val': 1},
                   'B': {'learning_rate': 1E-3, 'batch_size_train': 16, 'batch_size_val': 1, 'batch_size_test': 8,
                         'step_size': 1, 'gamma': 0.95, 'epochs': 5, 'num_workers_train': 4, 'num_workers_val': 1},
                   'C': {'learning_rate': 1E-3, 'batch_size_train': 64, 'batch_size_val': 8, 'batch_size_test': 64,
                         'step_size': 1, 'gamma': 0.95, 'epochs': 30, 'num_workers_train': 16, 'num_workers_val': 8},
                   'D': {'learning_rate': 1E-3, 'batch_size_train': 32, 'batch_size_val': 8, 'batch_size_test': 32,
                         'step_size': 1, 'gamma': 0.95, 'epochs': 5, 'num_workers_train': 16, 'num_workers_val': 8},
                   'E': {'learning_rate': 1E-3, 'batch_size_train': 16, 'batch_size_val': 1, 'batch_size_test': 8,
                         'step_size': 1, 'gamma': 0.95, 'epochs': 10, 'num_workers_train': 4, 'num_workers_val': 1}
                   }

# cross-validation weights
cv_weights = {'NO': lambda x: [1] * x,
              'A': lambda x: [5] + [1] * (x-2) + [5]}





