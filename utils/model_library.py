# Script to store model architectures and hyperparameter combinations
__all__ = ['model_archs', 'training_sets', 'hyperparameters', 'cv_weights', 'model_defs', 'dataloaders',
           'loss_functions']


from utils.custom_architectures import *
from utils.dataloaders import *
from torchvision import models, datasets
import torch
import torch.nn as nn

# architecture definitions with input size and whether the model is used at the haulout level or single seal level
model_archs = {'NasnetA': {'input_size': 299},
               'Resnet18': {'input_size': 224},
               'Resnet34': {'input_size': 224},
               'Resnet50': {'input_size': 224},
               'WideResnetA': {'input_size': 28},
               'WideResnetCount': {'input_size': 28},
               'WideResnetDet': {'input_size': 28},
               'Resnet18count': {'input_size': 224},
               'Resnet34count': {'input_size': 224},
               'Resnet50count': {'input_size': 224},
               'NasnetAcount': {'input_size': 224},
               'NasnetAe2e': {'input_size': 299},
               'CountCeption': {'input_size': 100},
               'Squeezenet11': {'input_size': 224},
               'Densenet121': {'input_size': 224},
               'Densenet169': {'input_size': 224},
               'Alexnet': {'input_size': 224},
               'VGG16': {'input_size': 224},
               'UnetDet': {'input_size': 224}
               }

# model definitions
model_defs = {'Pipeline1': {'NasnetA': lambda num_classes: NASNetA(in_channels_0=48, out_channels_0=24,
                                                                   out_channels_1=32, out_channels_2=64,
                                                                   out_channels_3=128, num_classes=num_classes),
                            'Resnet18': lambda num_classes: models.resnet18(pretrained=False, num_classes=num_classes),
                            'Resnet34': lambda num_classes: models.resnet34(pretrained=False, num_classes=num_classes),
                            'Resnet50': lambda num_classes: models.resnet50(pretrained=False, num_classes=num_classes),
                            'Densenet121': lambda num_classes: models.densenet121(pretrained=False,
                                                                                  num_classes=num_classes),
                            'Densenet169': lambda num_classes: models.densenet169(pretrained=False,
                                                                                  num_classes=num_classes),
                            'Alexnet': lambda num_classes: models.alexnet(pretrained=False,
                                                                          num_classes=num_classes),
                            'VGG16': lambda num_classes: models.vgg16(pretrained=False, num_classes=num_classes),
                            'Squeezenet11': lambda num_classes: models.squeezenet1_1(pretrained=False,
                                                                                     num_classes=num_classes)
                            },
              'Pipeline1.1': {'Resnet18count': resnet18_count(),
                              'Resnet34count': resnet34_count(),
                              'Resnet50count': resnet50_count(),
                              'NasnetAcount': NASNetA_count(),
                              'WideResnetCount': wrn_count(depth=28),
                              'CountCeption': ModelCountception()},
              'Pipeline1.2': {'WideResnetDet': wrn_det(depth=28),
                              'UnetDet': UNetDet(depth=28, scale=32)}}

# model dataloaders
dataloaders = {'Pipeline1': lambda dataset, transforms: datasets.ImageFolder(dataset, transforms),
               'Pipeline1.1': lambda dataset, shp_trans, int_trans: ImageFolderTrainDet(dataset, shp_trans, int_trans),
               'Pipeline1.2': lambda dataset, shp_trans, int_trans: ImageFolderTrainDet(dataset, shp_trans, int_trans)}

# model loss functions
loss_functions = {'Pipeline1': lambda weight: nn.CrossEntropyLoss(weight=torch.FloatTensor(weight)),
                  'Pipeline1.1': lambda _: nn.MSELoss(),
                  'Pipeline1.2': lambda _: nn.SmoothL1Loss()}

# training sets with number of classes and size of scale bands
training_sets = {'training_set_vanilla': {'num_classes': 11, 'scale_bands': [450, 450, 450]},
                 'training_set_multiscale_A': {'num_classes': 11, 'scale_bands': [450, 1350, 4000]},
                 'training_set_binary': {'num_classes': 2, 'scale_bands': [450, 450, 450]}
                 }

# hyperparameter sets
hyperparameters = {'A': {'learning_rate': 1E-3, 'batch_size_train': 64, 'batch_size_val': 8, 'batch_size_test': 64,
                         'step_size': 1, 'gamma': 0.95, 'epochs': 5, 'num_workers_train': 16, 'num_workers_val': 1},
                   'B': {'learning_rate': 1E-3, 'batch_size_train': 16, 'batch_size_val': 1, 'batch_size_test': 8,
                         'step_size': 1, 'gamma': 0.95, 'epochs': 5, 'num_workers_train': 8, 'num_workers_val': 1},
                   'C': {'learning_rate': 1E-3, 'batch_size_train': 64, 'batch_size_val': 8, 'batch_size_test': 64,
                         'step_size': 1, 'gamma': 0.95, 'epochs': 30, 'num_workers_train': 16, 'num_workers_val': 8},
                   'D': {'learning_rate': 1E-3, 'batch_size_train': 16, 'batch_size_val': 8, 'batch_size_test': 32,
                         'step_size': 1, 'gamma': 0.95, 'epochs': 5, 'num_workers_train': 8, 'num_workers_val': 8},
                   'E': {'learning_rate': 1E-3, 'batch_size_train': 8, 'batch_size_val': 1, 'batch_size_test': 4,
                         'step_size': 1, 'gamma': 0.95, 'epochs': 5, 'num_workers_train': 4, 'num_workers_val': 1}
                   }

# cross-validation weights
cv_weights = {'NO': lambda x: [1] * x,
              'WCV': lambda x: [5] + [1] * (x-2) + [5]}





