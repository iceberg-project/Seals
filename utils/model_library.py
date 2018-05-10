# Script to store model architectures and hyperparameter combinations


# architecture definitions with input size and whether the model is used at the haulout level or single seal level
model_archs = {'NasnetA': {'input_size': 299, 'haulout': True},
               'Resnet18': {'input_size': 224, 'haulout': True},
               'WideResnetA': {'input_size': 28, 'haulout': False},
               'CountCeption': {'input_size': 32, 'haulout': False},
               'NasnetACount': {'input_size': 299, 'haulout': False},
               'NasnetAe2e': {'input_size': 299, 'haulout': False}}

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
#TODO: make cv_weights flexible to number of classes
cv_weights = {'NO': [1 for ele in range(9)],
              'A': [5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5]}





