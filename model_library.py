# Script to store model architectures and hyperparameter combinations


# architecture definitions, architecture name: input size
model_archs = {'NasnetA': 299, 'Resnet18': 224}

# training set classes
training_sets = {'training_set_vanilla': 11, 'training_set_multiscale_A': 11}

# hyperparameter sets
hyperparameters = {'A': {'learning_rate': 1E-3, 'batch_size_train': 64, 'batch_size_val': 8, 'batch_size_test': 64,
                         'step_size': 1, 'gamma': 0.95, 'epochs': 5},
                   'B': {'learning_rate': 1E-3, 'batch_size_train': 16, 'batch_size_val': 1, 'batch_size_test': 8,
                         'step_size': 1, 'gamma': 0.95, 'epochs': 5}}

# cross-validation weights
cv_weights = {'NO': [1 for ele in range(11)],
              'A': [5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5]}





