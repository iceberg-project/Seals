# Script to store model architectures and hyperparameter combinations


# architecture definitions
model_archs = ['Nasnet1', 'Resnet18']


# hyperparameter sets
hyperparameters = {'A': {'learning_rate': 1E-3, 'batch_size_train': 64, 'batch_size_val': 8, 'step_size': 1,
                         'gamma': 0.95, 'epochs': 25},
                   'B': {'learning_rate': 1E-3, 'batch_size_train': 16, 'batch_size_val': 1, 'step_size': 1,
                         'gamma': 0.95, 'epochs': 25}}

# cross-validation weights
cv_weights = {'NO': [1 for ele in range(11)],
              'A': [5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5]}





