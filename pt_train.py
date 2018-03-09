import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import time
import os
import copy

# image transforms seem to cause truncated images, so we need this
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# we get an RGB warning, but the loader properly converts to RGB -after- this
import warnings

warnings.filterwarnings('ignore', module='PIL')

# Data augmentation and normalization for training
# Just normalization for validation
# IMPORTANT: set the correct dimension for your architecture using arch_input_size global variable
arch_input_size = 224
data_transforms = {
    'training': transforms.Compose([
        #transforms.Resize(int(arch_input_size * np.random.uniform(1.1, 1.3))),
        #transforms.RandomResizedCrop(arch_input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180, expand=True),
        transforms.CenterCrop(arch_input_size * 1.5),
        transforms.RandomResizedCrop(arch_input_size),
        transforms.ColorJitter(np.random.choice([0, 1]) * 0.05,
                               np.random.choice([0, 1]) * 0.05,
                               np.random.choice([0, 1]) * 0.05,
                               np.random.choice([0, 1]) * 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.CenterCrop(arch_input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = './nn_images'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['training', 'validation']}



# Force minibatches to have an equal representation amongst classes during training with a weighted sampler
def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


# For unbalanced dataset we create a weighted sampler
weights = make_weights_for_balanced_classes(image_datasets['training'].imgs, len(image_datasets['training'].classes))
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

# change batch size ot match number of GPU's being used?
dataloaders = {"training": torch.utils.data.DataLoader(image_datasets["training"], batch_size=16,
                                                       sampler=sampler, num_workers=1),
               "validation": torch.utils.data.DataLoader(image_datasets["validation"], batch_size=8,
                                                         num_workers=1)
               }
dataset_sizes = {x: len(image_datasets[x]) for x in ['training', 'validation']}
class_names = image_datasets['training'].classes

use_gpu = torch.cuda.is_available()

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            if phase == 'training':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'training':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # save the model
    torch.save(model, './nn_model.pth.tar')

    return model


def main():
    # loading the pretrained model and adding new classes to it
    model_ft = models.resnet152(pretrained=False, num_classes=5)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    if use_gpu:
        # i think we can set parallel GPU usage here. will test
        # http://pytorch.org/docs/master/nn.html
        # http://pytorch.org/docs/master/nn.html#dataparallel-layers-multi-gpu-distributed
        # The batch size should be larger than the number of GPUs used.
        # It should also be an integer multiple of the number of GPUs so that
        # each chunk is the same size (so that each GPU processes the same number of samples).
        # model_ft = nn.DataParallel(model_ft).cuda()
        model_ft = model_ft.cuda()

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss().cuda()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.5)

    # start training
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=100)


if __name__ == '__main__':
    main()
