import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
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
arch_input_size = 299
data_transforms = {
    'training': transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180, expand=True),
        transforms.CenterCrop(arch_input_size * 1.5),
        transforms.RandomResizedCrop(arch_input_size),
        transforms.ColorJitter(brightness = np.random.choice([0, 1]) * 0.05,
                               contrast = np.random.choice([0, 1]) * 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.5, ], [0.5, ])
    ]),
    'validation': transforms.Compose([
        transforms.Grayscale(),
        transforms.CenterCrop(arch_input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, ], [0.5, ])
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


class InceptionA(nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(88, 40, kernel_size=3)

        self.incept1 = InceptionA(in_channels=10)
        self.incept2 = InceptionA(in_channels=20)
        self.incept3 = InceptionA(in_channels=40)

        self.dropout = nn.Dropout2d(p=0.2)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(10240, 5)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incept1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incept2(x)
        x = F.relu(self.mp(self.conv3(x)))
        x = self.incept3(x)
        x = F.relu(self.mp(self.conv3(x)))
        x = self.incept3(x)
        x = F.relu(self.mp(self.conv3(x)))
        x = self.incept3(x)
        x = F.relu(self.mp(self.conv3(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.dropout(x)
        return F.log_softmax(x)




dataset_sizes = {x: len(image_datasets[x]) for x in ['training', 'validation']}
class_names = image_datasets['training'].classes

use_gpu = torch.cuda.is_available()

def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            if phase == 'training':
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
    model = Net()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    if use_gpu:
        # i think we can set parallel GPU usage here. will test
        # http://pytorch.org/docs/master/nn.html
        # http://pytorch.org/docs/master/nn.html#dataparallel-layers-multi-gpu-distributed
        # The batch size should be larger than the number of GPUs used.
        # It should also be an integer multiple of the number of GPUs so that
        # each chunk is the same size (so that each GPU processes the same number of samples).
        # model_ft = nn.DataParallel(model_ft).cuda()
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss().cuda()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.95)

    # start training
    model = train_model(model, criterion, optimizer_ft, num_epochs=100)


if __name__ == '__main__':
    main()

