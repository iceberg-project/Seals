import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import torchvision
from utils.dataloaders import ImageFolderTrainDet
from utils.dataloaders.transforms_det_resized import ShapeTransform


def imshow(inp, title=None, rgb=True):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    if not rgb:
        plt.imshow(inp, cmap='gray')
    else:
        plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# create dataloader
training_set = 'training_set_vanilla'
data_dir = '/home/bento/Seals/training_sets/{}'.format(training_set)
arch_input_size = 224

data_transforms = {
    'training': {'shape_transform': ShapeTransform(arch_input_size, train=True),
                 'int_transform': transforms.Compose([
                     transforms.ColorJitter(brightness=np.random.choice([0, 1]) * 0.05,
                                            contrast=np.random.choice([0, 1]) * 0.05),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485], [0.229])])},
    'validation': {'shape_transform': ShapeTransform(arch_input_size, train=False),
                   'int_transform': transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize([0.485], [0.229])])},
}
image_datasets = {x: ImageFolderTrainDet(root=os.path.join(data_dir, x),
                                         shape_transform=data_transforms[x]['shape_transform'],
                                         int_transform=data_transforms[x]['int_transform'],
                                         training_set=training_set)
                  for x in ['training', 'validation']}
# sampler
# Force minibatches to have an equal representation amongst classes during training with a weighted sampler


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = (N / float(count[i]))
    weight = [0] * len(images)
    # give more weight to seals
    weight_per_class[0] = weight_per_class[0] * 2.5
    weight_per_class[10] = weight_per_class[10] * 2.5
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


# For unbalanced dataset we create a weighted sampler
weights = make_weights_for_balanced_classes(image_datasets['training'].imgs, len(image_datasets['training'].classes))
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, 10000, replacement=False)
# Get a batch of training data
dataloaders = {"training": torch.utils.data.DataLoader(image_datasets["training"],
                                                       batch_size=4,
                                                       sampler=sampler, num_workers=1),
               "validation": torch.utils.data.DataLoader(image_datasets["validation"],
                                                         batch_size=4,
                                                         num_workers=1,
                                                         shuffle=True)}

# inputs contains 4 images because batch_size=4 for the dataloaders
inputs, classes, counts, locations = next(iter(dataloaders['training']))

# Make a grid from batch
out_img = torchvision.utils.make_grid(inputs)
out_location = torchvision.utils.make_grid(locations)

imshow(out_img)
imshow(out_location, rgb=False)
diff = 0
for i in range(len(counts)):
    diff += int(counts[i]) - np.sum(np.array(locations[i]))
    print('img {}, class = {}, counts: {}, sum: {}'.format(i, classes[i], int(counts[i]), np.sum(np.array(locations[i]))))

print('total difference :', diff)
