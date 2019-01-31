# modified from torchvision/datasets/folder.py

import torch.utils.data as data

from PIL import Image

import os
import os.path
import pandas as pd
import numpy as np


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions, training_set):
    images = []
    locations = []
    dir = os.path.expanduser(dir)
    det_df = pd.read_csv('./training_sets/{}/detections.csv'.format(training_set), index_col=0)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    images.append((path, class_to_idx[target]))
                    # get locations
                    locs = det_df.loc[int(fname.split('.')[0]), 'locations']
                    if type(locs) == str:
                        locs = [int(ele) for ele in locs.split("_")]
                        locs = np.array([(locs[i+1], locs[i]) for i in range(0, len(locs)-1, 2)]).reshape(-1, 2)
                    else:
                        locs = []
                    locations.append(locs)

    return [images, locations]


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, loader, extensions, training_set, shape_transform=None, int_transform=None, img_dim=450):
        classes, class_to_idx = find_classes(root)
        samples, locations = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.loader = loader
        self.extensions = extensions
        self.img_dim = img_dim

        self.samples = samples
        self.locations = locations

        self.shape_transform = shape_transform
        self.int_transform = int_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        locs = self.locations[index]
        sample = self.loader(path)
        hit_value = 255

        locations = np.zeros([self.img_dim, self.img_dim], dtype=np.uint8)
        for ele in locs:
            locations[ele[0], ele[1]] = hit_value
        locations = Image.fromarray(locations)

        if self.shape_transform is not None:
            sample, locations = self.shape_transform(sample, locations)

        count = np.float32(np.sum(np.not_equal(locations.numpy(), 0)))

        if self.int_transform is not None:
            sample = self.int_transform(sample)

        return sample, target, count

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Shape Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.shape_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Intensity Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.int_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolderTrainCount(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, shape_transform=None, int_transform=None,
                 loader=default_loader):
        super(ImageFolderTrainCount, self).__init__(root, loader, IMG_EXTENSIONS,
                                                    shape_transform=shape_transform,
                                                    int_transform=int_transform)
        self.imgs = self.samples

