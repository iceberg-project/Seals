import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np


class ShapeTransform(object):
    """
    Class to ensure image and locations get the same transformations during training
    """
    def __init__(self, output_size, train):
        self.output_size = output_size
        self.train = train

    # random rotation
    def __call__(self, image, locations):
        if self.train:
            #  left-right mirroring
            if np.random.random() > 0.5:
                image = TF.hflip(image)
                locations = TF.hflip(locations)

            #  left-right mirroring
            if np.random.random() > 0.5:
                image = TF.vflip(image)
                locations = TF.vflip(locations)

            # random rotation
            angle = np.random.uniform(-180, 180)
            image = TF.rotate(image, angle, expand=True)
            locations = TF.rotate(locations, angle, expand=True)

            # center crop
            center_crop = transforms.CenterCrop(self.output_size * 1.5)
            image = center_crop(image)
            locations = center_crop(locations)

            # random crop
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.output_size, self.output_size))
            image = TF.crop(image, i, j, h, w)
            locations = TF.crop(locations, i, j, h, w)

        else:
            center_crop = transforms.CenterCrop(self.output_size)
            image = center_crop(image)
            locations = center_crop(locations)

        # change locations to tensor
        locations = TF.to_tensor(locations)

        return image, locations
