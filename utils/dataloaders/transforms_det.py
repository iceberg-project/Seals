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
            center_crop = transforms.CenterCrop(int(self.output_size * 1.5))
            image = center_crop(image)
            locations = center_crop(locations)

            if np.random.random() > 1:
                # random crop
                i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.2, 0.9), ratio=(1, 1))
                image = TF.resized_crop(image, i, j, h, w, size=int(self.output_size))
                # get counts
                counts = np.float32(np.sum(np.not_equal(TF.crop(locations, i, j, h, w), 0)))
                locations = TF.resized_crop(locations, i, j, h, w, size=int(self.output_size))

                #if counts > 0:
                #    # update locations to have only maximum values
                #    locs = get_xy_locs(locations, int(counts))
                #    locations = np.zeros([self.output_size, self.output_size])
                #    for point in locs:
                #        locations[point[0], point[1]] = 1
                #locations = Image.fromarray(locations)

            else:
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.output_size, self.output_size))
                image = TF.crop(image, i, j, h, w)
                # get counts
                counts = np.float32(np.sum(np.not_equal(np.array(TF.crop(locations, i, j, h, w)), 0)))
                locations = TF.crop(locations, i, j, h, w)

        else:
            center_crop = transforms.CenterCrop(self.output_size)
            image = center_crop(image)
            locations = center_crop(locations)
            counts = np.float32(np.sum(np.not_equal(np.array(locations), 0)))

        # change locations to tensor
        locations = TF.to_tensor(locations) 

        return image, locations, counts
