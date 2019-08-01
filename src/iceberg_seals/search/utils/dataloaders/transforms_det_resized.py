import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
import cv2
import math
import random
from PIL import Image
from scipy import ndimage


class HideAndSeek(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.25, mean=(0.485, 0.456, 0.406), n_patches=16):
        assert np.sqrt(n_patches) % 1 == 0, 'invalid input value for n_patches, n_patches has to be quadratic'
        self.probability = probability
        self.mean = mean
        self.n_patches = n_patches

    def __call__(self, img, locations):
        img = np.array(img)
        if len(img.shape) < 3:
            img = img.reshape((img.shape[0], img.shape[1], 1))
        patch_size = int(img.shape[0] / np.sqrt(self.n_patches))
        n_bands = img.shape[2]

        for idx in range(self.n_patches):
            if np.random.rand() < self.probability:
                i = max(0, int(patch_size * (idx % np.sqrt(self.n_patches))))
                j = max(0, int(patch_size * (idx // np.sqrt(self.n_patches))))

                for band in range(n_bands):
                    img[i:i + patch_size, j: j + patch_size, band] = np.random.normal(self.mean[band] * 255,
                                                                                      self.mean[band] * 51,
                                                                                      (patch_size, patch_size))

                locations[i:i + patch_size, j: j + patch_size] = 0

        if n_bands == 1:
            img = img.reshape((img.shape[0], img.shape[1]))
        img = Image.fromarray(img)
        return img, locations


def get_xy_locs(array, count, min_dist=4):
    if count == 0:
        return np.array([])
    cols = array.shape[1]
    # flatten array, get rid of zeros and sort it
    flat = array.flatten()
    flat_order = (-flat).argsort()
    # find first zero and remove tail
    flat_order = flat_order[next((idx for idx, ele in enumerate(flat_order) if flat[ele]), None):]
    # check if detections are too close
    to_remove = []
    for idx, ele in enumerate(flat_order):
        if idx in to_remove:
            continue
        for idx2 in range(idx + 1, len(flat_order)):
            if np.linalg.norm(np.array([flat_order[idx] // cols, flat_order[idx] % cols]) -
                              np.array([flat_order[idx2] // cols, flat_order[idx2] % cols])) < min_dist:
                to_remove.append(idx2)
    flat_order = np.delete(flat_order, to_remove)
    # return x peaks
    return np.array([[x // cols, x % cols] for x in flat_order[:count]])


class ShapeTransform(object):
    """
    Class to ensure image and locations get the same transformations during training
    """
    def __init__(self, output_size, train):
        self.output_size = output_size
        self.train = train
        self.hide_and_seek = HideAndSeek()

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
            angle = np.random.uniform(0, 360)
            image = TF.rotate(image, angle, expand=True)
            locations = TF.rotate(locations, angle, expand=True)

            # center crop
            center_crop = transforms.CenterCrop(int(self.output_size * 1.5))
            image = center_crop(image)
            locations = center_crop(locations)

            # random resized crop
            i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.45, 0.8), ratio=(1, 1))
            image = TF.resized_crop(image, i, j, h, w, size=int(self.output_size))

            # get counts
            loc_map = TF.crop(locations, i, j, h, w)
            counts = np.float32(np.sum(np.not_equal(loc_map, 0)))
            locations = np.array(TF.resized_crop(locations, i, j, h, w, size=int(self.output_size)))

            # get locations
            if counts > 0:
                locs = get_xy_locs(locations, int(counts))
                #lbl, n_lbl = ndimage.label(locations)
                #locs = ndimage.center_of_mass(locations, lbl, [ele for ele in range(1, n_lbl + 1)])
                locations = np.zeros([self.output_size, self.output_size], dtype=np.uint8)
                for point in locs:
                    locations[int(round(point[0])), int(round(point[1]))] = 255

            # hide-and-seek
            if np.random.rand() > 0.1:
                image, locations = self.hide_and_seek(image, locations)
                counts = np.float32(np.sum(np.not_equal(locations, 0)))
            locations = Image.fromarray(locations)

        else:
            center_crop = transforms.CenterCrop(self.output_size)
            image = center_crop(image)
            locations = center_crop(locations)
            counts = np.float32(np.sum(np.not_equal(np.array(locations), 0)))

        # change locations to tensor
        locations = TF.to_tensor(locations)

        return image, locations, counts
