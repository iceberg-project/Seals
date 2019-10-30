import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from utils.getxy_max import getxy_max


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


class ShapeTransform(object):
    """
    Class to ensure image and locations get the same transformations during training
    """

    def __init__(self, output_size, train, kernel_size=5):
        self.kernel_size = kernel_size
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
                locs = getxy_max(locations, int(counts))
                locations = np.zeros([self.output_size, self.output_size], dtype=np.uint8)
                for point in locs:
                    locations[int(round(point[0])), int(round(point[1]))] = 255

            # hide-and-seek
            if np.random.rand() > 0.9:
                image, locations = self.hide_and_seek(image, locations)
                counts = np.float32(np.sum(np.not_equal(locations, 0)))
            

        else:
            center_crop = transforms.CenterCrop(self.output_size)
            image = center_crop(image)
            locations = np.array(center_crop(locations))
            counts = np.float32(np.sum(np.not_equal(np.array(locations), 0)))

        # add Gaussian kernel to locations and transform to tensor
        locations = cv2.GaussianBlur(locations, (5, 5), 0)
        locations = Image.fromarray(locations)
        locations = TF.to_tensor(locations)

        return image, locations, counts
