import torch
from torch.autograd import Variable
from torchvision import transforms
import os
import numpy as np
from PIL import Image

import argparse

parser = argparse.ArgumentParser(description='predict new images using a previously trained model')
parser.add_argument('-class_names', nargs="*", help='list of classes to make predictions from')
args = parser.parse_args()

# image transforms seem to cause truncated images, so we need this
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Just normalization for prediction
# IMPORTANT: set the correct dimension for your architecture using arch_input_size global variable
arch_input_size = 224
data_transforms = transforms.Compose([
    transforms.Resize(int(arch_input_size * np.random.uniform(1.05, 1.1))),
    transforms.CenterCrop(arch_input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img_exts = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
class_names = args.class_names
data_dir = 'test_images'

use_gpu = torch.cuda.is_available()


def main():
    # load saved model from pt_train.py
    nn_model = torch.load('./nn_model.pth.tar')
    # make sure it isn't in training mode
    nn_model.eval()
    # recursively iterate over all files in the base folder and classify them
    for path, subdirs, files in os.walk(data_dir):
        for filename in files:
            filename_lower = filename.lower()
            if not (any(filename_lower.endswith(ext) for ext in img_exts)):
                print('{} is not a valid image file.'.format(f))
                continue
            try:
                f = os.path.join(path, filename)
                image = Image.open(f)
                img_tensor = data_transforms(image)
                img_tensor.unsqueeze_(0)
                if use_gpu:
                    img_variable = Variable(img_tensor.cuda())
                else:
                    img_variable = Variable(img_tensor)
                output = nn_model(img_variable)
                _, preds = torch.max(output.data, 1)
                print('predicted {} as: {}'.format(filename, class_names[preds[0]]))
                image.save('classified_images/{}/{}'.format(class_names[preds[0]], filename))
            except Exception as e:
                print('{} is invalid: {}'.format(f, e))
                continue


if __name__ == '__main__':
    main()
