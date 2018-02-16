# Tyler Estro - Stony Brook University 02/06/18
#
# Recursively removes files that can not be opened with PIL.Image
# Assumes training images are located in ./nn_images
#
# Usage: python prep_train.py

# note: i'd like to add duplicate image detection as well

import os
from PIL import Image

for path, subdirs, files in os.walk('./nn_images'):
    for filename in files:
        f = os.path.join(path, filename)
        try:
            image = Image.open(f)
        except:
            print('prep_train: PIL.Image.open failed - Removed: ' + f)
            os.system("rm \"" + f + "\"")
            continue
print("prep_train: Removal of incompatible files completed")
