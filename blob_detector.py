# Import packages
import cv2
import os
import pandas as pd
import numpy as np
from collections import defaultdict

# Setup the detector
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 8
params.maxArea = 35
params.thresholdStep = 18
params.filterByCircularity = True
params.minCircularity = 0.3
params.filterByConvexity = True
params.minConvexity = 0.6
params.filterByInertia = True
params.maxInertiaRatio = 0.1
params.maxInertiaRatio = 0.6
detector = cv2.SimpleBlobDetector_create(params)

# Classes with positive examples
pos_classes = ['crabeater', 'weddell']

# Save output in a dictionary
out = {}

# Navigate to folders with seals
for folder in pos_classes:
    for path, _, files in os.walk('./classified_images/{}/'.format(folder)):
        for idx, filename in enumerate(files):
            f = os.path.join(path, filename)
            try:
                image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                # Use detector to find keypoints in image
                keypoints = detector.detect(image=image)
                im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
                                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.imshow("Keypoints", im_with_keypoints)
                cv2.waitKey(0)
                #cv2.imwrite("{}_with_keypoints.jpg".format(f[:-3]))
                #out[filename] = keypoints
            except:
                print('blob_detector: cv2.imread failed - Removed: ' + f)
                #os.system("rm \"" + f + "\"")
                continue

