"""
Tile raster
==========================================================

Tiling script for ICEBERG seals use case. Tiles rasters into predefined size patches. 
Patches are named according to the raster indices that define their boundaries.
Optional arguments allow padding and multiple scale bands. Also saves a .csv for 
the raster's affine matrix -- used on 'predict_sealnet.py' to go from raster 
index to projected 'x' and 'y' of predicted seals.

Author: Bento Goncalves, Ioannis Paraskevakos
License: MIT
Copyright: 2018-2019
"""

import os
import argparse
import time
import random
# import numpy as np
import pandas as pd
import rasterio
import cv2

from ..iceberg_zmq import Publisher, Subscriber

class ImageTilling(object):

    def __init__(self, name, scale_bands, output_path, queue_in, queue_out):
        
        self._name = name
        self._output_path = output_path
        self._scale_bands = scale_bands

        with open(queue_in) as fqueue:
            pub_addr_line, sub_addr_line = fqueue.readlines()

            if pub_addr_line.startswith('PUB'):
                print(pub_addr_line)
                self._in_addr_in = pub_addr_line.split()[1]
            else:
                RuntimeError('Publisher address not specified in %s' % queue_in)

            if sub_addr_line.startswith('SUB'):
                print(sub_addr_line)
                self._in_addr_out = sub_addr_line.split()[1]
            else:
                RuntimeError('Subscriber address not specified in %s' % queue_in)

        with open(queue_out) as fqueue:
            pub_addr_line, _ = fqueue.readlines()

            if pub_addr_line.startswith('PUB'):
                print(pub_addr_line)
                self._out_addr_in = pub_addr_line.split()[1]
            else:
                RuntimeError('Publisher address not specified in %s' % queue_out)

        self._publisher_in = Publisher(channel=self._name, url=self._in_addr_in)
        self._subscriber_in = Subscriber(channel=self._name, url=self._in_addr_out)
        self._subscriber_in.subscribe(topic=self._name)
        self._publisher_out = Publisher(channel=self._name, url=self._out_addr_in)

    def _connect(self):

        self._publisher_in.put(topic='request', msg={'name': self._name,
                                                     'request': 'connect',
                                                     'type': 'receiver'})

        self._publisher_out.put(topic='request', msg={'name': self._name,
                                                      'request': 'connect',
                                                      'type': 'sender'})

    def _disconnect(self):
        
        self._publisher_in.put(topic='request', msg={'name': self._name,
                                                     'request': 'disconnect',
                                                     'type': 'receiver'})

        self._publisher_out.put(topic='request', msg={'name': self._name,
                                                      'request': 'disconnect',
                                                      'type': 'sender'})

    def _get_image(self):

        self._publisher_in.put(topic='image', msg={'request': 'dequeue',
                                                   'name': self._name})

        _, recv_message = self._subscriber_in.get()

        if recv_message[b'type'] == b'image':
            return recv_message[b'data'].decode('utf-8')

        return None

    def _tile_raster(self, input_image, output_folder, scales, pad_img=True):

        #time.sleep(random.randint(10,30))
        
        # time it
        tic = time.time()

        # read image
        with rasterio.open(input_image) as src:
            band = np.array(src.read()[0, :, :], dtype=np.uint8)
            # save affine matrix
            affine_matrix = pd.DataFrame({'transform': src.transform[:6]})
            affine_matrix.to_csv('%s/affine_matrix.csv' % output_folder)

        # add tiles subfolder
        output_folder = '%s/%s/' % (output_folder, os.path.basename(input_image))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # pad image
        pad = 0

        if pad_img:
            pad = scales[-1] // 2
            band = np.pad(band, pad_width=pad, mode='constant', constant_values=0)

        # extract tile size and raster size
        tile_size = scales[0]
        raster_width = band.shape[0]
        raster_height = band.shape[1]

        # tile out image
        count = 0
        for x in range(tile_size // 2 + pad, raster_width - pad, tile_size):
            for y in range(tile_size // 2 + pad, raster_height - pad, tile_size):
                scale_bands = []
                # find corners for polygon
                up = y - scales[0] // 2
                left = x - scales[0] // 2
                down = y + scales[0] // 2
                right = x + scales[0] // 2
                for scale in scales:
                    curr_scale = band[x - scale // 2: x + scale // 2, 
                                      y - scale // 2: y + scale // 2]
                    curr_scale = cv2.resize(curr_scale, (scales[0], scales[0]))
                    scale_bands.append(curr_scale)
                # remove black corners
                if np.max(scale_bands[0]) == 0:
                    continue
                # combine scales and save tile
                scale_bands = np.dstack(scale_bands)
                # save it with polygon coordinates
                filename = "%s/tile_%d_%d_%d_%d_.jpg" % (output_folder, up, left, 
                                                         down, right)
                cv2.imwrite(filename, scale_bands)
                count += 1
        toc = time.time()
        elapsed = toc - tic
        print('\n%d tiles created in %d minutes' % (count, int(elapsed // 60)) +
              ' and %.2f seconds' % elapsed % 60)

        self._publisher_out.put(topic='image', msg={'name': self._name,
                                                    'request': 'enqueue',
                                                    'data': output_folder})

    def run(self):

        self._connect()
        cont = True

        while cont:
            image = self._get_image()
            print(image)
            if image not in ['disconnect','wait']:
                self._tile_raster(input_image=image,
                                output_folder=self._output_path,
                                scales=self._scale_bands)
            elif image == 'wait':
                time.sleep(1)
            else:
                self._disconnect()
                cont = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='divides a raster image into \
                                                  files')
    parser.add_argument('--name', type=str)
    parser.add_argument('--scale_bands', type=int, help='for multi-scale models, \
                                                         string with size of \
                                                         scale bands separated \
                                                         by spaces')
    parser.add_argument('--output_folder', type=str, help='folder where tiles \
                                                           will be stored')
    parser.add_argument('--queue_in', type=str)
    parser.add_argument('--queue_out', tpye=str)

    args = parser.parse_args()

    tiler = ImageTilling(name=args.name, scale_bands=args.scale_bands,
                        output_path=args.output_folder, queue_in=args.queue_in,
                        queue_out=args.queue_out)
    tiler.run()