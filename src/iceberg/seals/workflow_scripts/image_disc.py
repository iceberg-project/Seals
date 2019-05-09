"""
Image Discovery Kernel
==========================================================

This script takes as input a path and returns a dataframe
with all the images and their size.

Author: Ioannis Paraskevakos
License: MIT
Copyright: 2018-2019
"""
from glob import glob
import argparse
import os
import math
import pandas as pd

from ..iceberg_zmq import Publisher

class Discovery(object):

    def __init__(self, name='simple',queue_out='simple',path=None):

        self._name = name
        self._path = path
        with open(queue_out) as fqueue:
            pub_addr_line, _ = fqueue.readlines()
        print(pub_addr_line)
        if pub_addr_line.startswith('PUB'):
            self._addr_in = pub_addr_line.split()[1]
        else:
            RuntimeError('Publisher address not specified in %s' % queue_out)

        self._publisher = Publisher(channel=self._name, url=self._addr_in)

        self.dataset = None

    def _image_discovery(self, filesize=True):
        """
        This function creates a dataframe with image names and size from a path.

        :Arguments:
            :path: Images path, str
            :filename: The filename of the CSV file containing the dataframe.
                    Default Value: list.csv
            :filesize: Whether or not the image sizes should be inluded to the
                    dataframe. Default value: False
        """

        filepaths = glob(self._path + '/*.tif')
        if filesize:
            dataset_df = pd.DataFrame(columns=['Filename', 'Size'])
            for filepath in filepaths:
                filesize = 0  # int(math.ceil(os.path.getsize(filepath)/1024/1024))
                dataset_df.loc[len(dataset_df)] = [filepath, filesize]
        else:
            dataset_df = pd.DataFrame(columns=['Filename'])
            for filepath in filepaths:
                dataset_df.loc[len(dataset_df)] = [filepath]

        dataset_df.sort_values(by='Size',axis=0,inplace=True)
        dataset_df.reset_index(drop='index',inplace=True)

        self.dataset = dataset_df

    def _connect(self):

        self._publisher.put(topic='request', msg={'name': self._name,
                                                  'request': 'connect',
                                                  'type': 'sender'})
    def _send_data(self):

        for path, _ in self.dataset.values[0:int(len(self.dataset.values)/2)]:
            print('image {request: enqueue, data: %s}' % path)
            self._publisher.put(topic='image', msg={'request': 'enqueue',
                                                    'data': path})

    def _disconnect(self):

        self._publisher.put(topic='request', msg={'name': self._name,
                                                  'type': 'sender',
                                                  'request': 'disconnect'})

    def run(self):

        self._connect()

        self._image_discovery()

        self._send_data()

        self._disconnect()

        return 0

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('name', type=str)
    parser.add_argument('queue_file', type=str)

    args = parser.parse_args()

    discovery = Discovery(name=args.name, queue_out=args.queue_file,
                          path=args.path)
    discovery.run()
