"""
Predict sealnet
==========================================================

CNN deployment script for ICEBERG seals use case. Splits a raster into tiles and predicts seal counts and locations in
tiles with 'predict_sealnet.py'.

Author: Bento Goncalves
License: MIT
Copyright: 2018-2019
"""

import torch
import warnings
import argparse
import os
import shutil
import time
import random
import json
from ..utils.model_library import *
from .predict_sealnet import predict_patch
from ..iceberg_zmq import Publisher, Subscriber
warnings.filterwarnings('ignore', module='PIL')

class SealnetPredict(object):

    def __init__(self, name, queue_in, cfg):
         
        self._name = name

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

        self._publisher_in = Publisher(channel=self._name, url=self._in_addr_in)
        self._subscriber_in = Subscriber(channel=self._name, url=self._in_addr_out)
        self._subscriber_in.subscribe(topic=self._name)
    
        with open(cfg) as conf:
            self._cfg = json.load(conf)

    def _connect(self):

        self._publisher_in.put(topic='request', msg={'name': self._name,
                                                     'request': 'connect',
                                                     'type': 'receiver'})

    def _disconnect(self):

        self._publisher_in.put(topic='request', msg={'name': self._name,
                                                     'type': 'receiver',
                                                     'request': 'disconnect'})

    def _get_image(self):

        self._publisher_in.put(topic='image', msg={'request': 'dequeue',
                                                   'name': self._name})

        _, recv_message = self._subscriber_in.get()

        if recv_message[b'type'] == b'image':
            return recv_message[b'data'].decode('utf-8')

        return None

    def _predict_raster(self, input_image, model_arch, training_set, model_path,
                        hyperparameter_set, test_folder, output_folder):
        #time.sleep(random.randint(10,30))
        # predict tiles
        pipeline = model_archs[model_arch]['pipeline']
        model = model_defs[pipeline][model_arch]

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            model.cuda()
        model.eval()

        # load saved model weights from training
        model_name = model_arch + '_ts-' + training_set.split('_')[-1]
        model.load_state_dict(
            torch.load("%s/%s.tar" % (model_path, model_name)))
        print('input_image=', input_image, 'test_dir=', test_folder, 'output_dir=', output_folder)
        predict_patch(input_image=input_image, model=model,
                      input_size=model_archs[model_arch]['input_size'],
                      batch_size=hyperparameters[hyperparameter_set]['batch_size_test'],
                      test_dir=test_folder,
                      output_dir=output_folder,
                      num_workers=hyperparameters[hyperparameter_set]['num_workers_train'])

    def run(self):

        self._connect()

        cont = True

        while cont:
            image = self._get_image()
            if image not in ['disconnect','wait']:
                print(image)
                self._predict_raster(input_image=image.split('/')[-1],
                                     model_arch=self._cfg['model_arch'],
                                     training_set=self._cfg['training_set'],
                                     model_path=self._cfg['model_path'],
                                     hyperparameter_set=self._cfg['hyperparameter_set'],
                                     test_folder=image[:-len(image.split('/')[-1])],
                                     output_folder=image.split('/')[-1].split('.')[0])
            elif image == 'wait':
                time.sleep(1)
            else:
                self._disconnect()
                cont = False


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='validates a CNN at the haul out level')
    parser.add_argument('--name', type=str)
    parser.add_argument('--queue_in', type=str)
    parser.add_argument('--config_file',type=str)
    args = parser.parse_args()

    pred = SealnetPredict(name=args.name, queue_in=args.queue_in, cfg=args.config_file)

    pred.run()
