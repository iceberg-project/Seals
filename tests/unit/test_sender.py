"""
Project: ICEBERG Seals Project
Author: Ioannis Paraskevakos
License: MIT
Copyright: 2018-2019
"""
# pylint: disable=protected-access, unused-argument, unused-import

import os
import shutil
# import mock
import zmq
from src.queue.sender import Sender


# ------------------------------------------------------------------------------
#
def test_init():
    """
    Test the constructor
    """

    with open('test.queue.url', 'w') as fin:
        fin.write('ADDR tcp://0.0.0.0:8888\n')

    send = Sender()

    assert send.tag == 'ADDR'
    assert isinstance(send.socket, zmq.sugar.socket.Socket)
    assert send.addr == 'tcp://0.0.0.0:8888'
    os.remove('test.queue.url')

    os.mkdir('test_url')

    with open('test_url/test2.queue.url', 'w') as fin:
        fin.write('ADDR tcp://127.0.0.1:5555\n')

    send = Sender(name='test2', url_folder='test_url/')

    assert send.tag == 'ADDR'
    assert isinstance(send.socket, zmq.sugar.socket.Socket)
    assert send.addr == 'tcp://127.0.0.1:5555'
    shutil.rmtree('test_url')
