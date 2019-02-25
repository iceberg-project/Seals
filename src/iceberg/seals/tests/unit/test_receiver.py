"""
Project: ICEBERG Seals Project
Author: Ioannis Paraskevakos
License: MIT
Copyright: 2018-2019
"""
# pylint: disable=protected-access, unused-argument, unused-import

import os
import shutil
import mock
import zmq
import msgpack
from src.queue.receiver import Receiver


# ------------------------------------------------------------------------------
#
def test_init():
    """
    Test the constructor
    """

    with open('test.queue.url', 'w') as fin:
        fin.write('ADDR tcp://0.0.0.0:8888\n')

    rec = Receiver()

    assert rec.DELAY == 0.0
    assert rec.tag == 'ADDR'
    assert isinstance(rec.socket, zmq.sugar.socket.Socket)
    assert rec.addr == 'tcp://0.0.0.0:8888'
    os.remove('test.queue.url')

    os.mkdir('test_url')

    with open('test_url/test2.queue.url', 'w') as fin:
        fin.write('ADDR tcp://127.0.0.1:5555\n')

    rec = Receiver(name='test2', url_folder='test_url/')

    assert rec.DELAY == 0.0
    assert rec.tag == 'ADDR'
    assert isinstance(rec.socket, zmq.sugar.socket.Socket)
    assert rec.addr == 'tcp://127.0.0.1:5555'
    shutil.rmtree('test_url')


# -----------------------------------------------------------------------------
#
@mock.patch.object(Receiver, '__init__', return_value=None)
def test_get(mocked_init):
    """
    Test the constructor
    """

    mock_socket = mock.Mock()
    mock_socket.recv.return_value = msgpack.packb('Hello')
    rec = Receiver()
    rec.socket = mock_socket
    message = rec.get('dequeue')
    assert message == 'Hello'
