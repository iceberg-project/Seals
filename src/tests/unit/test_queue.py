"""
Project: ICEBERG Seals Project
Author: Ioannis Paraskevakos
License: MIT
Copyright: 2018-2019
"""
# pylint: disable=protected-access, unused-argument, unused-import

import os
import random
import mock
import zmq
from src.queue.queue import Queue


# ------------------------------------------------------------------------------
#
def test_init():
    """
    Test the constructor
    """
    q = Queue()

    assert q.delay == 0
    assert q._name == 'simple'
    assert q._senders == 0
    assert q._receivers == 0
    assert isinstance(q._socket, zmq.sugar.socket.Socket)
    assert q._addr is None
    assert q._queue == []

    q = Queue(name='test')

    assert q.delay == 0
    assert q._name == 'test'
    assert q._senders == 0
    assert q._receivers == 0
    assert isinstance(q._socket, zmq.sugar.socket.Socket)
    assert q._addr is None
    assert q._queue == []


# ------------------------------------------------------------------------------
#
@mock.patch.object(Queue, '__init__', return_value=None)
def test_configure(mocked_init):
    """
    Test _configure method
    """

    q = Queue()
    q._socket = zmq.Context().socket(zmq.REP)
    q._name = 'test'
    q._addr = None
    q._configure(addr='0.0.0.0', port='8888')

    assert q._addr == 'tcp://0.0.0.0:8888'
    assert os.path.isfile('test.queue.url')

    with open('test.queue.url', 'r') as addr_file:
        line = addr_file.readline()

    assert line == 'ADDR tcp://0.0.0.0:8888\n'


# ------------------------------------------------------------------------------
#
@mock.patch.object(Queue, '__init__', return_value=None)
def test_connect(mocked_init):
    """
    Test _connect method
    """

    q = Queue()
    q._senders = 0
    q._receivers = 0

    num_senders = random.randint(0, 100)
    for _ in xrange(num_senders):
        q._connect(send_rec='sender')
    assert q._senders == num_senders

    num_receivers = random.randint(0, 100)
    for _ in xrange(num_receivers):
        q._connect(send_rec='receiver')
    assert q._receivers == num_receivers


# ------------------------------------------------------------------------------
#
@mock.patch.object(Queue, '__init__', return_value=None)
def test_disconnect(mocked_init):
    """
    Test _disconnect method
    """

    q = Queue()
    num_senders = random.randint(50, 100)
    q._senders = num_senders
    num_receivers = random.randint(50, 100)
    q._receivers = num_receivers

    remove_senders = random.randint(0, num_senders)
    for _ in xrange(remove_senders):
        q._disconnect(send_rec='sender')
    assert q._senders == num_senders - remove_senders

    remove_receivers = random.randint(0, num_receivers)
    print num_receivers, remove_receivers
    for _ in xrange(remove_receivers):
        q._disconnect(send_rec='receiver')
    assert q._receivers == num_receivers - remove_receivers


# ------------------------------------------------------------------------------
#
@mock.patch.object(Queue, '__init__', return_value=None)
def test_check_status(mocked_init):
    """
    Test _check_status method
    """

    q = Queue(name='test')
    q._senders = 0
    q._receivers = 0
    q._queue = list()

    # Case 1: No Data, No Senders, No Receivers:
    assert not q._check_status()

    # Case 2: No Data, No Senders, Receivers:
    q._receivers = 1
    assert q._check_status()

    # Case 3: No Data, Senders, No Receivers:
    q._receivers = 0
    q._senders = 1
    assert q._check_status()

    # Case 4: No Data, Senders, Receivers:
    q._receivers = 1
    assert q._check_status()

    # Case 5: Data, No Senders, No Receivers:
    q._queue.append(1)
    q._receivers = 0
    q._senders = 0
    assert q._check_status()

    # Case 6: Data, No Senders, Receivers:
    q._receivers = 1
    assert q._check_status()

    # Case 7: Data, Senders, No Receivers:
    q._receivers = 0
    q._senders = 1
    assert q._check_status()

    # Case 8: Data, Senders, Receivers:
    q._receivers = 1
    assert q._check_status()

# ------------------------------------------------------------------------------
#
@mock.patch.object(Queue, '__init__', return_value=None)
def test_enqueue(mocked_init):
    """
    Test _enqueue method
    """

    q = Queue()
    q._queue = list()
    test_list = list()

    for _ in xrange(random.randint(0, 100)):
        data = random.randint(0, 1000)
        q._enqueue(data)
        test_list.append(data)

    assert q._queue == test_list

# ------------------------------------------------------------------------------
#
@mock.patch.object(Queue, '__init__', return_value=None)
def test_dequeue(mocked_init):
    """
    Test _dequeue method
    """

    q = Queue()
    data = [random.randint(0, 100) for _ in xrange(random.randint(0, 1000))]
    q._queue = list(data)

    for _ in xrange(random.randint(0, len(data))):
        test_datum = q._dequeue()
        datum = data.pop(0)
        assert test_datum == datum
    