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
from seals.iceberg_queue.Queue import Queue


# ------------------------------------------------------------------------------
#
@mock.patch.object(Queue, '_configure', return_value=None)
def test_init(mocked_configure):
    """
    Test the constructor
    """
    q = Queue()

    assert q._enqueuers == list()
    assert q._delay == 0.0
    assert q._dequeuers == list()
    assert q._queue == list()
    assert q._pub == None
    assert q._sub == None
    assert q._pubsub_bridge == None
    assert q._addr_in == None
    assert q._addr_out == None
    assert q._name == 'simple'

    q = Queue(name='test')

    assert q._enqueuers == list()
    assert q._delay == 0.0
    assert q._dequeuers == list()
    assert q._queue == list()
    assert q._pub == None
    assert q._sub == None
    assert q._pubsub_bridge == None
    assert q._addr_in == None
    assert q._addr_out == None
    assert q._name == 'test'

# ------------------------------------------------------------------------------
#
@mock.patch.object(Queue, '__init__', return_value=None)
def test_connect(mocked_init):
    """
    Test _connect method
    """

    q = Queue()
    q._enqueuers = []
    q._dequeuers = []

    num_senders = random.randint(0, 100)
    senders = list()
    for i in range(num_senders):
        q._connect(send_rec=b'sender',name='test%d' % i)
        senders.append('test%d' % i)
    assert q._enqueuers == senders

    num_receivers = random.randint(0, 100)
    receivers = list()
    for i in range(num_receivers):
        q._connect(send_rec=b'receiver',name='test%d' % i)
        receivers.append('test%d' % i)
    assert q._dequeuers == receivers

    q._enqueuers = []
    q._dequeuers = []

    num_senders = random.randint(0, 100)
    for i in range(num_senders):
        q._connect(send_rec=b'sender',name='test')
    assert q._enqueuers == ['test']

    num_receivers = random.randint(0, 100)
    for i in range(num_receivers):
        q._connect(send_rec=b'receiver',name='test')
    assert q._dequeuers == ['test']

# ------------------------------------------------------------------------------
#
@mock.patch.object(Queue, '__init__', return_value=None)
def test_disconnect(mocked_init):
    """
    Test _disconnect method
    """

    q = Queue()
    num_senders = random.randint(50, 100)
    q._enqueuers = ['test%d' % i for i in range(num_senders)]
    senders = ['test%d' % i for i in range(num_senders)]
    num_receivers = random.randint(50, 100)
    q._dequeuers = ['test%d' % i for i in range(num_receivers)]
    receivers = ['test%d' % i for i in range(num_receivers)]

    remove_senders = random.randint(1, num_senders)
    for i in range(remove_senders):
        q._disconnect(send_rec=b'sender', name='test%d' % i)
        senders.remove('test%d' % i)
    assert q._enqueuers == senders

    
    for i in range(remove_senders):
        q._disconnect(send_rec=b'sender', name='test%d' % i)
    assert q._enqueuers == senders

    remove_receivers = random.randint(1, num_receivers)
    for i in range(remove_receivers):
        q._disconnect(send_rec=b'receiver', name='test%d' % i)
        receivers.remove('test%d' % i)
    assert q._dequeuers == receivers

    for i in range(remove_receivers):
        q._disconnect(send_rec=b'receiver', name='test%d' % i)
    assert q._dequeuers == receivers


# ------------------------------------------------------------------------------
#
@mock.patch.object(Queue, '__init__', return_value=None)
def test_check_status(mocked_init):
    """
    Test _check_status method
    """

    q = Queue()
    q._enqueuers = list()
    q._dequeuers = list()
    q._queue = list()

    # Case 1: No Data, No Senders, No Receivers:
    assert not q._check_status()

    # Case 2: No Data, No Senders, Receivers:
    q._dequeuers = ['test']
    assert q._check_status()

    # Case 3: No Data, Senders, No Receivers:
    q._enqueuers = list()
    q._dequeuers = ['test']
    assert q._check_status()

    # Case 4: No Data, Senders, Receivers:
    q._dequeuers = ['test']
    assert q._check_status()

    # Case 5: Data, No Senders, No Receivers:
    q._queue.append(1)
    q._dequeuers = list()
    q._senders = list()
    assert q._check_status()

    # Case 6: Data, No Senders, Receivers:
    q._dequeuers = ['test']
    assert q._check_status()

    # Case 7: Data, Senders, No Receivers:
    q._dequeuers = list()
    q._senders = ['test']
    assert q._check_status()

    # Case 8: Data, Senders, Receivers:
    q._dequeuers = ['test']
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

    for i in range(random.randint(0, 100)):
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
    data = [random.randint(0, 100) for _ in range(random.randint(0, 1000))]
    q._queue = list(data)

    for _ in range(random.randint(0, len(data))):
        test_datum = q._dequeue()
        datum = data.pop(0)
        assert test_datum == datum
