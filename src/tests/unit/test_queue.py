"""
Project: ICEBERG Seals Project
Author: Ioannis Paraskevakos
License: MIT
Copyright: 2018-2019
"""
# pylint: disable=protected-access, unused-argument, unused-import

import mock
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
    assert q._socket is None
    assert q._addr is None
    assert q._queue == []

    q = Queue(name='test')

    assert q.delay == 0
    assert q._name == 'test'
    assert q._senders == 0
    assert q._receivers == 0
    assert q._socket is None
    assert q._addr is None
    assert q._queue == []


# ------------------------------------------------------------------------------
#
def test_check_status():
    """
    Test _check_status method
    """

    q = Queue(name='test')

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
