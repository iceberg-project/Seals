
#pylint: disable=protected-access, unused-argument

from src.queue.queue import Queue

try:
    import mock
except:
    from unittest import mock

# ------------------------------------------------------------------------------
#
def test_check_status():

    q = Queue(name='test')

    # Case 1: No Data, No Senders, No Receivers:
    assert(q._check_status() == False)

    # Case 2: No Data, No Senders, Receivers:
    q._receivers = 1
    assert(q._check_status() == True)
    
    # Case 3: No Data, Senders, No Receivers:
    q._receivers = 0
    q._senders = 1
    assert(q._check_status() == True)
    
    # Case 4: No Data, Senders, Receivers:
    q._receivers = 1
    assert(q._check_status() == True)

    # Case 5: Data, No Senders, No Receivers:
    q._queue.append(1)
    q._receivers = 0
    q._senders = 0
    assert(q._check_status() == True)

    # Case 6: Data, No Senders, Receivers:
    q._receivers = 1
    assert(q._check_status() == True)
    
    # Case 7: Data, Senders, No Receivers:
    q._receivers = 0
    q._senders = 1
    assert(q._check_status() == True)
    
    # Case 8: Data, Senders, Receivers:
    q._receivers = 1
    assert(q._check_status() == True)


