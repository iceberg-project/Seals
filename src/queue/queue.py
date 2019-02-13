"""
Project: ICEBERG Seals Project
Author: Ioannis Paraskevakos
License: MIT
Copyright: 2018-2019
"""
import time
import zmq


# ------------------------------------------------------------------------------
#
class Queue():
    """
    This class creates a queue and a communication protocol that allows the
    queue to know how many producers and consumer are connected to the queue.

    Each producer connects to the  queue and starts pushing data to consumers.

    When all producers have disconnected and the queue is empty then, consumers
    are informed to disconnect.
    """

    def __init__(self, name='simple'):
        """
        Contructor method. It instantiates a ZMQ channel to speak between
        processes and writes it in the filesystem.


        Parameters:
        **name** : The name of the queue.
        """

        self.delay = 0.0
        self._senders = 0
        self._receivers = 0
        self._queue = list()
        context = zmq.Context()
        self._socket = context.socket(zmq.REP)
        self._addr = None
        self._name = name

    def _configure(self, addr='*', port='*'):
        self._socket.bind("tcp://%s:%s" % (addr, port))

        self._addr = self._socket.getsockopt(zmq.LAST_ENDPOINT)

        with open('%s.queue.url' % self._name, 'w') as fout:
            fout.write('ADDR %s\n' % self._addr)

    def _connect(self, send_rec):
        """
        Adds a producer or a consumer to the queue.
        """

        if send_rec == 'sender':
            self._senders += 1
        elif send_rec == 'receiver':
            self._receivers += 1

    def _disconnect(self, send_rec):
        """
        Removes a producer or a consumer to the queue.
        """

        if send_rec == 'sender':
            self._senders -= 1
        elif send_rec == 'receiver':
            self._receivers -= 1

    def _check_status(self):

        """
        Reports the status of the queue
        """

        if self._queue or self._senders or self._receivers:
            return True

        return False

    def _enqueue(self, data):
        """
        Inserts an element in the queue
        """

        self._queue.append(data)

    def _dequeue(self):
        """
        Returns the top element of the queue
        """

        data = self._queue.pop(0)

        return data

    def run(self):
        """
        This is the main method that makes the queue run.

        Based on messages from producers and consumers, it connects a producer,
        or consumer, enqueues/dequeues data, and disconnects consumers.

        When all connections have been closed and no data exist in the queue,
        the process terminates and closes all channels.
        """

        status = True
        while status:
            message = self._socket.recv()
            print message, status
            if 'disconnect' in message:
                self._disconnect(message.split()[1])
            elif 'add' in message:
                self._enqueue(message.split()[1])
            elif 'dequeue' in message:
                send_message = self._dequeue()
                self._socket.send(send_message.encode('utf-8'))
            elif 'connect' in message:
                self._connect(message.split()[1])

            status = self._check_status()
            time.sleep(self.delay)
