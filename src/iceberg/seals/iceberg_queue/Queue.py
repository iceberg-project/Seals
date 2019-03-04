"""
Project: ICEBERG Seals Project
Author: Ioannis Paraskevakos
License: MIT
Copyright: 2018-2019
"""
import time
import random

from ..iceberg_zmq import PubSub, Publisher, Subscriber

# ------------------------------------------------------------------------------
#
class Queue(object):
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
        self._delay = 0.0
        self._enqueuers = list()
        self._dequeuers = list()
        self._queue = list()
        self._pub = None
        self._sub = None
        self._pubsub_bridge = None
        self._addr_in = None
        self._addr_out = None
        self._name = name

        self._configure()

    def _configure(self):

        self._pubsub_bridge = PubSub.create(cfg={'name': self._name,
                                                 'uid': self._name,
                                                 'kind':'pubsub'})

        self._addr_in = self._pubsub_bridge.addr_in
        self._addr_out = self._pubsub_bridge.addr_out

        with open('%s.queue.url' % self._name, 'w') as fout:
            fout.write('PUB: %s\n' % self._addr_in)
            fout.write('SUB: %s\n' % self._addr_out)

        self._pub = Publisher(channel=self._name, url=self._addr_in)
        self._sub = Subscriber(channel=self._name, url=self._addr_out)
        self._sub.subscribe(topic='request')
        self._sub.subscribe(topic='image')


    def _connect(self, send_rec, name):
        """
        Adds a producer or a consumer to the queue.
        """

        if send_rec == b'sender' and name not in self._enqueuers:
            self._enqueuers.append(name)
        elif send_rec == b'receiver' and name not in self._dequeuers:
            self._dequeuers.append(name)

    def _disconnect(self, send_rec, name):
        """
        Removes a producer or a consumer to the queue.
        """

        if send_rec == b'sender' and name in self._enqueuers:
            self._enqueuers.remove(name)
        elif send_rec == b'receiver' and name in self._dequeuers:
            self._dequeuers.remove(name)

    def _check_status(self):

        """
        Reports the status of the queue
        """

        if self._queue or self._enqueuers or self._dequeuers:
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

        if self._queue:
            data = self._queue.pop(0)
        elif not self._queue and self._enqueuers:
            data = 'wait'
        else:
            data = 'disconnect'

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
            topic, recv_message = self._sub.get()
            print(topic, recv_message)
            if topic == b'request':
                if recv_message[b'request'] == b'connect':
                    self._connect(recv_message[b'type'], recv_message[b'name'])
                elif recv_message[b'request'] == b'disconnect':
                    self._disconnect(recv_message[b'type'], recv_message[b'name'])
            elif topic == b'image':
                if recv_message[b'request'] == b'dequeue':
                    image = self._dequeue()
                    send_message = {'type': 'image', 'data': image}
                    send_topic = recv_message[b'name'].decode('utf-8')
                    print('Dequeueing', send_topic, image)
                    self._pub.put(topic=send_topic, msg=send_message)
                elif recv_message[b'request'] == b'enqueue':
                    self._enqueue(recv_message[b'data'])

            status = self._check_status()
            time.sleep(self._delay)
