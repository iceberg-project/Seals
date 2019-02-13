"""
Project: ICEBERG Seals Project
Author: Ioannis Paraskevakos
License: MIT
Copyright: 2018-2019
"""

import zmq
import msgpack


# ------------------------------------------------------------------------------
#
class Sender():
    """
    This class is a sender class to a queue. It connects to a queue and
    pushes data.
    """

    def __init__(self, name='test', url_folder='.'):
        """
        The constructor.

        Attributes:
        * socket *: The socket that the receiver is connected at.

        Parameters:
        * name *: The name of the queue that it will connect.
        * url_folder *: The folder where the file with the URL lies

        Returns: Nothing
        """

        self.addr = None
        with open('%s/%s.queue.url' % (url_folder, name), 'r') as fin:
            self.tag, self.addr = fin.readline().split()

        context    = zmq.Context()
        self.socket     = context.socket(zmq.PUSH)
        self.socket.hwm = 1
        self.socket.connect(self.addr)


    def connect(self):
        """
        This method is used to allow the sender class to connect to the queue
        and push data upon request.
        """

        msg = 'connect sender'

        self.socket.send(msgpack.packb(msg))

        return 0

    def disconnect(self):
        """
        This method is used to allow the sender class to connect to the queue
        and receive data upon request.
        """

        msg = 'disconnect sender'

        self.socket.send(msgpack.packb(msg))

        return 0

    def put(self, msg):
        """
        This is the put method. It sends a message to the queue
        """
        
        sending_msg = 'add ' + msg

        self.socket.send(msgpack.packb(sending_msg))
