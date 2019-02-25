"""
Project: ICEBERG Seals Project
Author: Ioannis Paraskevakos
License: MIT
Copyright: 2018-2019
"""

import zmq


# ------------------------------------------------------------------------------
#
class Receiver():
    """
    This class is a receiver class from a queue. It connects to a queue and
    requests data.
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

        self.DELAY = 0.0

        self.addr = None
        with open('%s/%s.queue.url' % (url_folder, name), 'r') as fin:
            self.tag, self.addr = fin.readline().split()

        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.hwm = 1
        self.socket.connect(self.addr)

    def connect(self):
        """
        This method is used to allow the receiver class to connect to the queue
        and receive data upon request.
        """

        msg = 'connect receiver'

        self.socket.send(msgpack.packb(msg))

        return 0

    def disconnect(self):
        """
        This method is used to allow the receiver class to connect to the queue
        and receive data upon request.
        """

        msg = 'disconnect receiver'

        self.socket.send(msgpack.packb(msg))

        return 0

    def get(self, req):
        """
        This is the get method. It sends a request out for receiving a data
        point and returns the message
        """

        self.socket.send(req)
        msg = msgpack.unpackb(self.socket.recv())
        return msg
