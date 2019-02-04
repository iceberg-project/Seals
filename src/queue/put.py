import zmq
import time
import msgpack
import os

# ------------------------------------------------------------------------------
#
class Putter():

    def __init__(self, name, url_folder='.'):

        self.addr = None
        with open('%s/%s.queue.url' % (url_folder, name), 'r') as fin:
            for line in fin.readlines():
                self.tag, self.addr = line.split()
                if self.tag == 'PUT':
                    break

        context    = zmq.Context()
        self.socket     = context.socket(zmq.PUSH)
        self.socket.hwm = 1
        self.socket.connect(self.addr)

    def put(self, msg):
        
        self.socket.send(msgpack.packb(msg))
        
if __name__ == '__main__':

    p = Putter('test')

    for i in range(20):
        msg = 'request %d data: %d' % (os.getpid(), i)
        p.put(msg)
        time.sleep(2)

    p.put('empty')