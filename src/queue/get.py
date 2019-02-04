import os
import zmq
import time
import msgpack


# ------------------------------------------------------------------------------
#
class Getter():

    def __init__(self, name, url_folder='.'):
        self.DELAY = 0.0

        self.addr = None
        with open('%s/%s.queue.url' % (url_folder, name), 'r') as fin:
            for line in fin.readlines():
                self.tag, self.addr = line.split()
                if self.tag == 'GET':
                    break


        context    = zmq.Context()
        self.socket     = context.socket(zmq.REQ)
        self.socket.hwm = 1
        self.socket.connect(self.addr)

    def get(self, req):
        self.socket.send(req)
        msg = msgpack.unpackb(self.socket.recv())
        return msg
        


# ------------------------------------------------------------------------------
if __name__ == '__main__':

    g = Getter('test')
    time.sleep(10)
    receiving = True
    while receiving:
        msg = g.get('request %d' % os.getpid())
        if msg == 'empty':
            receiving = False
        else:
            print msg
            time.sleep(5)