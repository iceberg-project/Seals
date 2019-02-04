import zmq
import time
import msgpack

# ------------------------------------------------------------------------------
#
class Queue():

    def __init__(self, name=None):
        
        self.delay = 0.0
        self._senders = 0
        self._receivers = 0
        
        # src side is proper push/pull (queue)
        context       = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind("tcp://*:*")

        self._addr  = self.socket.getsockopt(zmq.LAST_ENDPOINT)
        
        with open('%s.queue.url' % name, 'w') as fout:
            fout.write('ADDR %s\n' % self._addr)
    
    def _connect(self, send_rec):

        if send_rec == 'sender':
            self._senders += 1
        elif send_rec == 'receiver':
            self._receivers += 1
    
        
    def _disconnect(self, send_rec):

        if send_rec == 'sender':
            self._senders -= 1
        elif send_rec == 'receiver':
            self._receivers -= 1


    def _check_status(self):
        if self._queue and self._senders and self._receivers:
            return True
        else:
            return False


    def run(self):

        status = True
        while status:
            message = socket.recv()
            if 'connect' in message:
                self._connect(message.split()[0])
            elif 'add' in message:
                self._enqueue(message.split()[1])
            elif 'dequeue' in message:
                send_message = self._dequeue()
                self.socket.send(send_message.encode('utf-8'))
            elif 'disconnect' in message:
                self._disconnect(message.split[1])
            status = self._check_status()



        finished = False
        cont = True
        while cont:
            # only read from socket_in once we have a consumer requesting the data on
            # socket_out.
            req = self.socket_out.recv()
            if req == 'connect':
                self._connection += 1
                self.socket_out.send(msgpack.packb('connect'))
            elif req == 'image' and not finished:
                msg = msgpack.unpackb(self.socket_in.recv())
                if msg == 'Finished':
                    finished = True
                    self.socket_out.send(msgpack.packb('disconnect'))
                    self._connections -= 1
                else:
                    self.socket_out.send(msgpack.packb(msg))
            elif finished and self._connections != 0:
                self.socket_out.send(msgpack.packb('disconnect'))
                self._connections -= 1
            else:
                cont = False

            time.sleep(self.delay)


# ------------------------------------------------------------------------------
if __name__ == '__main__':

    q = Queue('test')
    q.run()