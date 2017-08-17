import zprocess
import time

class ImageServer(zprocess.ZMQServer):
    def __init__(self, port):
           zprocess.ZMQServer.__init__(self, port, type='raw')

    def handler(self, request_data):
        if request_data == 'hello':
            return 'hello'
        elif request_data.endswith('.h5'):
            self.send('ok')
            self.recv()
            return 'done'
        elif request_data == 'done':
            self.send('ok')
            self.recv()
            return 'done'
        elif request_data == 'abort':
            return 'done'
        else:
            raise ValueError('invalid request: %s'%request_data)


if __name__ == '__main__':
    port = 1027
    print('starting dummy camera server on port %d...'%port)
    server = ImageServer(port)
    try:
        while True:
            time.sleep(1)
    except:
        server.shutdown()


