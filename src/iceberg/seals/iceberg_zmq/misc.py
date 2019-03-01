import os
import sys
import time
import signal
import socket

import threading  as mt


import netifaces


# ------------------------------------------------------------------------------
#

def get_hostname():
    '''
    Look up the hostname
    '''
    global _hostname
    
    _hostname = None

    if not _hostname:

        if socket.gethostname().find('.') >= 0:
            _hostname = socket.gethostname()
        else:
            _hostname = socket.gethostbyaddr(socket.gethostname())[0]

    return _hostname


# ------------------------------------------------------------------------------
#
def get_hostip(req=None):
    '''
    Look up the ip number for a given requested interface name.
    If interface is not given, do some magic.
    '''

    global _hostip
    
    _hostip = None

    if _hostip:
        return _hostip

    AF_INET = netifaces.AF_INET

    # We create a ordered preference list, consisting of:
    #   - given arglist
    #   - white list (hardcoded preferred interfaces)
    #   - black_list (hardcoded unfavorable interfaces)
    #   - all others (whatever is not in the above)
    # Then this list is traversed, we check if the interface exists and has an
    # IP address.  The first match is used.

    if req: 
        if not isinstance(req, list):
            req = [req]
    else:
        req = []

    white_list = [
                  'ipogif0',  # Cray's
                  'br0',      # SuperMIC
                  'eth0',     # desktops etc.
                  'wlan0'     # laptops etc.
                 ]

    black_list = [
                  'lo',      # takes the 'inter' out of the 'net'
                  'sit0'     # ?
                 ]

    all  = netifaces.interfaces()
    rest = [iface for iface in all
                   if iface not in req and
                      iface not in white_list and
                      iface not in black_list]

    preflist = req + white_list + black_list + rest

    for iface in preflist:

        if iface not in all:
            continue

        info = netifaces.ifaddresses(iface)
        if AF_INET not in info:
            continue

        if not len(info[AF_INET]):
            continue

        if not info[AF_INET][0].get('addr'):
            continue

        ip = info[AF_INET][0].get('addr')

        if ip:
            _hostip = ip
            return ip

    raise RuntimeError('could not determine ip on %s' % preflist)


# ------------------------------------------------------------------------------
#
class Heartbeat(object):

    # --------------------------------------------------------------------------
    #
    def __init__(self, uid, timeout, frequency=1):
        '''
        This is a simple hearteat monitor: after construction, it needs to be
        called in ingtervals shorter than the given `timeout` value.  A thread
        will be created which checks if heartbeats arrive timeely - if not, the
        current process is killed via `os.kill()`.

        If no timeout is given, all class methods are noops.
        '''

        self._uid     = uid
        self._timeout = timeout
        self._freq    = frequency
        self._last    = time.time()
        self._cnt     = 0
        self._pid     = os.getpid()

        if self._timeout:
            self._watcher = mt.Thread(target=self._watch)
            self._watcher.daemon = True
            self._watcher.start()


    # --------------------------------------------------------------------------
    #
    def _watch(self):

        while True:

            time.sleep(self._freq)

            if time.time() - self._last > self._timeout:

                os.kill(self._pid, signal.SIGTERM)
                sys.stderr.write('Heartbeat timeout: %s\n' % self._uid)
                sys.stderr.flush()


    # --------------------------------------------------------------------------
    #
    def beat(self, timestamp=None):

        if not self._timeout:
            return

        if not timestamp:
            timestamp = time.time()

        self._last = timestamp
        self._cnt += 1


# ------------------------------------------------------------------------------

