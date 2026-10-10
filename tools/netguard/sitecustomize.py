"""Network guard for the offline test (tools/offline_check.sh puts this folder on PYTHONPATH).

Every Python process then logs each internet connection or DNS lookup it attempts to $NETGUARD_LOG
and blocks it. Local Unix sockets are allowed and not counted.
"""
import os
import socket
import traceback

_LOG = os.environ.get("NETGUARD_LOG")


def _record(kind, target):
    if _LOG:
        stack = "".join(traceback.format_stack(limit=8)[:-2])
        with open(_LOG, "a") as fh:
            fh.write(f"[pid {os.getpid()}] {kind} {target}\n{stack}\n")
    raise OSError(f"network blocked by netguard: {kind} {target}")


_connect, _connect_ex = socket.socket.connect, socket.socket.connect_ex


def connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        _record("connect", address)
    return _connect(self, address)


def connect_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        _record("connect_ex", address)
    return _connect_ex(self, address)


def getaddrinfo(host, *a, **k):
    _record("getaddrinfo", host)


socket.socket.connect = connect
socket.socket.connect_ex = connect_ex
socket.getaddrinfo = getaddrinfo
