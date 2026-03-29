"""
Graph client — connects to the graph daemon.

Auto-starts the daemon if it's not running.

Usage:
    from gnn.data.graph_client import GraphClient

    client = GraphClient()
    features = client.walk_patient('P-0000001')
    df = client.walk_all()
"""

import os
import sys
import json
import socket
import subprocess
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import GNN_CACHE

SOCKET_PATH = os.path.join(GNN_CACHE, "graph_daemon.sock")
PID_PATH = os.path.join(GNN_CACHE, "graph_daemon.pid")


def _send_json(sock, obj):
    body = json.dumps(obj).encode('utf-8')
    header = f"{len(body):08d}".encode('utf-8')
    sock.sendall(header + body)


def _recv_json(sock):
    header = b''
    while len(header) < 8:
        chunk = sock.recv(8 - len(header))
        if not chunk:
            return None
        header += chunk
    msg_len = int(header)

    data = b''
    while len(data) < msg_len:
        chunk = sock.recv(min(msg_len - len(data), 65536))
        if not chunk:
            return None
        data += chunk
    return json.loads(data.decode('utf-8'))


class GraphClient:
    """Client for the graph daemon. Auto-starts daemon if needed."""

    def __init__(self, auto_start=True, start_timeout=120, check_stale=True):
        self._auto_start = auto_start
        self._start_timeout = start_timeout
        self._check_stale = check_stale
        self._stale_warned = False

    def _is_daemon_running(self):
        """Check if daemon is alive."""
        if not os.path.exists(PID_PATH):
            return False
        with open(PID_PATH) as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False

    def _start_daemon(self):
        """Start daemon in background."""
        print("Starting graph daemon...", flush=True)
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        subprocess.Popen(
            [sys.executable, '-m', 'gnn.data.graph_daemon', 'start'],
            cwd=root,
            stdout=open(os.path.join(GNN_CACHE, 'daemon.log'), 'w'),
            stderr=subprocess.STDOUT,
        )

        # Wait for socket to appear
        t0 = time.time()
        while time.time() - t0 < self._start_timeout:
            if os.path.exists(SOCKET_PATH):
                # Try to connect
                try:
                    resp = self._request({'cmd': 'ping'})
                    if resp and resp.get('status') == 'ok':
                        print(f"Daemon ready ({resp['nodes']} nodes, "
                              f"{resp['edges']} edges)", flush=True)
                        return
                except Exception:
                    pass
            time.sleep(1)

        raise RuntimeError(f"Daemon failed to start within {self._start_timeout}s. "
                           f"Check {os.path.join(GNN_CACHE, 'daemon.log')}")

    def _connect(self):
        """Get a connected socket, starting daemon if needed."""
        if self._auto_start and not self._is_daemon_running():
            self._start_daemon()

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(SOCKET_PATH)
        return sock

    def _request(self, req):
        """Send request, get response."""
        sock = self._connect()
        try:
            _send_json(sock, req)
            resp = _recv_json(sock)
        finally:
            sock.close()

        # Check staleness on first data request
        if (self._check_stale and not self._stale_warned
                and resp and resp.get('stale')):
            self._stale_warned = True
            print("\n*** GRAPH DATA IS STALE ***", flush=True)
            print("Datasources have changed since the graph was built.", flush=True)
            stale_info = self._raw_request({'cmd': 'check_stale'})
            if stale_info and stale_info.get('changed_files'):
                print(f"Changed: {', '.join(stale_info['changed_files'])}", flush=True)
            print("Run client.reload() to rebuild, or continue with current data.",
                  flush=True)

        return resp

    def _raw_request(self, req):
        """Send request without staleness check (avoids recursion)."""
        sock = self._connect()
        try:
            _send_json(sock, req)
            return _recv_json(sock)
        finally:
            sock.close()

    def ping(self):
        return self._raw_request({'cmd': 'ping'})

    def check_stale(self):
        """Check if datasources have changed. Returns dict with stale flag and changed files."""
        return self._raw_request({'cmd': 'check_stale'})

    def reload(self):
        """Rebuild graph from updated datasources."""
        print("Rebuilding graph from updated sources...", flush=True)
        resp = self._raw_request({'cmd': 'rebuild'})
        self._stale_warned = False
        return resp

    def walk_patient(self, patient_id):
        """Walk one patient. Returns dict of features."""
        resp = self._request({'cmd': 'walk_patient', 'patient_id': patient_id})
        if resp['status'] == 'ok':
            return resp['data']
        raise ValueError(resp['msg'])

    def walk_batch(self, patient_ids):
        """Walk a batch of patients. Returns list of dicts."""
        resp = self._request({'cmd': 'walk_batch', 'patient_ids': patient_ids})
        if resp['status'] == 'ok':
            return resp['data']
        raise ValueError(resp['msg'])

    def walk_all(self):
        """Walk all patients. Returns DataFrame."""
        resp = self._request({'cmd': 'walk_all'})
        if resp['status'] == 'ok':
            return pd.DataFrame(resp['data'])
        raise ValueError(resp['msg'])

    def stats(self):
        return self._request({'cmd': 'stats'})

    def rebuild(self):
        """Force rebuild the graph from data."""
        return self._request({'cmd': 'rebuild'})

    def shutdown(self):
        """Stop the daemon."""
        return self._request({'cmd': 'shutdown'})
