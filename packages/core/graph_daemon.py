"""
Graph daemon — keeps the mutation graph hot in memory.

Starts a local Unix socket server. Any script can connect, send a query,
and get results back without rebuilding the graph.

Usage:
    # Start daemon (blocks, or background it):
    python3 -m gnn.data.graph_daemon start

    # From any script:
    from gnn.data.graph_client import GraphClient
    client = GraphClient()  # auto-starts daemon if not running
    features = client.walk_patient('P-0000001')
    df = client.walk_all_patients()

Protocol: JSON over Unix socket. One JSON object per request, one per response.
"""

import os
import sys
import json
import socket
import signal
import time
import threading
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.config import GNN_CACHE, MSK_DATASETS, ANALYSIS_CACHE

SOCKET_PATH = os.path.join(GNN_CACHE, "graph_daemon.sock")
PID_PATH = os.path.join(GNN_CACHE, "graph_daemon.pid")
GRAPH_CACHE = os.path.join(GNN_CACHE, "mutation_graph.pkl")

# Datasources to watch for changes
WATCH_FILES = [
    MSK_DATASETS["msk_impact_50k"]["mutations"],
    MSK_DATASETS["msk_impact_50k"]["clinical"],
    MSK_DATASETS["msk_impact_50k"]["sample_clinical"],
    os.path.join(ANALYSIS_CACHE, "survival_atlas_full.csv"),
]
CHECK_INTERVAL = 30  # seconds between mtime checks


def _recv_json(conn):
    """Read a length-prefixed JSON message."""
    # Read 8-byte length header
    header = b''
    while len(header) < 8:
        chunk = conn.recv(8 - len(header))
        if not chunk:
            return None
        header += chunk
    msg_len = int(header)

    # Read message body
    data = b''
    while len(data) < msg_len:
        chunk = conn.recv(min(msg_len - len(data), 65536))
        if not chunk:
            return None
        data += chunk
    return json.loads(data.decode('utf-8'))


def _send_json(conn, obj):
    """Send a length-prefixed JSON message."""
    body = json.dumps(obj).encode('utf-8')
    header = f"{len(body):08d}".encode('utf-8')
    conn.sendall(header + body)


class GraphDaemon:
    """Persistent graph server over Unix socket."""

    def __init__(self):
        self.mg = None
        self.running = False
        self._source_mtimes = {}  # file -> last known mtime
        self._data_stale = False  # True when sources changed since last build

    def _snapshot_mtimes(self):
        """Record current mtimes of all watched datasources."""
        self._source_mtimes = {}
        for f in WATCH_FILES:
            if os.path.exists(f):
                self._source_mtimes[f] = os.path.getmtime(f)

    def _check_sources_changed(self):
        """Check if any datasource has been modified since last snapshot."""
        for f in WATCH_FILES:
            if os.path.exists(f):
                current = os.path.getmtime(f)
                if f not in self._source_mtimes or current != self._source_mtimes[f]:
                    return True
        return False

    def _watch_sources(self):
        """Background thread: periodically check datasource mtimes."""
        while self.running:
            time.sleep(CHECK_INTERVAL)
            if self._check_sources_changed():
                if not self._data_stale:
                    self._data_stale = True
                    changed = []
                    for f in WATCH_FILES:
                        if os.path.exists(f):
                            current = os.path.getmtime(f)
                            if f not in self._source_mtimes or current != self._source_mtimes[f]:
                                changed.append(os.path.basename(f))
                    print(f"[watch] Datasources changed: {', '.join(changed)}. "
                          f"Graph is stale — clients will be notified.",
                          flush=True)

    def _load_graph(self):
        """Load or build the graph."""
        from gnn.data.mutation_graph import MutationGraph

        if os.path.exists(GRAPH_CACHE):
            print("Loading cached graph...", flush=True)
            self.mg = MutationGraph()
            self.mg.load(GRAPH_CACHE)
        else:
            print("Building graph (first run)...", flush=True)
            self.mg = MutationGraph()
            self.mg.build()
            self.mg.save(GRAPH_CACHE)

        self.mg.stats()
        self._snapshot_mtimes()
        self._data_stale = False

    def _handle_request(self, request):
        """Process a single request and return response."""
        cmd = request.get('cmd')

        if cmd == 'ping':
            return {'status': 'ok', 'msg': 'pong',
                    'nodes': self.mg.G.number_of_nodes(),
                    'edges': self.mg.G.number_of_edges(),
                    'stale': self._data_stale}

        elif cmd == 'check_stale':
            changed = []
            if self._data_stale:
                for f in WATCH_FILES:
                    if os.path.exists(f):
                        current = os.path.getmtime(f)
                        if f not in self._source_mtimes or current != self._source_mtimes[f]:
                            changed.append(os.path.basename(f))
            return {'status': 'ok', 'stale': self._data_stale,
                    'changed_files': changed}

        elif cmd == 'walk_patient':
            pid = request['patient_id']
            result = self.mg.walk_patient(pid)
            if result is None:
                return {'status': 'error', 'msg': f'patient {pid} not found'}
            # Convert numpy types to Python types for JSON
            clean = {}
            for k, v in result.items():
                if isinstance(v, (np.integer,)):
                    clean[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    clean[k] = float(v)
                else:
                    clean[k] = v
            return {'status': 'ok', 'data': clean}

        elif cmd == 'walk_batch':
            pids = request['patient_ids']
            results = []
            for pid in pids:
                rec = self.mg.walk_patient(pid)
                if rec:
                    clean = {}
                    for k, v in rec.items():
                        if isinstance(v, (np.integer,)):
                            clean[k] = int(v)
                        elif isinstance(v, (np.floating,)):
                            clean[k] = float(v)
                        else:
                            clean[k] = v
                    results.append(clean)
            return {'status': 'ok', 'data': results, 'n': len(results)}

        elif cmd == 'walk_all':
            df = self.mg.walk_all_patients()
            # Send as records
            records = df.to_dict(orient='records')
            # Clean numpy types
            for rec in records:
                for k, v in rec.items():
                    if isinstance(v, (np.integer,)):
                        rec[k] = int(v)
                    elif isinstance(v, (np.floating,)):
                        rec[k] = float(v)
            return {'status': 'ok', 'data': records, 'n': len(records)}

        elif cmd == 'stats':
            from collections import defaultdict
            ntypes = defaultdict(int)
            for _, d in self.mg.G.nodes(data=True):
                ntypes[d.get('ntype', 'unknown')] += 1
            etypes = defaultdict(int)
            for _, _, d in self.mg.G.edges(data=True):
                etypes[d.get('etype', 'unknown')] += 1
            return {'status': 'ok',
                    'nodes': dict(ntypes), 'edges': dict(etypes),
                    'total_nodes': self.mg.G.number_of_nodes(),
                    'total_edges': self.mg.G.number_of_edges()}

        elif cmd == 'rebuild':
            print("Rebuilding graph...", flush=True)
            self.mg = None
            if os.path.exists(GRAPH_CACHE):
                os.remove(GRAPH_CACHE)
            self._load_graph()
            return {'status': 'ok', 'msg': 'rebuilt'}

        elif cmd == 'shutdown':
            self.running = False
            return {'status': 'ok', 'msg': 'shutting down'}

        else:
            return {'status': 'error', 'msg': f'unknown command: {cmd}'}

    def _handle_client(self, conn):
        """Handle one client connection (may send multiple requests)."""
        try:
            while True:
                request = _recv_json(conn)
                if request is None:
                    break
                response = self._handle_request(request)
                _send_json(conn, response)
        except Exception as e:
            try:
                _send_json(conn, {'status': 'error', 'msg': str(e)})
            except Exception:
                pass
        finally:
            conn.close()

    def start(self):
        """Start the daemon."""
        self._load_graph()

        # Clean up old socket
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(SOCKET_PATH)
        server.listen(5)
        server.settimeout(1.0)  # So we can check self.running

        # Write PID file
        with open(PID_PATH, 'w') as f:
            f.write(str(os.getpid()))

        self.running = True

        # Start datasource watcher
        watcher = threading.Thread(target=self._watch_sources, daemon=True)
        watcher.start()

        print(f"\nGraph daemon listening on {SOCKET_PATH}", flush=True)
        print(f"PID: {os.getpid()}", flush=True)
        print(f"Watching {len(WATCH_FILES)} datasources (every {CHECK_INTERVAL}s)",
              flush=True)

        def shutdown_handler(signum, frame):
            print("\nShutting down...", flush=True)
            self.running = False

        signal.signal(signal.SIGTERM, shutdown_handler)
        signal.signal(signal.SIGINT, shutdown_handler)

        while self.running:
            try:
                conn, _ = server.accept()
                t = threading.Thread(target=self._handle_client, args=(conn,),
                                     daemon=True)
                t.start()
            except socket.timeout:
                continue

        server.close()
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)
        if os.path.exists(PID_PATH):
            os.remove(PID_PATH)
        print("Daemon stopped.", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'stop', 'status'])
    args = parser.parse_args()

    if args.action == 'start':
        daemon = GraphDaemon()
        daemon.start()

    elif args.action == 'stop':
        if os.path.exists(PID_PATH):
            with open(PID_PATH) as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"Sent SIGTERM to {pid}")
            except ProcessLookupError:
                print(f"Process {pid} not found, cleaning up")
                if os.path.exists(SOCKET_PATH):
                    os.remove(SOCKET_PATH)
                if os.path.exists(PID_PATH):
                    os.remove(PID_PATH)
        else:
            print("Daemon not running")

    elif args.action == 'status':
        if os.path.exists(PID_PATH):
            with open(PID_PATH) as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, 0)  # Check if alive
                print(f"Daemon running (PID {pid})")
            except ProcessLookupError:
                print("Daemon not running (stale PID file)")
        else:
            print("Daemon not running")


if __name__ == "__main__":
    main()
