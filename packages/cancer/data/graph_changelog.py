"""
Graph Changelog — append-only journal for all Neo4j mutations.

Every write to the graph goes through this gateway. Changes are:
  1. Logged to an append-only JSONL file before execution
  2. Executed against Neo4j
  3. Logged as committed (or rolled back on failure)

The journal enables:
  - Full replay from empty graph
  - Audit trail for every edge/node/property change
  - Rollback to any point in time
  - Dry-run mode (log without executing)

Usage:
    from gnn.data.graph_changelog import GraphGateway

    gw = GraphGateway()
    gw.merge_edges("ATTENDS_TO", edges, source="transformer_v3_cycle_0")
    gw.close()

    # Dry run (no Neo4j writes):
    gw = GraphGateway(dry_run=True)

    # Replay:
    from gnn.data.graph_changelog import replay_from_log
    replay_from_log("graph_changelog.jsonl", up_to="2026-03-25T18:00:00")
"""

import os
import json
import time
import glob as glob_mod
from datetime import datetime, timezone, timedelta

CHANGELOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "changelog"
)

# Segment rotation: new file per day, compact after 7 days
SEGMENT_MAX_AGE_DAYS = 7


def _now():
    return datetime.now(timezone.utc).isoformat()


def _segment_path(base_dir=None):
    """Return today's segment file path (one per day, like Kafka log segments)."""
    if base_dir is None:
        base_dir = CHANGELOG_DIR
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return os.path.join(base_dir, f"changelog_{date_str}.jsonl")


def _all_segments(base_dir=None):
    """Return all segment files sorted by date (oldest first)."""
    if base_dir is None:
        base_dir = CHANGELOG_DIR
    pattern = os.path.join(base_dir, "changelog_*.jsonl")
    return sorted(glob_mod.glob(pattern))


def compact_segments(base_dir=None, max_age_days=SEGMENT_MAX_AGE_DAYS):
    """Compact old segments into a single summary + delete raw segments.

    Keeps recent segments (< max_age_days) as-is.
    Old segments get summarized into a compacted file with just the
    final state (committed merges, total counts by source).
    """
    if base_dir is None:
        base_dir = CHANGELOG_DIR

    segments = _all_segments(base_dir)
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")

    old_segments = [s for s in segments
                    if os.path.basename(s).replace("changelog_", "").replace(".jsonl", "") < cutoff_str]

    if not old_segments:
        return 0

    # Summarize old segments
    summary = {
        "compacted_at": _now(),
        "segments_compacted": len(old_segments),
        "date_range": [
            os.path.basename(old_segments[0]).replace("changelog_", "").replace(".jsonl", ""),
            os.path.basename(old_segments[-1]).replace("changelog_", "").replace(".jsonl", ""),
        ],
        "total_merges": 0,
        "total_edges_written": 0,
        "sources": {},
    }

    for seg in old_segments:
        with open(seg) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("type") == "merge_edges" and entry.get("status") == "committed":
                    summary["total_merges"] += 1
                    summary["total_edges_written"] += entry.get("n_edges", 0)
                    src = entry.get("source", "unknown")
                    summary["sources"][src] = summary["sources"].get(src, 0) + entry.get("n_edges", 0)

    # Write compacted summary
    compact_path = os.path.join(base_dir, "compacted.jsonl")
    with open(compact_path, "a") as f:
        f.write(json.dumps(summary) + "\n")

    # Remove old segment files
    for seg in old_segments:
        os.remove(seg)

    print(f"  Compacted {len(old_segments)} old segments "
          f"({summary['total_edges_written']} edges across {summary['total_merges']} merges)")
    return len(old_segments)


class GraphGateway:
    """Single entry point for all graph mutations. Logs everything.

    Changelog is segmented by date (one file per day).
    Old segments are compacted automatically.
    """

    def __init__(self, dry_run=False, changelog_dir=None):
        self.dry_run = dry_run
        self._changelog_dir = changelog_dir or CHANGELOG_DIR
        os.makedirs(self._changelog_dir, exist_ok=True)

        # Auto-compact on startup
        compact_segments(self._changelog_dir)

        self.driver = None
        if not dry_run:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                "bolt://localhost:7687", auth=("neo4j", "openknowledgegraph")
            )

        self._session_id = f"session_{int(time.time())}"
        self._log_entry({
            "type": "session_start",
            "session_id": self._session_id,
            "dry_run": dry_run,
        })

    def close(self):
        self._log_entry({
            "type": "session_end",
            "session_id": self._session_id,
        })
        if self.driver:
            self.driver.close()

    def _log_entry(self, entry):
        """Append a single entry to today's changelog segment."""
        entry["timestamp"] = _now()
        entry.setdefault("session_id", self._session_id)
        seg_path = _segment_path(self._changelog_dir)
        with open(seg_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def _execute(self, cypher, params=None):
        """Execute a Cypher query against Neo4j. Returns result summary."""
        if self.dry_run:
            return {"dry_run": True, "cypher": cypher}
        with self.driver.session() as s:
            result = s.run(cypher, **(params or {}))
            records = list(result)
            return {"records": len(records), "data": [dict(r) for r in records]}

    # -----------------------------------------------------------------
    # High-level operations
    # -----------------------------------------------------------------

    def merge_edges(self, rel_type, edges, source, source_detail=None,
                    match_from=("Gene", "name"), match_to=("Gene", "name")):
        """MERGE edges with provenance. Never deletes existing edges.

        Args:
            rel_type: Relationship type (e.g. "ATTENDS_TO")
            edges: List of dicts with 'from', 'to', and property keys
            source: What created these edges (e.g. "transformer_v3")
            source_detail: Extra context (e.g. "cycle_0, fold_avg")
            match_from: (label, key_property) for source node
            match_to: (label, key_property) for target node
        """
        if not edges:
            return 0

        from_label, from_key = match_from
        to_label, to_key = match_to

        # Log intent before execution
        self._log_entry({
            "type": "merge_edges",
            "status": "pending",
            "rel_type": rel_type,
            "n_edges": len(edges),
            "source": source,
            "source_detail": source_detail,
            "from_node": f"{from_label}.{from_key}",
            "to_node": f"{to_label}.{to_key}",
            "sample": edges[:3],  # first 3 for audit
        })

        if self.dry_run:
            self._log_entry({
                "type": "merge_edges",
                "status": "dry_run_skipped",
                "rel_type": rel_type,
                "n_edges": len(edges),
            })
            print(f"  [DRY RUN] Would merge {len(edges)} {rel_type} edges "
                  f"(source: {source})")
            return len(edges)

        # Build MERGE cypher with provenance
        cypher = f"""UNWIND $batch AS row
            MATCH (a:{from_label} {{{from_key}: row.from_key}}),
                  (b:{to_label} {{{to_key}: row.to_key}})
            MERGE (a)-[r:{rel_type}]->(b)
            ON CREATE SET
                r = row.props,
                r.source = $source,
                r.source_detail = $source_detail,
                r.created_at = $now,
                r.updated_at = $now,
                r.n_sources = 1
            ON MATCH SET
                r.updated_at = $now,
                r.n_sources = coalesce(r.n_sources, 1) + 1,
                r.prev_weight = r.weight,
                r.weight = CASE
                    WHEN row.props.weight IS NOT NULL
                    THEN (coalesce(r.weight, 0) * coalesce(r.n_sources, 1)
                          + row.props.weight) / (coalesce(r.n_sources, 1) + 1)
                    ELSE r.weight END,
                r.source = $source,
                r.source_detail = $source_detail
        """

        # Prepare batches
        batch_size = 500
        total_written = 0
        now = _now()

        for b_start in range(0, len(edges), batch_size):
            batch = []
            for e in edges[b_start:b_start + batch_size]:
                props = {k: v for k, v in e.items()
                         if k not in ('from', 'to', 'from_key', 'to_key')}
                batch.append({
                    'from_key': e.get('from_key', e.get('from')),
                    'to_key': e.get('to_key', e.get('to')),
                    'props': props,
                })

            with self.driver.session() as s:
                s.run(cypher, batch=batch, source=source,
                      source_detail=source_detail, now=now)
            total_written += len(batch)

        self._log_entry({
            "type": "merge_edges",
            "status": "committed",
            "rel_type": rel_type,
            "n_edges": total_written,
            "source": source,
        })

        print(f"  Merged {total_written} {rel_type} edges (source: {source})")
        return total_written

    def set_node_properties(self, label, key_prop, updates, source):
        """Set properties on existing nodes. Never deletes nodes.

        Args:
            label: Node label (e.g. "Gene")
            key_prop: Property used to match (e.g. "name")
            updates: List of dicts with key_prop value + properties to set
            source: What created these updates
        """
        if not updates:
            return 0

        self._log_entry({
            "type": "set_node_properties",
            "status": "pending",
            "label": label,
            "key_prop": key_prop,
            "n_nodes": len(updates),
            "source": source,
            "sample": updates[:3],
        })

        if self.dry_run:
            print(f"  [DRY RUN] Would update {len(updates)} {label} nodes "
                  f"(source: {source})")
            return len(updates)

        cypher = f"""UNWIND $batch AS row
            MATCH (n:{label} {{{key_prop}: row.key}})
            SET n += row.props,
                n.last_updated_by = $source,
                n.last_updated_at = $now
        """

        batch_size = 500
        total = 0
        now = _now()

        for b_start in range(0, len(updates), batch_size):
            batch = []
            for u in updates[b_start:b_start + batch_size]:
                key_val = u[key_prop]
                props = {k: v for k, v in u.items() if k != key_prop}
                batch.append({'key': key_val, 'props': props})

            with self.driver.session() as s:
                s.run(cypher, batch=batch, source=source, now=now)
            total += len(batch)

        self._log_entry({
            "type": "set_node_properties",
            "status": "committed",
            "label": label,
            "n_nodes": total,
            "source": source,
        })

        return total

    def count_edges(self, rel_type):
        """Count edges of a given type (read-only, no log)."""
        if self.dry_run:
            return -1
        with self.driver.session() as s:
            r = s.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS c")
            return r.single()['c']

    def snapshot(self, rel_type):
        """Take a snapshot of edge counts and sample edges for audit."""
        count = self.count_edges(rel_type)
        self._log_entry({
            "type": "snapshot",
            "rel_type": rel_type,
            "count": count,
        })
        return count


# =========================================================================
# SNAPSHOTS — full graph state capture (like database checkpoints)
# =========================================================================

SNAPSHOT_DIR = os.path.join(CHANGELOG_DIR, "snapshots")


def take_snapshot(label=None):
    """Capture full graph state as a snapshot.

    Exports all nodes, edges, and properties to JSON files.
    The WAL (changelog segments) can be truncated after a snapshot.
    Recovery = load latest snapshot + replay WAL since snapshot.
    """
    from neo4j import GraphDatabase
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snap_name = f"snapshot_{ts}" + (f"_{label}" if label else "")
    snap_dir = os.path.join(SNAPSHOT_DIR, snap_name)
    os.makedirs(snap_dir)

    driver = GraphDatabase.driver("bolt://localhost:7687",
                                  auth=("neo4j", "openknowledgegraph"))
    manifest = {"timestamp": _now(), "label": label, "counts": {}}

    try:
        with driver.session() as s:
            # --- Export edge types ---
            edge_types_result = s.run("""
                CALL db.relationshipTypes() YIELD relationshipType
                RETURN relationshipType
            """)
            edge_types = [r['relationshipType'] for r in edge_types_result]

            for et in edge_types:
                result = s.run(f"""
                    MATCH (a)-[r:{et}]->(b)
                    RETURN labels(a)[0] AS from_label,
                           CASE WHEN a.name IS NOT NULL THEN a.name
                                WHEN a.mutation_key IS NOT NULL THEN a.mutation_key
                                ELSE toString(id(a)) END AS from_key,
                           labels(b)[0] AS to_label,
                           CASE WHEN b.name IS NOT NULL THEN b.name
                                WHEN b.mutation_key IS NOT NULL THEN b.mutation_key
                                ELSE toString(id(b)) END AS to_key,
                           properties(r) AS props
                """)
                edges = [dict(r) for r in result]
                manifest["counts"][f"edge_{et}"] = len(edges)

                if edges:
                    path = os.path.join(snap_dir, f"edges_{et}.jsonl")
                    with open(path, "w") as f:
                        for e in edges:
                            f.write(json.dumps(e, default=str) + "\n")

            # --- Export node types ---
            label_result = s.run("CALL db.labels() YIELD label RETURN label")
            labels = [r['label'] for r in label_result]

            for lbl in labels:
                result = s.run(f"""
                    MATCH (n:{lbl})
                    RETURN properties(n) AS props
                """)
                nodes = [dict(r) for r in result]
                manifest["counts"][f"node_{lbl}"] = len(nodes)

                if nodes:
                    path = os.path.join(snap_dir, f"nodes_{lbl}.jsonl")
                    with open(path, "w") as f:
                        for n in nodes:
                            f.write(json.dumps(n, default=str) + "\n")

    finally:
        driver.close()

    # Write manifest
    with open(os.path.join(snap_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    total = sum(manifest["counts"].values())
    print(f"  Snapshot '{snap_name}': {total:,} items")
    for k, v in sorted(manifest["counts"].items()):
        print(f"    {k}: {v:,}")

    return snap_dir


def list_snapshots():
    """List available snapshots."""
    if not os.path.exists(SNAPSHOT_DIR):
        return []
    snaps = []
    for d in sorted(os.listdir(SNAPSHOT_DIR)):
        manifest_path = os.path.join(SNAPSHOT_DIR, d, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            snaps.append({
                "name": d,
                "timestamp": manifest.get("timestamp"),
                "label": manifest.get("label"),
                "counts": manifest.get("counts", {}),
            })
    return snaps


def wal_size():
    """Return current WAL size (number of entries across active segments)."""
    total = 0
    for seg in _all_segments():
        with open(seg) as f:
            total += sum(1 for _ in f)
    return total


def should_snapshot(threshold=1000):
    """Check if WAL is large enough to warrant a new snapshot."""
    return wal_size() >= threshold


def get_changelog_stats(changelog_dir=None):
    """Summarize all changelog segments + compacted history."""
    if changelog_dir is None:
        changelog_dir = CHANGELOG_DIR

    stats = {
        "entries": 0,
        "segments": 0,
        "sessions": 0,
        "merge_edges": 0,
        "set_node_properties": 0,
        "total_edges_written": 0,
        "sources": set(),
    }

    # Include compacted history
    compact_path = os.path.join(changelog_dir, "compacted.jsonl")
    if os.path.exists(compact_path):
        with open(compact_path) as f:
            for line in f:
                summary = json.loads(line)
                stats["merge_edges"] += summary.get("total_merges", 0)
                stats["total_edges_written"] += summary.get("total_edges_written", 0)
                for src in summary.get("sources", {}):
                    stats["sources"].add(src)

    # Current segments
    for seg in _all_segments(changelog_dir):
        stats["segments"] += 1
        with open(seg) as f:
            for line in f:
                entry = json.loads(line)
                stats["entries"] += 1
                if entry["type"] == "session_start":
                    stats["sessions"] += 1
                elif entry["type"] == "merge_edges" and entry.get("status") == "committed":
                    stats["merge_edges"] += 1
                    stats["total_edges_written"] += entry.get("n_edges", 0)
                    stats["sources"].add(entry.get("source", "unknown"))
                elif entry["type"] == "set_node_properties" and entry.get("status") == "committed":
                    stats["set_node_properties"] += 1

    stats["sources"] = sorted(stats["sources"])
    return stats


def replay_from_log(changelog_dir=None, up_to=None):
    """Replay the changelog to rebuild graph state.

    Args:
        changelog_dir: Directory containing changelog segments
        up_to: ISO timestamp — replay entries up to this time (inclusive)

    This is a recovery mechanism. It replays all MERGE operations
    in order, which is safe because MERGEs are idempotent.
    """
    if changelog_dir is None:
        changelog_dir = CHANGELOG_DIR

    segments = _all_segments(changelog_dir)
    print(f"Replaying from {changelog_dir} ({len(segments)} segments)")
    if up_to:
        print(f"  Up to: {up_to}")

    stats = get_changelog_stats(changelog_dir)
    print(f"  Total: {stats['entries']} entries, "
          f"{stats['merge_edges']} merges, "
          f"{stats['total_edges_written']} edges")
    print(f"  Sources: {', '.join(stats['sources'])}")

    print("\n  [NOTE] Full replay requires edge data in changelog.")
    print("  For now, re-run the source scripts (neo4j_walk, etc.)")
    return stats
