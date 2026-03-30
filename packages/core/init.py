#!/usr/bin/env python3
"""
okg init — bootstrap the knowledge graph from source manifests.

Usage:
    python3 -m packages.core.init                      # core only
    python3 -m packages.core.init --graphs bio cancer   # core + bio + cancer
    python3 -m packages.core.init --graphs all          # everything

Core is always included. Dependencies are resolved automatically
from packages/INDEX.yaml.

Each package's sources/ directory contains YAML manifests declaring
external data dependencies. Init walks them in dependency order:
    1. Download public sources (skip gated with instructions)
    2. Translate raw data → okg format
    3. Seed the local graph
    4. Validate
"""

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

import yaml

PACKAGES_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGES_ROOT.parent


def load_package_index():
    """Load packages/INDEX.yaml and return {short_name: package_info}."""
    index_path = PACKAGES_ROOT / "INDEX.yaml"
    with open(index_path) as f:
        index = yaml.safe_load(f)

    packages = {}
    for pkg in index["packages"]:
        # okg.core → core, okg.bio → bio
        short = pkg["id"].split(".")[-1]
        packages[short] = {
            "id": pkg["id"],
            "path": PACKAGES_ROOT / pkg["path"],
            "dependencies": [d.split(".")[-1] for d in pkg.get("dependencies", [])],
            "description": pkg.get("description", ""),
        }
    return packages


def resolve_graphs(requested, all_packages):
    """Resolve --graphs argument into ordered list including dependencies."""
    if "all" in requested:
        requested = list(all_packages.keys())

    # Always include core
    if "core" not in requested:
        requested = ["core"] + requested

    # Resolve dependencies (topological)
    resolved = []
    seen = set()

    def visit(name):
        if name in seen:
            return
        if name not in all_packages:
            print(f"  [warn] Unknown package: {name}, skipping")
            return
        seen.add(name)
        for dep in all_packages[name]["dependencies"]:
            visit(dep)
        resolved.append(name)

    for name in requested:
        visit(name)

    return resolved


def load_manifests(pkg_path):
    """Load all source manifests from a package's sources/ directory."""
    manifests = []
    # Check both sources/ and data/sources/ (cancer has data/sources/)
    for sources_dir in [pkg_path / "sources", pkg_path / "data" / "sources"]:
        if sources_dir.is_dir():
            for f in sorted(sources_dir.glob("*.yaml")):
                with open(f) as fh:
                    manifest = yaml.safe_load(fh)
                    manifest["_file"] = f
                    manifests.append(manifest)
    # Also check language/sources/ for core
    lang_sources = pkg_path / "language" / "sources"
    if lang_sources.is_dir():
        for f in sorted(lang_sources.glob("*.yaml")):
            with open(f) as fh:
                manifest = yaml.safe_load(fh)
                manifest["_file"] = f
                manifests.append(manifest)
    return manifests


def download_source(manifest, pkg_path):
    """Download a single source per its manifest. Returns True if data is available."""
    name = manifest.get("name", manifest.get("id", "unknown"))
    access = manifest.get("access", "public")
    status = manifest.get("status", "active")

    if status == "planned":
        print(f"    [skip] {name} — planned, not yet available")
        return False

    if access == "gated":
        apply_url = manifest.get("apply_url", "see manifest for details")
        print(f"    [gated] {name} — requires manual download")
        print(f"            Apply at: {apply_url}")

        # Check if data already exists in cache
        cache_dir = manifest.get("cache_dir")
        if cache_dir:
            cache_path = pkg_path / "data" / cache_dir if "data" in str(manifest["_file"]) else pkg_path / cache_dir
            if cache_path.is_dir() and any(cache_path.iterdir()):
                print(f"            Cache exists: {cache_path}")
                return True
        return False

    # Public source — check for download script
    download_script = manifest.get("download_script")
    if download_script:
        script_path = pkg_path / download_script
        if not script_path.exists():
            # Try relative to data/
            script_path = pkg_path / "data" / download_script
        if not script_path.exists():
            print(f"    [warn] Download script not found: {download_script}")
            return False

        # Check if cache already populated
        cache_dir = manifest.get("cache_dir")
        if cache_dir:
            cache_path = pkg_path / "data" / cache_dir if "data" in str(manifest["_file"]) else pkg_path / cache_dir
            if cache_path.is_dir() and any(cache_path.iterdir()):
                print(f"    [cached] {name} — already downloaded")
                return True

        print(f"    [download] {name} via {download_script}")
        try:
            subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(REPO_ROOT),
                check=True,
                timeout=600,
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"    [error] Download failed: {e}")
            return False
        except subprocess.TimeoutExpired:
            print(f"    [error] Download timed out (10 min)")
            return False

    # No download script — check for direct download command
    download_cmd = manifest.get("download")
    if download_cmd:
        print(f"    [download] {name}")
        try:
            subprocess.run(
                download_cmd,
                shell=True,
                cwd=str(pkg_path),
                check=True,
                timeout=300,
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"    [error] Download failed: {e}")
            return False

    # No download mechanism — assume data is fetched at runtime (API sources)
    print(f"    [api] {name} — fetched at runtime")
    return True


def init_package(name, pkg_info, dry_run=False):
    """Initialize a single package: download sources, seed graph."""
    pkg_path = pkg_info["path"]
    print(f"\n{'='*60}")
    print(f"  {pkg_info['id']}: {pkg_info['description']}")
    print(f"{'='*60}")

    manifests = load_manifests(pkg_path)
    if not manifests:
        print(f"  No source manifests found")
        return

    print(f"  {len(manifests)} source(s) declared:")
    available = []
    for m in manifests:
        source_name = m.get("name", m.get("id", "?"))
        access = m.get("access", "public")
        status = m.get("status", "active")
        marker = {"public": "+", "gated": "!", "planned": "~"}.get(access, "?")
        if status == "planned":
            marker = "~"
        print(f"    [{marker}] {source_name}")

    if dry_run:
        return

    # Download phase
    print(f"\n  Downloading...")
    for m in manifests:
        download_source(m, pkg_path)

    # Seed phase — run package-specific seed script if it exists
    seed_script = pkg_path / "seed.py"
    if not seed_script.exists():
        seed_script = pkg_path / "python" / "seed_from_cancer.py"
    if seed_script.exists():
        print(f"\n  Seeding graph from {seed_script.name}...")
        if not dry_run:
            try:
                subprocess.run(
                    [sys.executable, str(seed_script)],
                    cwd=str(REPO_ROOT),
                    check=True,
                    timeout=300,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print(f"  [error] Seed failed: {e}")

    # Validate phase
    validate_script = PACKAGES_ROOT / "core" / "validate.py"
    if validate_script.exists() and name != "core":
        print(f"\n  Validating {name}...")
        # Validation runs per-package in future; for now just note it
        print(f"  [todo] Package-specific validation not yet wired")


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap the Open Knowledge Graph from source manifests.",
        prog="okg init",
    )
    parser.add_argument(
        "--graphs",
        nargs="*",
        default=[],
        help='Packages to initialize (e.g., bio cancer). "all" for everything. Core always included.',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_packages",
        help="List available packages and exit.",
    )
    args = parser.parse_args()

    all_packages = load_package_index()

    if args.list_packages:
        print("Available packages:")
        for name, info in sorted(all_packages.items()):
            deps = ", ".join(info["dependencies"]) if info["dependencies"] else "none"
            sources_dir = info["path"] / "sources"
            data_sources = info["path"] / "data" / "sources"
            n_sources = 0
            for d in [sources_dir, data_sources]:
                if d.is_dir():
                    n_sources += len(list(d.glob("*.yaml")))
            print(f"  {name:12s}  deps=[{deps}]  sources={n_sources}  {info['description']}")
        return

    requested = args.graphs if args.graphs else []
    if not requested:
        # Default: core only
        requested = ["core"]

    graph_order = resolve_graphs(requested, all_packages)

    mode = "DRY RUN" if args.dry_run else "INIT"
    print(f"okg {mode}: {' → '.join(graph_order)}")

    for name in graph_order:
        init_package(name, all_packages[name], dry_run=args.dry_run)

    print(f"\n{'='*60}")
    print(f"  Done. {len(graph_order)} package(s) initialized.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
