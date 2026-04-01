#!/usr/bin/env python3
"""
Schema validator for knowledge graph nodes and proofs.

Usage:
    python validate.py <file.yaml>           # validate one file
    python validate.py --all                  # validate everything
    python validate.py --proof PR01-sqrt2     # validate one proof
"""

import sys
import os
import yaml
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # knowledge-graph/
NODES = ROOT / "nodes"
# Directories with knowledge structure (skip large data files like authors/)
KNOWLEDGE_DIRS = ["math", "information-theory", "neurology", "endocrinology", "biology",
                  "definitions", "theorems", "emergent", "empirical",
                  "equivalency", "novel", "open-questions", "overlap", "references"]
MAX_FILE_SIZE = 500_000  # skip files larger than 500KB


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _knowledge_yamls():
    """Yield YAML paths from knowledge directories only (skip large data files)."""
    for dirname in KNOWLEDGE_DIRS:
        d = NODES / dirname
        if d.exists():
            for p in d.rglob("*.yaml"):
                if p.stat().st_size < MAX_FILE_SIZE:
                    yield p


def collect_all_concept_ids():
    """Gather every concept id across all knowledge YAML files."""
    ids = set()
    for p in _knowledge_yamls():
        try:
            d = load_yaml(p)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        for c in d.get("concepts", []):
            if isinstance(c, dict) and "id" in c:
                ids.add(c["id"])
    return ids


def collect_all_proof_ids():
    """Gather every proof id."""
    ids = set()
    for p in (NODES / "math" / "proofs").rglob("*.yaml"):
        if p.stat().st_size > MAX_FILE_SIZE:
            continue
        try:
            d = load_yaml(p)
        except Exception:
            continue
        if isinstance(d, dict) and d.get("type") in ("proof", "shared_lemma"):
            ids.add(d["id"])
    return ids


def validate_knowledge(data, path, all_concept_ids=None):
    """Validate a knowledge domain YAML file."""
    errors = []
    warnings = []

    if "id" not in data:
        errors.append("missing top-level 'id'")
    if "name" not in data:
        errors.append("missing top-level 'name'")
    if "type" not in data:
        warnings.append("missing 'type' (expected 'knowledge')")

    concepts = data.get("concepts", [])
    if not concepts:
        warnings.append("no concepts defined")

    seen_ids = set()
    for i, c in enumerate(concepts):
        if not isinstance(c, dict):
            errors.append(f"concept {i}: not a dict")
            continue
        cid = c.get("id")
        if not cid:
            errors.append(f"concept {i}: missing 'id'")
            continue
        if cid in seen_ids:
            errors.append(f"concept '{cid}': duplicate id in this file")
        seen_ids.add(cid)

        if "name" not in c:
            warnings.append(f"concept '{cid}': missing 'name'")

        # Check for global duplicates
        if all_concept_ids and cid in all_concept_ids:
            # Only warn if it's from a DIFFERENT file
            pass  # Would need file tracking — skip for now

    return errors, warnings


def validate_proof(data, path, all_concept_ids=None, all_proof_ids=None):
    """Validate a proof YAML file."""
    errors = []
    warnings = []

    if "id" not in data:
        errors.append("missing 'id'")
    if "name" not in data:
        errors.append("missing 'name'")
    if "assertions" not in data:
        errors.append("missing 'assertions' list")
        return errors, warnings

    assertions = data.get("assertions", [])
    step_ids = set()

    for i, a in enumerate(assertions):
        if not isinstance(a, dict):
            errors.append(f"assertion {i}: not a dict")
            continue

        aid = a.get("id")
        if not aid:
            errors.append(f"assertion {i}: missing 'id'")
            continue

        if aid in step_ids:
            errors.append(f"assertion '{aid}': duplicate step id")
        step_ids.add(aid)

        if "type" not in a:
            warnings.append(f"step '{aid}': missing assertion type")
        if "statement" not in a:
            warnings.append(f"step '{aid}': missing statement")

        # Check depends_on references
        for dep in a.get("depends_on", []):
            if dep not in step_ids:
                errors.append(f"step '{aid}': depends_on '{dep}' not defined in earlier steps")

        # Check concept references
        if all_concept_ids:
            for ref in a.get("references", []):
                if ref not in all_concept_ids:
                    warnings.append(f"step '{aid}': references '{ref}' not found in knowledge graph")

    # Check proof_links
    if all_proof_ids:
        for link in data.get("proof_links", []):
            target = link.get("target", "")
            if target and target not in all_proof_ids:
                warnings.append(f"proof_link target '{target}' not found")

    if not data.get("qed"):
        warnings.append("proof not marked qed: true")

    return errors, warnings


def validate_shared_lemma(data, path, all_proof_ids=None):
    """Validate a shared lemma YAML file."""
    errors = []
    warnings = []

    if "id" not in data:
        errors.append("missing 'id'")
    if "statement" not in data:
        errors.append("missing 'statement'")
    if "justification" not in data:
        warnings.append("missing 'justification'")

    for u in data.get("used_by", []):
        pid = u.get("proof", "")
        if all_proof_ids and pid and pid not in all_proof_ids:
            warnings.append(f"used_by proof '{pid}' not found")
        if "step" not in u:
            warnings.append(f"used_by entry for '{pid}' missing 'step' field")

    return errors, warnings


def validate_file(path, all_concept_ids=None, all_proof_ids=None):
    """Validate a single YAML file."""
    path = Path(path)
    try:
        data = load_yaml(path)
    except Exception as e:
        return [f"YAML parse error: {e}"], []

    if not isinstance(data, dict):
        return ["top-level is not a dict"], []

    t = data.get("type", "")
    if t == "proof":
        return validate_proof(data, path, all_concept_ids, all_proof_ids)
    elif t == "shared_lemma":
        return validate_shared_lemma(data, path, all_proof_ids)
    elif t == "meta":
        return [], []  # Framework files are fine
    else:
        return validate_knowledge(data, path, all_concept_ids)


def validate_all():
    """Validate every YAML file in the knowledge graph."""
    all_concept_ids = collect_all_concept_ids()
    all_proof_ids = collect_all_proof_ids()

    total_errors = 0
    total_warnings = 0
    total_files = 0

    for path in sorted(_knowledge_yamls()):
        errors, warnings = validate_file(path, all_concept_ids, all_proof_ids)
        total_files += 1

        if errors or warnings:
            rel = path.relative_to(ROOT)
            if errors:
                print(f"\n✗ {rel}")
                for e in errors:
                    print(f"  ERROR: {e}")
                total_errors += len(errors)
            if warnings:
                if not errors:
                    print(f"\n⚠ {rel}")
                for w in warnings:
                    print(f"  WARN:  {w}")
                total_warnings += len(warnings)

    print(f"\n{'='*60}")
    print(f"Validated {total_files} files")
    print(f"  {total_errors} errors, {total_warnings} warnings")
    if total_errors == 0:
        print("  ✓ All files pass validation")
    return total_errors


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "--all":
        errors = validate_all()
        sys.exit(1 if errors else 0)
    else:
        path = sys.argv[1]
        if not os.path.exists(path):
            # Try finding it
            matches = list(NODES.rglob(f"*{path}*"))
            if matches:
                path = matches[0]
            else:
                print(f"File not found: {path}")
                sys.exit(1)

        all_concept_ids = collect_all_concept_ids()
        all_proof_ids = collect_all_proof_ids()
        errors, warnings = validate_file(path, all_concept_ids, all_proof_ids)

        if errors:
            print(f"✗ {path}")
            for e in errors:
                print(f"  ERROR: {e}")
        if warnings:
            print(f"⚠ {path}")
            for w in warnings:
                print(f"  WARN:  {w}")
        if not errors and not warnings:
            print(f"✓ {path} — valid")

        sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
