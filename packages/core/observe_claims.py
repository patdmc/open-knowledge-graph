#!/usr/bin/env python3
"""
FUNCTION: observe_claims(assistant_response) → gap_report
TRIGGER: Stop hook (fires after every assistant response)
PURPOSE: Scan assistant output for biological claims.
         Check each against the knowledge graph.
         Novel claims → saved as pending gaps.
         Report injected into next prompt for user awareness.

PIPELINE:
  1. READ(last_assistant_message)
  2. EXTRACT(biological_claims) → structured claims
  3. VALIDATE(claims) → confirmed / novel / no_data
  4. SAVE(novel_claims) → pending_gaps.json
  5. EXIT(0) — never block
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure our hooks dir is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate import Validator, extract_claims
from precipitate import Precipitator, GAPS_DIR


def main():
    # Read hook input from stdin
    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, Exception):
        sys.exit(0)  # never block

    # Get the assistant's last message
    messages = hook_input.get("messages", [])
    if not messages:
        sys.exit(0)

    last = messages[-1]
    if last.get("role") != "assistant":
        sys.exit(0)

    content = last.get("content", "")
    if isinstance(content, list):
        # Handle structured content blocks
        content = " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )

    if not content or len(content) < 50:
        sys.exit(0)  # too short to contain claims

    # Extract and validate claims
    claims = extract_claims(content)
    if not claims:
        sys.exit(0)

    v = Validator()
    p = Precipitator()

    novel_claims = []
    confirmed_claims = []
    no_data_claims = []

    for claim in claims:
        result = v._check_single_claim(claim)

        if result["status"] == "novel":
            gap = p.save_gap(claim, result["reason"])
            novel_claims.append(gap)
        elif result["status"] == "confirmed":
            confirmed_claims.append(claim)
        elif result["status"] == "no_data":
            no_data_claims.append(claim)

    # Shadow mode: log everything to file, don't inject into prompt.
    # When we trust the signal, flip SHADOW_MODE to False and the
    # report goes into additionalContext instead.
    SHADOW_MODE = True

    total = len(confirmed_claims) + len(novel_claims) + len(no_data_claims)
    if total == 0:
        sys.exit(0)

    report_lines = [
        f"[KNOWLEDGE_GRAPH] {total} claims: "
        f"{len(confirmed_claims)} confirmed, "
        f"{len(novel_claims)} novel (need source), "
        f"{len(no_data_claims)} outside graph"
    ]
    for c in confirmed_claims:
        report_lines.append(
            f"  OK: {c['subject']} {c['verb']} {c['object']}"
        )
    for g in novel_claims:
        report_lines.append(
            f"  GAP [{g['id']}]: {g['subject']} {g['verb']} {g['object']}"
            f" — needs source"
        )
    for c in no_data_claims:
        report_lines.append(
            f"  ??: {c['subject']} {c['verb']} {c['object']}"
            f" — not in graph"
        )

    report = "\n".join(report_lines)

    # Always log to file for evaluation
    log_path = GAPS_DIR / "shadow_log.jsonl"
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_claims": total,
        "confirmed": len(confirmed_claims),
        "novel": len(novel_claims),
        "no_data": len(no_data_claims),
        "details": report,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    if not SHADOW_MODE and novel_claims:
        output = {
            "hookSpecificOutput": {
                "additionalContext": report,
            }
        }
        print(json.dumps(output))

    sys.exit(0)


if __name__ == "__main__":
    main()
