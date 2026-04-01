#!/usr/bin/env python3
"""Extract quantitative claims from all papers and check for contradictions.

Builds a claims graph: each claim is a node with paper source, value, and type.
Edges connect claims about the same quantity. Contradictions are flagged.

Reads: publications/arxiv/*.tex, publications/biorxiv/*.tex
Writes: knowledge-graph/nodes/empirical/paper_claims.yaml
        knowledge-graph/scripts/contradictions_report.txt
"""

import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
PUB_DIRS = [ROOT / "publications" / "arxiv", ROOT / "publications" / "biorxiv"]


def extract_claims(tex_path):
    """Extract quantitative claims from a .tex file."""
    with open(tex_path) as f:
        text = f.read()

    filename = tex_path.name
    claims = []

    # Channel count claims
    for m in re.finditer(r'(six|eight|6|8)\s+(?:coupling\s+)?channels?', text, re.IGNORECASE):
        val = m.group(1).lower()
        val_num = {"six": 6, "eight": 8, "6": 6, "8": 8}.get(val, val)
        line_num = text[:m.start()].count('\n') + 1
        claims.append({
            "quantity": "n_channels",
            "value": val_num,
            "context": text[max(0, m.start()-40):m.end()+40].replace('\n', ' ').strip(),
            "file": filename,
            "line": line_num,
        })

    # Gene count claims
    for m in re.finditer(r'(\d+)\s*(?:[-–])?genes?(?:\s+(?:mapped|across|in))', text):
        val = int(m.group(1))
        if val > 10:
            line_num = text[:m.start()].count('\n') + 1
            claims.append({
                "quantity": "n_genes",
                "value": val,
                "context": text[max(0, m.start()-40):m.end()+40].replace('\n', ' ').strip(),
                "file": filename,
                "line": line_num,
            })

    # n= sample sizes
    for m in re.finditer(r'[nN]\s*=\s*([\d,{} ]+?)(?:\s|\\|;|\)|\]|$)', text):
        raw = m.group(1).replace(',', '').replace('{', '').replace('}', '').replace(' ', '')
        try:
            val = int(raw)
            if val > 100:
                line_num = text[:m.start()].count('\n') + 1
                context = text[max(0, m.start()-60):m.end()+60].replace('\n', ' ').strip()
                # Try to identify which dataset
                dataset = "unknown"
                window = text[max(0, m.start()-200):m.end()].lower()
                if "2017" in window or "msk-impact 2017" in window:
                    dataset = "MSK-IMPACT-2017"
                elif "met" in window and "tropi" in window:
                    dataset = "MSK-MET"
                elif "50k" in window or "50,000" in window or "50K" in window:
                    dataset = "MSK-IMPACT-50K"
                elif "metabric" in window:
                    dataset = "METABRIC"
                elif "tcga" in window:
                    dataset = "TCGA"

                claims.append({
                    "quantity": f"sample_size_{dataset}",
                    "value": val,
                    "context": context,
                    "file": filename,
                    "line": line_num,
                })
        except ValueError:
            pass

    # HR values
    for m in re.finditer(r'HR\s*[=~≈]\s*\$?\s*([\d.]+)\s*\$?', text):
        val = float(m.group(1))
        line_num = text[:m.start()].count('\n') + 1
        context = text[max(0, m.start()-80):m.end()+80].replace('\n', ' ').strip()
        # Identify what the HR is for
        window = text[max(0, m.start()-150):m.end()].lower()
        hr_type = "unknown"
        if "channel" in window and "count" in window:
            hr_type = "channel_count_hr"
        elif "mutation" in window and ("count" in window or "burden" in window):
            hr_type = "mutation_count_hr"
        elif "cross" in window and "channel" in window:
            hr_type = "cross_channel_hr"

        claims.append({
            "quantity": hr_type,
            "value": val,
            "context": context,
            "file": filename,
            "line": line_num,
        })

    # p-value claims
    for m in re.finditer(r'p\s*[<≈=~]\s*\$?\s*10\^?\{?\s*-\s*(\d+)\s*\}?\s*\$?', text):
        exp = int(m.group(1))
        line_num = text[:m.start()].count('\n') + 1
        context = text[max(0, m.start()-80):m.end()+80].replace('\n', ' ').strip()
        claims.append({
            "quantity": "p_value_exponent",
            "value": -exp,
            "context": context,
            "file": filename,
            "line": line_num,
        })

    # Tier claims
    for m in re.finditer(r'(four|4|three|3)\s+tiers?', text, re.IGNORECASE):
        val = m.group(1).lower()
        val_num = {"four": 4, "three": 3, "4": 4, "3": 3}.get(val, val)
        line_num = text[:m.start()].count('\n') + 1
        claims.append({
            "quantity": "n_tiers",
            "value": val_num,
            "context": text[max(0, m.start()-40):m.end()+40].replace('\n', ' ').strip(),
            "file": filename,
            "line": line_num,
        })

    return claims


def find_contradictions(all_claims):
    """Group claims by quantity and flag contradictions."""
    by_quantity = defaultdict(list)
    for claim in all_claims:
        by_quantity[claim["quantity"]].append(claim)

    contradictions = []
    for quantity, claims in sorted(by_quantity.items()):
        values = set(c["value"] for c in claims)
        if len(values) > 1 and quantity not in ("p_value_exponent", "unknown"):
            # Filter out generic sample sizes
            if quantity.startswith("sample_size_unknown"):
                continue
            contradictions.append({
                "quantity": quantity,
                "values": sorted(values),
                "claims": claims,
            })

    return contradictions


def main():
    print("Extracting claims from all papers...")
    all_claims = []

    tex_files = []
    for pub_dir in PUB_DIRS:
        tex_files.extend(sorted(pub_dir.glob("McCarthy2026_*.tex")))

    for tex_path in tex_files:
        claims = extract_claims(tex_path)
        all_claims.extend(claims)
        print(f"  {tex_path.name}: {len(claims)} claims")

    print(f"\nTotal: {len(all_claims)} claims from {len(tex_files)} papers")

    # Group and find contradictions
    contradictions = find_contradictions(all_claims)

    print(f"\n{'='*60}")
    print(f"CONTRADICTIONS FOUND: {len(contradictions)}")
    print(f"{'='*60}\n")

    report_lines = []
    for c in contradictions:
        header = f"CONTRADICTION: {c['quantity']} has values {c['values']}"
        print(header)
        report_lines.append(header)
        for claim in c["claims"]:
            line = f"  {claim['file']}:{claim['line']} -> {claim['value']}"
            print(line)
            report_lines.append(line)
            ctx = f"    Context: ...{claim['context'][:120]}..."
            print(ctx)
            report_lines.append(ctx)
        print()
        report_lines.append("")

    # Also print consistent quantities
    by_quantity = defaultdict(list)
    for claim in all_claims:
        by_quantity[claim["quantity"]].append(claim)

    print(f"\n{'='*60}")
    print("CONSISTENT QUANTITIES:")
    print(f"{'='*60}\n")
    for quantity, claims in sorted(by_quantity.items()):
        values = set(c["value"] for c in claims)
        if len(values) == 1:
            files = sorted(set(c["file"] for c in claims))
            print(f"  {quantity} = {list(values)[0]} (consistent across {', '.join(files)})")

    # Write report
    report_path = ROOT / "knowledge-graph" / "scripts" / "contradictions_report.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
