#!/usr/bin/env python3
"""
Run benchmark comparisons against published models.

Usage:
    python3 -m gnn.scripts.benchmark
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gnn.benchmark.compare import print_comparison_table, compile_comparison_table
from gnn.config import GNN_RESULTS


def main():
    print_comparison_table()

    # Check if we have results to include
    cv_path = os.path.join(GNN_RESULTS, "cv_msk_impact_50k", "cv_summary.json")
    if os.path.exists(cv_path):
        print("\n" + "=" * 80)
        print("  OUR RESULTS")
        print("=" * 80)
        compile_comparison_table(cv_path)
    else:
        print(f"\n[No results yet — run training first]")
        print(f"  python3 -m gnn.scripts.train")


if __name__ == "__main__":
    main()
