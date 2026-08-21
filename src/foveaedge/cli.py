"""Command-line entry points for FoveaEdge.

foveaedge-info  -- print environment/device report
foveaedge-bench -- (placeholder) run benchmark suite
"""

from __future__ import annotations

import argparse
import sys


def info_main() -> None:
    """Print the environment and device report."""
    from foveaedge.environment import detect_environment

    info = detect_environment()
    print(info.summary())

    # Also dump JSON for machine consumption
    import json
    print("")
    print("JSON:")
    print(json.dumps(info.to_dict(), indent=2))


def bench_main() -> None:
    """Run the benchmark suite (placeholder for Day 3+)."""
    print("FoveaEdge benchmark suite - not yet implemented.")
    print("This will be built during Day 3 (baseline) and Day 16 (full suite).")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="foveaedge",
        description="FoveaEdge - OpenVINO Event-Driven Foveated Edge Vision Engine",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="Print environment and device report")
    sub.add_parser("bench", help="Run benchmark suite")

    args = parser.parse_args()
    if args.command == "info":
        info_main()
    elif args.command == "bench":
        bench_main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
