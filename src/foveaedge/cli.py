"""FoveaEdge command-line interface."""

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the foveaedge CLI."""
    parser = argparse.ArgumentParser(
        prog="foveaedge",
        description="Selective inference research for OpenVINO edge devices",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__import__('foveaedge').__version__}"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="status",
        help="Command to run (status, benchmark, etc.)",
    )

    args = parser.parse_args(argv)

    if args.command == "status":
        try:
            from openvino.runtime import Core

            core = Core()
            devices = core.available_devices
            print(f"FoveaEdge v{__import__('foveaedge').__version__}")
            print(f"OpenVINO devices: {devices}")
        except ImportError:
            print("FoveaEdge v0.1.0")
            print("OpenVINO not available")
        return 0

    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
