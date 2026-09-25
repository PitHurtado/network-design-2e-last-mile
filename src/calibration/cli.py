"""`calibrate`: choose the SAA sample size N (planned; see `src.calibration.sample_size`).

calibrate sample-size --scenarios v1 --ns 5 10 20 30 --replications 10
"""

import argparse
import sys

from src.calibration.sample_size import SampleSizeConfig, SampleSizeStudy
from src.optimization.instance import InstanceBuilder


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="calibrate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    size = sub.add_parser("sample-size", help="stability and MMW gap as a function of N")
    size.add_argument("--scenarios", required=True)
    size.add_argument("--ns", nargs="+", type=int, default=[5, 10, 20, 30, 50])
    size.add_argument("--replications", type=int, default=10)
    args = parser.parse_args(argv)
    config = SampleSizeConfig(args.scenarios, tuple(args.ns), args.replications)
    try:
        SampleSizeStudy(config, InstanceBuilder.for_version(args.scenarios)).run()
    except NotImplementedError as error:
        raise SystemExit(f"calibrate sample-size: {error}") from error


if __name__ == "__main__":
    main()
