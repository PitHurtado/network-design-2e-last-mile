"""Capture (or re-capture) the golden expectations.

    poetry run python -m tests.golden.capture              # every golden
    poetry run python -m tests.golden.capture g3_generate  # only some

Re-capturing overwrites `expected/<name>.json`: do it only for a deliberate change, and
say so in the commit.
"""

import json
import sys
import time

from tests.golden.producers import PRODUCERS
from tests.golden.support import EXPECTED, Workspace


def main() -> None:
    names = sys.argv[1:] or list(PRODUCERS)
    unknown = set(names) - set(PRODUCERS)
    if unknown:
        raise SystemExit(f"Unknown goldens: {sorted(unknown)}")
    EXPECTED.mkdir(parents=True, exist_ok=True)
    ws = Workspace()
    try:
        for name in names:
            start = time.time()
            result = PRODUCERS[name](ws)
            (EXPECTED / f"{name}.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
            print(f"{name}: captured in {time.time() - start:.1f}s")
    finally:
        ws.close()


if __name__ == "__main__":
    main()
