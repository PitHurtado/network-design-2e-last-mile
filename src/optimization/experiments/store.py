"""`ResultStore`: the one place that knows where each experiment leaf is persisted.

Layout: `<root>/<version>/<flexibility>/<regime>[/<case>]/<filename>`. Sibling cases of
the same policy and regime share a parent, which the evaluation relies on to reuse an
identical fixed-Y evaluation.
"""

from collections.abc import Iterator
from pathlib import Path

from src.tools.io import read_json, write_json


class ResultStore:
    def __init__(self, root: Path, filename: str):
        self.root = Path(root)
        self.filename = filename

    def leaf_path(self, version: str, flexibility: str, regime: str, case: str | None = None) -> Path:
        directory = self.root / version / flexibility / regime
        return (directory / case if case is not None else directory) / self.filename

    def write(self, path: Path, payload: dict, overwrite: bool = False) -> Path:
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; use --overwrite or a new results version.")
        return write_json(path, payload)

    @staticmethod
    def read(path: Path) -> dict:
        return read_json(path)

    def siblings(self, path: Path) -> Iterator[Path]:
        """Leaves of the other cases under the same policy and regime."""
        for candidate in path.parent.parent.glob(f"*/{self.filename}"):
            if candidate != path:
                yield candidate

    def iter_leaves(self, version: str) -> Iterator[Path]:
        yield from sorted((self.root / version).rglob(self.filename))
