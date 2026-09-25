"""JSON I/O and content hashing.

`write_json` reproduces `json.dump(obj, file, indent=2)` byte for byte, which is what
every writer in the study used; the scenario goldens depend on it.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Callable


def read_json(path: Path) -> Any:
    with open(path) as file:
        return json.load(file)


def write_json(path: Path, obj: Any, indent: int | None = 2, default: Callable | None = None) -> Path:
    """Write `obj` as JSON, creating the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, default=default))
    return path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(obj: Any) -> str:
    """Stable digest of a JSON-serializable object, independent of key order and formatting."""
    encoded = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
