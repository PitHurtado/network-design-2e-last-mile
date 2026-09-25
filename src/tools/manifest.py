"""The manifest of an artifact: everything needed to reproduce it and to trace where it came from.

command      the program, subcommand, argv and fully-resolved configuration
inputs       sha256 of every raw file read, and the parent artifacts (id + digests)
seeds        the base seed and the seeding scheme
code         git commit, and whether the tree had uncommitted changes
environment  interpreter and library versions (numpy/BLAS change Cholesky bytes)
content      digest of the artifact's content, as produced
details      stage-specific summary (e.g. calibrated multipliers, solved leaves)
"""

import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path

from src.tools.artifacts import Artifact, content_digest
from src.tools.io import read_json, sha256_file, write_json
from src.tools.paths import ROOT_DIR

SCHEMA_VERSION = 2
LIBRARIES = ("numpy", "pandas", "scipy", "plotly", "gurobipy")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def raw_input(path: Path) -> dict:
    """Identity of one raw input file: repository-relative path, sha256 and size."""
    path = Path(path)
    try:
        shown = path.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        shown = str(path)
    return {"path": shown, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def parent_ref(artifact: Artifact) -> dict:
    """Identity of a parent artifact: id plus the digests it had when it was used."""
    manifest = read_json(artifact.manifest_path)
    return {
        "kind": artifact.kind.name.lower(),
        "id": artifact.id,
        "manifest_sha256": sha256_file(artifact.manifest_path),
        "content_sha256": manifest["content"]["sha256"],
    }


def git_info() -> dict:
    """HEAD commit and whether tracked files differ from it (untracked files are ignored)."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT_DIR, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": True}


def environment_info() -> dict:
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version.split()[0], **versions, "platform": platform.platform()}


@dataclass
class Manifest:  # pylint: disable=too-many-instance-attributes
    kind: str
    id: str
    command: dict
    inputs: dict = field(default_factory=lambda: {"raw": [], "parents": []})
    seeds: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)
    code: dict = field(default_factory=git_info)
    environment: dict = field(default_factory=environment_info)
    content: dict = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    promoted_from: str | None = None
    promoted_at: str | None = None
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        return cls(**read_json(path))

    def save(self, artifact: Artifact) -> Path:
        """Record the content digest as produced and write the manifest into the artifact."""
        self.content = content_digest(artifact.path)
        return write_json(artifact.manifest_path, asdict(self))

    @property
    def parents(self) -> list[dict]:
        return self.inputs.get("parents", [])

    def parent(self, kind: str) -> dict | None:
        return next((item for item in self.parents if item["kind"] == kind), None)
