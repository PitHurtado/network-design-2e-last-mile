"""Versioned artifacts: official, immutable versions and disposable candidates.

Every stage of the study produces an artifact: fitted parameters (`p1`, `p2`, ...),
scenario versions (`v1`, ...), optimization runs (`r1`, ...). Any command writes a
*candidate* into the sandbox, named after its kind and creation time (`cp-20260925-101530`).
A candidate becomes the next official version only through `validate` + `promote`.

    <official root>/<p|v|r><N>/      immutable once promoted
    <sandbox root>/c<p|v|r|c>-<timestamp>/  disposable, gitignored

Inside an artifact, `manifest.json`, `validation.json` and anything under `reports/` are
metadata: they are excluded from the content digest, so a report can be regenerated
without touching an official version's identity.
"""

import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.tools.io import sha256_bytes, sha256_file
from src.tools.paths import DATA_DIR, RESULTS_DIR

MANIFEST = "manifest.json"
VALIDATION = "validation.json"
REPORTS_DIR = "reports"
METADATA = (MANIFEST, VALIDATION)


class ArtifactKind(Enum):
    """Kind of artifact: its id prefix, where officials live, where candidates live."""

    PARAMS = ("p", DATA_DIR / "params", DATA_DIR / "sandbox" / "params")
    SCENARIOS = ("v", DATA_DIR / "scenarios", DATA_DIR / "sandbox" / "scenarios")
    RUNS = ("r", RESULTS_DIR / "runs", RESULTS_DIR / "sandbox" / "runs")
    COMPARISONS = ("c", None, DATA_DIR / "sandbox" / "comparisons")  # exploratory, never promoted

    def __init__(self, prefix: str, official_root: Path | None, sandbox_root: Path):
        self.prefix = prefix
        self.official_root = official_root
        self.sandbox_root = sandbox_root

    @property
    def promotable(self) -> bool:
        return self.official_root is not None

    @property
    def candidate_prefix(self) -> str:
        return f"c{self.prefix}-"

    @classmethod
    def of(cls, ref: str) -> "ArtifactKind":
        """The kind an id belongs to: `p3` -> PARAMS, `cv-...` -> SCENARIOS."""
        for kind in cls:
            if ref.startswith(kind.candidate_prefix) or (kind.promotable and re.fullmatch(rf"{kind.prefix}\d+", ref)):
                return kind
        raise ValueError(f"{ref!r} is not an artifact id (expected e.g. p1, v2, r3 or a cp-/cv-/cr-/cc- candidate).")


@dataclass(frozen=True)
class Artifact:
    kind: ArtifactKind
    id: str
    path: Path

    @property
    def official(self) -> bool:
        return not self.id.startswith("c")

    @property
    def manifest_path(self) -> Path:
        return self.path / MANIFEST

    @property
    def validation_path(self) -> Path:
        return self.path / VALIDATION

    @property
    def reports_dir(self) -> Path:
        return self.path / REPORTS_DIR

    def __str__(self) -> str:
        return self.id


def content_files(root: Path) -> list[Path]:
    """Files that make up an artifact's content: everything but its metadata and reports."""
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if (len(relative.parts) == 1 and relative.name in METADATA) or relative.parts[0] == REPORTS_DIR:
            continue
        if path.name == ".DS_Store":
            continue
        files.append(path)
    return files


def content_digest(root: Path) -> dict:
    """`{sha256, n_files}` over the sorted `(relative path, file sha)` pairs of the content."""
    entries = [f"{path.relative_to(root).as_posix()}\t{sha256_file(path)}" for path in content_files(root)]
    return {"sha256": sha256_bytes("\n".join(entries).encode()), "n_files": len(entries)}


class ArtifactStore:
    """Creates, finds and numbers the artifacts of every kind.

    Roots default to the repository's `data/` and `results/`; tests pass their own.
    """

    def __init__(self, roots: dict[ArtifactKind, tuple[Path | None, Path]] | None = None):
        self.roots = roots or {kind: (kind.official_root, kind.sandbox_root) for kind in ArtifactKind}

    @classmethod
    def under(cls, base: Path) -> "ArtifactStore":
        """A store with every root below `base`, mirroring the repository layout."""
        base = Path(base)
        return cls(
            {
                ArtifactKind.PARAMS: (base / "data" / "params", base / "data" / "sandbox" / "params"),
                ArtifactKind.SCENARIOS: (base / "data" / "scenarios", base / "data" / "sandbox" / "scenarios"),
                ArtifactKind.RUNS: (base / "results" / "runs", base / "results" / "sandbox" / "runs"),
                ArtifactKind.COMPARISONS: (None, base / "data" / "sandbox" / "comparisons"),
            }
        )

    def official_root(self, kind: ArtifactKind) -> Path:
        root = self.roots[kind][0]
        if root is None:
            raise ValueError(f"{kind.name.lower()} artifacts are exploratory and have no official versions.")
        return root

    def sandbox_root(self, kind: ArtifactKind) -> Path:
        return self.roots[kind][1]

    def new_candidate(self, kind: ArtifactKind) -> Artifact:
        """A fresh, empty candidate directory, named after its creation time."""
        root = self.sandbox_root(kind)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        artifact_id, suffix = f"{kind.candidate_prefix}{stamp}", 1
        while (root / artifact_id).exists():
            suffix += 1
            artifact_id = f"{kind.candidate_prefix}{stamp}-{suffix}"
        path = root / artifact_id
        path.mkdir(parents=True)
        return Artifact(kind, artifact_id, path)

    def official_ids(self, kind: ArtifactKind) -> list[str]:
        root = self.roots[kind][0]
        if root is None or not root.exists():
            return []
        numbered = [p.name for p in root.iterdir() if p.is_dir() and re.fullmatch(rf"{kind.prefix}\d+", p.name)]
        return sorted(numbered, key=lambda name: int(name[1:]))

    def candidate_ids(self, kind: ArtifactKind) -> list[str]:
        root = self.sandbox_root(kind)
        if not root.exists():
            return []
        return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith(kind.candidate_prefix))

    def next_official_id(self, kind: ArtifactKind) -> str:
        ids = self.official_ids(kind)
        return f"{kind.prefix}{int(ids[-1][1:]) + 1 if ids else 1}"

    def resolve(self, ref: str, kind: ArtifactKind | None = None) -> Artifact:
        """An existing artifact from its id; `latest` needs `kind` and means the newest official."""
        if ref == "latest":
            if kind is None:
                raise ValueError("'latest' needs the artifact kind.")
            ids = self.official_ids(kind)
            if not ids:
                raise FileNotFoundError(f"There is no official {kind.name.lower()} version yet.")
            ref = ids[-1]
        found = ArtifactKind.of(ref)
        if kind is not None and found is not kind:
            raise ValueError(f"{ref!r} is a {found.name.lower()} artifact, expected {kind.name.lower()}.")
        root = self.sandbox_root(found) if ref.startswith("c") else self.official_root(found)
        path = root / ref
        if not path.is_dir():
            raise FileNotFoundError(f"Artifact {ref} not found under {root}.")
        return Artifact(found, ref, path)
