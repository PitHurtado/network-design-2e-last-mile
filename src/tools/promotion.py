"""Promotion of a validated candidate to the next official, immutable version.

`promote` refuses unless every one of these holds:

* the candidate passed `validate`, and its content has not changed since;
* every parent it was built from is an official version, unchanged;
* it was produced from a clean git tree at the current HEAD, and the tree is still clean;
* re-executing its recorded command reproduces its content byte for byte (when the stage
  provides a reproducer; optimization runs record solver settings instead, because a
  time-limited MIP is not deterministic).
"""

import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from src.tools.artifacts import Artifact, ArtifactStore, content_digest, content_files
from src.tools.io import write_json
from src.tools.manifest import Manifest, git_info
from src.tools.validation import load_validation

# (candidate, its manifest, empty scratch directory) -> writes the re-executed content
Reproducer = Callable[[Artifact, Manifest, Path], None]


class PromotionError(RuntimeError):
    """A candidate that does not meet the conditions to become official."""


class Promoter:
    def __init__(self, store: ArtifactStore, require_clean_tree: bool = True):
        self.store = store
        self.require_clean_tree = require_clean_tree

    def _check_validation(self, artifact: Artifact) -> None:
        validation = load_validation(artifact)
        if validation is None:
            raise PromotionError(f"{artifact} has not been validated; run validate first.")
        if not validation["passed"]:
            failed = [c["name"] for c in validation["checks"] if c["status"] == "fail"]
            raise PromotionError(f"{artifact} failed validation: {failed}")
        if validation["content_sha256"] != content_digest(artifact.path)["sha256"]:
            raise PromotionError(f"{artifact} changed after it was validated; validate it again.")

    def _check_parents(self, manifest: Manifest) -> None:
        for parent in manifest.parents:
            if parent["id"].startswith("c"):
                raise PromotionError(f"Parent {parent['id']} is a candidate; promote it first and rebuild from it.")
            current = self.store.resolve(parent["id"])
            if Manifest.load(current.manifest_path).content["sha256"] != parent["content_sha256"]:
                raise PromotionError(f"Parent {parent['id']} no longer matches the digest recorded at build time.")

    def _check_code(self, manifest: Manifest) -> None:
        if not self.require_clean_tree:
            return
        if manifest.code.get("git_dirty", True):
            raise PromotionError("The candidate was produced from a dirty git tree; commit and regenerate it.")
        now = git_info()
        if now["git_dirty"]:
            raise PromotionError("The git tree has uncommitted changes; commit them before promoting.")
        if now["git_commit"] != manifest.code.get("git_commit"):
            raise PromotionError(
                f"HEAD is {now['git_commit']} but the candidate was produced at {manifest.code.get('git_commit')}; regenerate it."
            )

    def _check_reproduction(self, artifact: Artifact, manifest: Manifest, reproducer: Reproducer) -> None:
        scratch = self.store.sandbox_root(artifact.kind) / f".reproduce-{artifact.id}"
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
        try:
            reproducer(artifact, manifest, scratch)
            original = {p.relative_to(artifact.path).as_posix(): p for p in content_files(artifact.path)}
            again = {p.relative_to(scratch).as_posix(): p for p in content_files(scratch)}
            if set(original) != set(again):
                missing, extra = sorted(set(original) - set(again)), sorted(set(again) - set(original))
                raise PromotionError(f"Re-execution produced a different file set: missing {missing[:5]}, extra {extra[:5]}")
            differing = [name for name, path in original.items() if path.read_bytes() != again[name].read_bytes()]
            if differing:
                raise PromotionError(f"Re-execution is not byte-identical in {len(differing)} files, e.g. {differing[:5]}")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def promote(self, artifact: Artifact, reproducer: Reproducer | None = None) -> Artifact:
        if artifact.official:
            raise PromotionError(f"{artifact} is already official.")
        if not artifact.kind.promotable:
            raise PromotionError(f"{artifact.kind.name.lower()} artifacts are exploratory and cannot be promoted.")
        manifest = Manifest.load(artifact.manifest_path)
        self._check_validation(artifact)
        self._check_parents(manifest)
        self._check_code(manifest)
        if reproducer is not None:
            self._check_reproduction(artifact, manifest, reproducer)

        new_id = self.store.next_official_id(artifact.kind)
        target = self.store.official_root(artifact.kind) / new_id
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(artifact.path, target)
        promoted = Artifact(artifact.kind, new_id, target)
        manifest.id, manifest.promoted_from, manifest.promoted_at = new_id, artifact.id, time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json(promoted.manifest_path, asdict(manifest))
        return promoted
