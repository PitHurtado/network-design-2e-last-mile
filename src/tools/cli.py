"""Argparse helpers shared by the `scenarios`, `optimize` and `calibrate` commands."""

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from src.tools.artifacts import Artifact, ArtifactKind, ArtifactStore
from src.tools.io import read_json
from src.tools.manifest import Manifest
from src.tools.promotion import Promoter, PromotionError, Reproducer
from src.tools.validation import FAIL, WARN, Validator

REGIMES = ("low", "normal", "high")


def add_regimes(parser: argparse.ArgumentParser, default=REGIMES) -> None:
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(default))


def add_seed_base(parser: argparse.ArgumentParser, default: int) -> None:
    parser.add_argument("--seed-base", type=int, default=default)


def print_validation(result) -> None:
    for item in result.checks:
        mark = {"pass": "OK   ", WARN: "WARN ", FAIL: "FALLA"}[item.status]
        print(f"  {mark} {item.name}: {item.message}")
    print(f"\n{'Validación OK' if result.passed else 'Validación FALLIDA'} · {result.content_sha256[:12]}")
    if result.report:
        print(f"Reporte: {result.report}")


def validate(store: ArtifactStore, ref: str, validators: dict[ArtifactKind, Validator]) -> int:
    artifact = store.resolve(ref)
    if artifact.kind not in validators:
        raise SystemExit(f"{artifact.kind.name.lower()} artifacts are not validated by this command.")
    print(f"Validando {artifact.id} ({artifact.path})")
    result = validators[artifact.kind].run(artifact)
    print_validation(result)
    return 0 if result.passed else 1


def promote(store: ArtifactStore, ref: str, reproducers: dict[ArtifactKind, Reproducer | None]) -> int:
    artifact = store.resolve(ref)
    if artifact.kind not in reproducers:
        raise SystemExit(f"{artifact.kind.name.lower()} artifacts are not promoted by this command.")
    try:
        promoted = Promoter(store).promote(artifact, reproducers[artifact.kind])
    except PromotionError as error:
        print(f"No se promovió {artifact.id}: {error}", file=sys.stderr)
        return 1
    print(f"{artifact.id} -> {promoted.id} ({promoted.path})")
    return 0


def list_artifacts(store: ArtifactStore, kinds: list[ArtifactKind]) -> None:
    for kind in kinds:
        print(f"{kind.name.lower()}:")
        for artifact_id in (store.official_ids(kind) if kind.promotable else []) + store.candidate_ids(kind):
            artifact = store.resolve(artifact_id, kind)
            status = "sin validar"
            if artifact.validation_path.exists():
                status = "validado" if read_json(artifact.validation_path)["passed"] else "validación fallida"
            created = Manifest.load(artifact.manifest_path).created_at if artifact.manifest_path.exists() else "?"
            print(f"  {artifact_id:24s} {created}  {status}")


def show(store: ArtifactStore, ref: str) -> None:
    artifact: Artifact = store.resolve(ref)
    print(json.dumps(read_json(artifact.manifest_path), indent=2, ensure_ascii=False))


def add_lifecycle(subparsers, kinds: str) -> None:
    """`validate`, `promote`, `list`, `show` subcommands."""
    subparsers.add_parser("validate", help=f"run the checks of a {kinds} artifact").add_argument("ref")
    subparsers.add_parser("promote", help=f"turn a validated {kinds} candidate into the next official version").add_argument(
        "ref"
    )
    subparsers.add_parser("list", help=f"official versions and candidates ({kinds})")
    subparsers.add_parser("show", help="print an artifact's manifest").add_argument("ref")


def dispatch(args, handlers: dict[str, Callable]) -> None:
    code = handlers[args.command](args)
    raise SystemExit(code or 0)


def output_path(value: str | None) -> Path | None:
    return Path(value) if value else None
