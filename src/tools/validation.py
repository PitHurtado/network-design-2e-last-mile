"""Pass/fail checks, and the `validation.json` a validator leaves inside an artifact."""

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

from src.tools.artifacts import Artifact, content_digest
from src.tools.io import read_json, write_json
from src.tools.manifest import git_info

PASS, WARN, FAIL = "pass", "warn", "fail"


@dataclass(frozen=True)
class Check:
    """One named check with its outcome and a human-readable detail."""

    name: str
    status: str  # pass | warn | fail
    message: str = ""
    value: float | int | str | None = None
    threshold: float | int | str | None = None

    @property
    def passed(self) -> bool:
        return self.status != FAIL

    def as_dict(self) -> dict:
        return asdict(self)


def check(name: str, ok: bool, message: str = "", warn_only: bool = False, **extra) -> Check:
    return Check(name, PASS if ok else (WARN if warn_only else FAIL), message, **extra)


def n_failed(checks: list[Check]) -> int:
    return sum(1 for item in checks if item.status == FAIL)


@dataclass
class ValidationResult:
    artifact_id: str
    content_sha256: str
    checks: list[Check]
    report: str | None = None

    @property
    def passed(self) -> bool:
        return n_failed(self.checks) == 0

    def save(self, artifact: Artifact) -> Path:
        return write_json(
            artifact.validation_path,
            {
                "schema_version": 1,
                "artifact_id": self.artifact_id,
                "content_sha256": self.content_sha256,
                "validated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "git_commit": git_info()["git_commit"],
                "passed": self.passed,
                "checks": [item.as_dict() for item in self.checks],
                "report": self.report,
            },
        )


def load_validation(artifact: Artifact) -> dict | None:
    return read_json(artifact.validation_path) if artifact.validation_path.exists() else None


class Validator(ABC):
    """Runs a stage's checks on an artifact and records the outcome in `validation.json`."""

    @abstractmethod
    def checks(self, artifact: Artifact) -> list[Check]:
        """The stage-specific checks."""

    def report(self, artifact: Artifact) -> Path | None:  # pylint: disable=unused-argument
        """Optional HTML report written under `artifact.reports_dir`."""
        return None

    def run(self, artifact: Artifact) -> ValidationResult:
        checks = self.checks(artifact)
        report = self.report(artifact)
        result = ValidationResult(
            artifact_id=artifact.id,
            content_sha256=content_digest(artifact.path)["sha256"],
            checks=checks,
            report=report.relative_to(artifact.path).as_posix() if report else None,
        )
        result.save(artifact)
        return result
