"""Pass/fail checks: the vocabulary every validator reports in."""

from dataclasses import asdict, dataclass

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


def check(name: str, ok: bool, message: str = "", **extra) -> Check:
    return Check(name, PASS if ok else FAIL, message, **extra)


def n_failed(checks: list[Check]) -> int:
    return sum(1 for item in checks if item.status == FAIL)
