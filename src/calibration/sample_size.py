"""Choosing the SAA sample size N.

The plan, per candidate N and replication r (independent scenario draws of size N):

* in-sample stability: dispersion of the SAA objective across replications;
* out-of-sample stability: each replication's first-stage decision (installation Y)
  evaluated on a common large validation set, as `optimize evaluate` already does;
* optimality gap (Mak, Morton & Wood, 1999): the mean of the SAA objectives is a
  statistical lower bound, the validation cost of a candidate Y an upper bound; their
  difference with a confidence interval bounds the gap of that Y.

N is the smallest value whose gap and out-of-sample dispersion fall under a tolerance.
Replications need scenario sets larger than one optimization set; they will come from
dedicated `scenarios generate` versions, with streams disjoint from validation.
"""

from dataclasses import dataclass, field

from src.optimization.instance import InstanceBuilder


@dataclass(frozen=True)
class SampleSizeConfig:
    scenarios: str  # scenario version with enough simulated scenarios for every replication
    ns: tuple[int, ...] = (5, 10, 20, 30, 50)
    replications: int = 10
    regimes: tuple[str, ...] = ("normal",)
    flexibility: str = "up_to_installed"
    confidence: float = 0.95
    solver: dict = field(default_factory=lambda: {"TimeLimit": 600.0, "MIPGap": 0.0})


@dataclass
class SampleSizeResult:
    """Per (regime, N): objective mean/std across replications, out-of-sample mean/std, MMW gap and CI."""

    rows: list[dict]
    recommended_n: dict[str, int]


class SampleSizeStudy:
    """Runs the replications for every N and summarizes stability and the optimality gap."""

    def __init__(self, config: SampleSizeConfig, builder: InstanceBuilder):
        self.config = config
        self.builder = builder

    def in_sample_stability(self) -> list[dict]:
        """SAA objective per (regime, N, replication)."""
        raise NotImplementedError("Sample-size calibration is planned but not implemented yet.")

    def out_of_sample_stability(self) -> list[dict]:
        """Validation cost of each replication's installation decision."""
        raise NotImplementedError("Sample-size calibration is planned but not implemented yet.")

    def mmw_gap(self) -> list[dict]:
        """Mak-Morton-Wood optimality-gap estimate and confidence interval per (regime, N)."""
        raise NotImplementedError("Sample-size calibration is planned but not implemented yet.")

    def run(self) -> SampleSizeResult:
        raise NotImplementedError("Sample-size calibration is planned but not implemented yet.")
