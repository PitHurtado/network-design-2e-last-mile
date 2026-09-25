"""Contract invariants and degeneracy checks of generated scenario sets."""

import numpy as np
import pandas as pd

from src.core.constants import N_PERIODS
from src.tools.validation import Check, check


def contract_checks(generated: dict[str, pd.DataFrame], manifests: dict[str, dict]) -> list[Check]:
    """What must hold for the CA and the Gurobi models to run on a set untouched.

    `generated` holds the long frame of each regime's simulated set, `manifests` its manifest.
    """
    checks = []
    for regime, frame in generated.items():
        manifest = manifests[regime]
        n_pixels = frame["id_pixel"].nunique()
        checks += [
            check(f"[{regime}] píxeles por escenario == 161", n_pixels == 161, f"{n_pixels}"),
            check(f"[{regime}] períodos == {N_PERIODS}", frame["period"].nunique() == N_PERIODS, f"{frame['period'].nunique()}"),
            check(f"[{regime}] stop >= 1", int(frame["stop"].min()) >= 1, f"min {int(frame['stop'].min())}"),
            check(f"[{regime}] drop > 0", float(frame["drop"].min()) > 0, f"min {frame['drop'].min():.4f}"),
            check(f"[{regime}] demand > 0", float(frame["demand"].min()) > 0, f"min {frame['demand'].min():.4f}"),
            check(
                f"[{regime}] demand == stop × drop",
                bool(np.allclose(frame["demand"], (frame["stop"] * frame["drop"]).round(4), atol=1e-6)),
                "exacto",
            ),
            check(
                f"[{regime}] piso max(1,·) no liga",
                manifest["stop_floor_share"] < 1e-4,
                f"{manifest['stop_floor_share'] * 100:.3f}% de celdas",
            ),
            check(
                f"[{regime}] objetivo alcanzado (±1%)",
                abs(manifest["period_total_mean"] / manifest["target_period_demand"] - 1) < 0.01,
                f"{manifest['period_total_mean']:,.0f} vs {manifest['target_period_demand']:,.0f}",
            ),
        ]
    return checks
