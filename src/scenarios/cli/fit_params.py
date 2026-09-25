"""Fit the scenario shape parameters and write `shape_params.json`.

    poetry run python -m src.scenarios.cli.fit_params

Produces the single reproducible input of the whole study: pixel levels, layer
seasonality, dispersion, the spatial decay, and the regime multipliers.
"""

import argparse

from src.core.constants import PATH_SHAPE_PARAMS, REGIME_TARGETS
from src.core.inputs import get_pixels
from src.scenarios.fitting.fitter import ParamsFitter
from src.scenarios.fitting.panel import load_panel
from src.scenarios.spatial import model_correlation
from src.tools.logging import get_logger

logger = get_logger("FitParams")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=12, help="distance bins for the correlogram")
    parser.add_argument("--n", type=int, default=50, help="scenarios per regime (must match the generate step)")
    args = parser.parse_args()

    result = ParamsFitter(bins=args.bins, n_calibration=args.n).fit(load_panel(), set(get_pixels()))
    spatial = result.spatial

    print("\nCorrelograma empírico vs ajuste:")
    print(f"  {'h (km)':>8} {'emp.':>8} {'ajuste':>8} {'pares':>9}")
    for _, row in result.correlogram.iterrows():
        pred = model_correlation(row["h_km"], spatial["rho_km"], spatial["nugget"], spatial["plateau"])
        print(f"  {row['h_km']:8.2f} {row['corr']:8.3f} {pred:8.3f} {int(row['n_pairs']):9d}")

    result.params.save(PATH_SHAPE_PARAMS)
    logger.info(f"Shape parameters written to {PATH_SHAPE_PARAMS}")

    print("\nRegímenes calibrados:")
    for regime, calibrated in result.regimes.items():
        print(
            f"  {regime:8s} objetivo={REGIME_TARGETS[regime]:>9,.0f} "
            f"multiplicador={calibrated['multiplier']:6.4f}  realizado={calibrated['realized']:>9,.0f}"
        )


if __name__ == "__main__":
    main()
