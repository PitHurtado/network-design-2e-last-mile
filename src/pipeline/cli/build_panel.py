"""Build the monthly demand panel from the raw delivery-event file.

poetry run python -m src.pipeline.cli.build_panel
"""

import argparse

from src.core.logging import get_logger
from src.pipeline.demand_panel import build_panel, save_panel

logger = get_logger("BuildPanel")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="build the panel but do not write it")
    args = parser.parse_args()

    panel = build_panel()

    print("\nDemanda por año:")
    print(panel[~panel["excluded"]].groupby("year")["items"].sum().round(0).to_string())
    print("\nClientes promedio por período, por píxel:")
    per_pixel = panel[~panel["excluded"]].groupby("id_pixel")["n_customers"].mean()
    print(
        f"  min={per_pixel.min():.2f}  p05={per_pixel.quantile(0.05):.2f}  "
        f"mediana={per_pixel.median():.2f}  max={per_pixel.max():.2f}"
    )
    print(f"  píxeles bajo 5 clientes promedio: {int((per_pixel < 5).sum())} de {len(per_pixel)}")
    print("\nCobertura del crosswalk (share de items con layer conocido):")
    cw = panel.groupby("id_pixel")["crosswalk_share"].mean()
    print(
        f"  media={cw.mean():.3f}  mediana={cw.median():.3f}  " f"píxeles con menos de 50% del crosswalk: {int((cw < 0.5).sum())}"
    )

    if args.dry_run:
        logger.info("--dry-run: panel not written.")
        return
    save_panel(panel)


if __name__ == "__main__":
    main()
