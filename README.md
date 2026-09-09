# Two-Echelon Last-Mile Delivery Network Design under Uncertainty

**Research project** — stochastic optimization of two-echelon (DC → satellites → pixels) last-mile delivery networks with multiple capacity levels and multi-period demand uncertainty.

**Authors:** Pitehr Hurtado-Cayo · Juan C. Pina-Pardo · Selene Silvestri · Matthias Winkenbach

---

## Overview

The network has two echelons:

1. **Distribution Center (DC)** — serves pixels directly using large vehicles when no satellite is open nearby.
2. **Satellites** — intermediate facilities that can be opened or closed; serve pixels using small vehicles.

Demand is uncertain and modeled via sampled scenarios. The model decides which pixels each satellite (or the DC) should serve in each period and scenario, minimizing expected total routing cost.

The Continuous Approximation (CA) method estimates routing costs analytically from pixel geometry (area, density, demand), so the LP does not require explicit route enumeration.

---

## Repository structure

`src/` splits along the one seam that matters: everything that happens **before** a
solve, and everything that happens **during and after** one. The two never import each
other — they meet on disk, at the scenario contract.

```
.
├── data/
│   ├── raw_demand/          # 192 MB delivery-event CSV (untracked)
│   ├── raw_pixel/           # Pixel grid, geometry, customer-pixel crosswalk
│   ├── raw_facility/        # Satellite location and cost data
│   ├── raw_distance/        # Distance matrices (facility↔pixel, DC↔facility)
│   ├── raw_location/
│   └── scenarios/           # Monthly panel, fitted shape params, generated scenarios
├── results/                 # Result JSONs and HTML reports (untracked)
└── src/
    ├── core/                # Shared: paths, vehicle params, entities, logging, input readers
    ├── pipeline/            # Preprocessing: events → panel → shape params → scenarios
    │   ├── reports/         #   scenario validation and exploration HTML
    │   └── cli/             #   runnable steps, in pipeline order
    └── optimization/        # Scenarios → CA → Gurobi → results → HTML
        ├── routing/         #   Continuous Approximation
        ├── models/          #   base.py + one thin subclass per variant
        ├── experiments/     #   powerset sweeps            (not ported yet)
        ├── reports/         #   interactive result HTML    (not ported yet)
        └── cli/             #   runnable solves
```

The dependency direction is the invariant worth checking after any change:

```bash
grep -rn "^from src\.optimization" src/pipeline      # must print nothing
grep -rn "^from src\.pipeline"     src/optimization  # must print nothing
```

---

## Setup

Requires Python ≥ 3.10, [Poetry](https://python-poetry.org/), and a valid **Gurobi license**.

```bash
poetry install
```

---

## Running the pipeline

To reproduce the full historical fit, run the following commands in order. `--n` must
match between calibration and generation: the regime multipliers use the exact seed
sequence consumed by the generator.

```bash
poetry run python -m src.pipeline.cli.build_panel            # raw events → monthly panel
poetry run python -m src.pipeline.cli.fit_params --n 50      # panel → shape_params.json
poetry run python -m src.pipeline.cli.generate --all --version v3  # params → versioned scenario sets
poetry run python -m src.pipeline.cli.analyze                # validation report (exit ≠ 0 on failure)
poetry run python -m src.pipeline.cli.explore                # exploratory report
```

### Reproduce the regime comparison from persisted parameters

When the historical raw demand/panel is unavailable, use the versioned
`data/scenarios/shape_params.json` to recalibrate only the regime policy (70% of the
regime effect in `stop`, 30% in `drop`), regenerate every scenario and rebuild both
reports:

```bash
poetry run python -m src.pipeline.cli.recalibrate_regimes --validation-n 100
poetry run python -m src.pipeline.cli.generate --all --version v3
poetry run python -m src.pipeline.cli.analyze
poetry run python -m src.pipeline.cli.explore
open results/explore_scenarios.html
```

The explorer compares low/normal/high on a common map scale for demand, stops, or
drop. It also reports scenario × period distributions, per-pixel variability, and
period bands for all three measures. `normal` is the reference for relative changes.

Each version is stored as `data/scenarios/generated/<version>/<regime>/<set>/` with a
manifest containing canonical IDs, seed scheme and a SHA-256 of `shape_params.json`.
`optimization` has 30 simulated scenarios, `validation` has 100 independent simulated
scenarios, `expected` has the 12-period mean-shock scenario, and `annual_expected` has
one annual-average period for descriptive analysis only (it cannot enter the 12-period
optimizer). Use `--optimization-n`, `--validation-n` and `--seed-base` to create a
different explicitly labelled version. Existing folders are protected; use
`--overwrite` only to rebuild the identical version deliberately.

## Running the models

```bash
poetry run python -m src.optimization.cli.verify_end_to_end --n 3   # CA + Gurobi smoke test
```

The powerset experiment and the extended-model drivers have not been ported to this
layout yet; their reference implementation is under `OLD/src/entrypoints/`.

### The model family

The four formulations nest strictly, so they are one base class plus thin subclasses.
`src/optimization/models/base.py` owns the shared formulation — the assignment
variables, both routing terms, the demand constraint, `solve()` — and a registry of
optional blocks. A variant declares which blocks it enables; it does not restate the
base.

| Model | Adds over the base | Status |
|---|---|---|
| `uncapacitated` | nothing | ported |
| `capacitated` | `Y[i,q]`, installation cost, one-level and capacity constraints | pending |
| `flex` | `Z[i,q,t,n]`, operation cost, one-operating-level and `Z ≤ Y` | pending |
| `extended` | flex's variable set under `type_of_flexibility` | pending |

Two properties the registry enforces, because both were silent failure modes before:

- **Installation cost is not averaged by `1/N`.** Every objective block declares whether
  it is scenario-dependent, so a new variant cannot get the averaging wrong by omission.
  This is also why cost components in a result JSON do not sum to `objective`.
- **`Status` travels with the objective.** `solve()` records `status` and `is_optimal`,
  and reports `objective_value = None` when there is no feasible solution, so a
  time-limit incumbent can no longer be read as an optimum.

Ablations do not need a new class — `disabled_blocks` names blocks, not flags, so a
single constraint can be dropped while its variables stay:

```python
from dataclasses import replace
model = FlexSAAModel(instance, features=replace(FlexSAAModel.DEFAULT_FEATURES,
                                                disabled_blocks=frozenset({"capacity"})))
```

The enabled feature set is recorded in the results dict, so every run says what it was.

---

## Visualization

### Single-solution map

Generates an interactive HTML map from any result JSON. Dropdowns for period and layer (DC / satellite). Hover shows demand, cost, assigned satellite, and fleet size per pixel.

> Not ported to this layout yet. The reference implementation is
> `OLD/src/visualization/solution_map.py`; the port lands in `src/optimization/reports/`.

```bash
cd OLD && poetry run python -m src.visualization.solution_map results/uncapacitated_saa/uncapacitated_1_True_None.json
```

### Powerset summary

Builds one HTML per satellite combination plus a master `summary.html` with:
- Scatter of Δ% objective vs. number of active satellites (with efficiency frontier)
- Boxplot of objective distribution by subset size
- Sortable table of all 512 configurations
- Iframe viewer to inspect any individual solution map

> Not ported to this layout yet — see `OLD/src/visualization/powerset_html.py`.

```bash
cd OLD && poetry run python -m src.visualization.powerset_html
```

Use `--skip-individual` to regenerate only the summary without re-rendering per-config maps.

---

## Analysis

### CA factor analysis

Identifies which input factors (distance, density, demand, area, drop size) drive routing costs under the Continuous Approximation. Outputs a self-contained HTML report with four sections:

| Section | Question answered |
|---|---|
| **Cost decomposition** | Which cost component (line-haul, intra-stop, fixed, preparation) dominates, and how does it vary across satellites? |
| **Log-linear elasticities** | Which factor has the largest % impact on each cost component? (OLS in log-log space) |
| **Partial Dependence Plots** | What is the shape of the relationship between each factor and total cost? |
| **Fleet size analysis** | What drives fleet size requirements at each satellite? |

Each section includes a methodology box (what data was used and how the plot was built) and an insight box (why the observed behavior occurs mathematically).

> Not ported to this layout yet — see `OLD/src/analysis/ca_factor_analysis.py`.

```bash
cd OLD && poetry run python -m src.analysis.ca_factor_analysis
```

---

## Model formulation

### Decision variables

| Variable | Domain | Description |
|---|---|---|
| `X[i,k,t,n]` | [0,1] (continuous) or {0,1} | Fraction of pixel `k` served by satellite `i` in period `t`, scenario `n` |
| `W[k,t,n]` | [0,1] | Fraction of pixel `k` served directly from the DC in period `t`, scenario `n` |

### Objective

Minimize expected total routing cost across scenarios:

```
min  (1/N) · Σ_n [ Σ_{i,k,t} c_facility[i,k,t,n] · X[i,k,t,n]
                  + Σ_{k,t}   c_dc[k,t,n]         · W[k,t,n]   ]
```

where costs `c_facility` and `c_dc` are computed by the Continuous Approximation before solving the LP.

### Constraints

- **Coverage:** every pixel must be fully covered in every period and scenario — `Σ_i X[i,k,t,n] + W[k,t,n] = 1`
- **Satellite activation:** assignment to satellite `i` is only allowed if `i` is in the active subset

### Continuous Approximation cost components

For each (satellite, pixel, vehicle type, period):

| Component | Formula sketch |
|---|---|
| Line-haul | distance × fleet trips × cost per km |
| Intra-stop | √(area / stops) × stop density × cost per km |
| Fixed | fleet size × fixed vehicle cost |
| Tour preparation | number of tours × preparation cost |

---

## Key parameters

| Parameter | Description |
|---|---|
| `N` | Number of demand scenarios per sample |
| `T` (periods) | Number of time periods (default: 12) |
| `type_of_flexibility` | `FIXED_CAPACITY` or `FLEX_CAPACITY` — whether satellites can change capacity across periods |
| `use_euclidean_distance` | Use straight-line distances instead of road distances |
| `facilities_subset` | Restrict the active satellite set (used by the powerset experiment) |
| `max_run_time` | Gurobi time limit in seconds |
