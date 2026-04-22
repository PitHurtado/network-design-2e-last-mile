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

```
.
├── data/
│   ├── facilities/          # Satellite location and cost data
│   ├── pixels/              # Delivery zone geometry and demand
│   ├── distances/           # Pre-computed distance matrices (facility↔pixel, DC↔facility)
│   └── scenarios/           # Sampled demand scenarios
├── results/
│   ├── uncapacitated_saa/   # Single-run model outputs (JSON + HTML)
│   ├── powerset_experiment/ # 2^9 = 512 satellite-combination outputs
│   └── analysis/            # CA factor analysis HTML report
├── src/
│   ├── models/              # Gurobi LP models
│   ├── routing_tools/       # Continuous Approximation implementation
│   ├── utils/               # Instance, Scenario, data classes
│   ├── data/                # ETL loaders
│   ├── entrypoints/         # Runnable scripts
│   ├── visualization/       # Interactive HTML generators
│   └── analysis/            # Factor analysis scripts
└── pyproject.toml
```

---

## Setup

Requires Python ≥ 3.10, [Poetry](https://python-poetry.org/), and a valid **Gurobi license**.

```bash
poetry install
```

---

## Running the models

### Uncapacitated SAA (single run)

Solves the assignment LP for all 9 satellites with euclidean distances and N=1 scenario.

```bash
poetry run python -m src.entrypoints.run_uncapacitated_saa
# → results/uncapacitated_saa/uncapacitated_1_True_None.json
```

### Powerset experiment (all satellite combinations)

Runs the model for all 2^9 = 512 subsets of satellites (including DC-only). Skips configs that already have a result JSON.

```bash
poetry run python -m src.entrypoints.run_powerset_experiment
```

### Extended SAA

Solves the full model with flexible or fixed capacity, using pre-sampled expected scenarios.

```bash
poetry run python -m src.entrypoints.run_extended_saa
```

---

## Visualization

### Single-solution map

Generates an interactive HTML map from any result JSON. Dropdowns for period and layer (DC / satellite). Hover shows demand, cost, assigned satellite, and fleet size per pixel.

```bash
poetry run python -m src.visualization.solution_map results/uncapacitated_saa/uncapacitated_1_True_None.json
open results/uncapacitated_saa/uncapacitated_1_True_None.html
```

### Powerset summary

Builds one HTML per satellite combination plus a master `summary.html` with:
- Scatter of Δ% objective vs. number of active satellites (with efficiency frontier)
- Boxplot of objective distribution by subset size
- Sortable table of all 512 configurations
- Iframe viewer to inspect any individual solution map

```bash
poetry run python -m src.visualization.powerset_html
open results/powerset_experiment/summary.html
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

```bash
poetry run python -m src.analysis.ca_factor_analysis
open results/analysis/ca_factor_analysis.html
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
