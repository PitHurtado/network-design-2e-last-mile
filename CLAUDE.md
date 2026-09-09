# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

Two-echelon (DC → satellites → pixels) last-mile network design under demand
uncertainty, solved with Gurobi over scenarios whose routing costs come from a
Continuous Approximation. Research code for a paper.

The repo is mid-refactor on `feature/refactor-and-cleaning`. The **scenario pipeline
has been rebuilt** in `src/` and is working end to end; the **optimization
experiments have not been ported yet**.

`src/` was reorganized into three packages along the preprocess / solve seam:
`src/core/` (shared), `src/pipeline/` (preprocessing) and `src/optimization/`.
`pipeline` and `optimization` must never import each other — they meet on disk, at the
scenario contract. That is the one structural invariant, and it is cheap to check:

```bash
grep -rn "^from src\.optimization" src/pipeline      # must print nothing
grep -rn "^from src\.pipeline"     src/optimization  # must print nothing
```

Ported and working: all of `src/core/` and `src/pipeline/`, plus
`src/optimization/{scenario,instance}.py`, `routing/`, `models/{base,uncapacitated}.py`
and `cli/verify_end_to_end.py`.

Still only in `OLD/src/`: the capacitated / flex / extended models, the powerset and
experiment drivers, `visualization/`, and `analysis/ca_factor_analysis.py`. The
directories that will receive them (`src/optimization/experiments/`,
`src/optimization/reports/`) exist and say so in their `__init__.py`.

**`OLD/` is gitignored and untracked** — it exists only on this machine (plus an
`OLD.zip` beside it) and cannot be recovered from git. It is the reference
implementation for everything not yet ported. Never modify or delete it; copy out of
it.

## Commands

Dependencies are managed with Poetry. A Gurobi license is required for any solve
(academic license, expires 2027-04-22).

```bash
poetry install
poetry shell        # then run the module commands below directly
```

`pyproject.toml` was realigned with what the code actually imports, so
**`poetry.lock` is stale** — run `poetry lock` before trusting a fresh
`poetry install`. Notebook-only dependencies live in an optional group:
`poetry install --with notebooks`.

### Preprocessing pipeline (`src/pipeline/`)

```bash
python -m src.pipeline.cli.build_panel            # raw events -> monthly panel
python -m src.pipeline.cli.fit_params --n 50      # panel -> shape_params.json
python -m src.pipeline.cli.generate --all --n 50  # params -> scenarios per regime
python -m src.pipeline.cli.analyze                # validation report (exit != 0 on failure)
python -m src.pipeline.cli.explore                # exploratory report
```

`--n` must match between `fit_params` and `generate`: the regime multipliers are
calibrated against the exact seed sequence the generator consumes.

### Optimization (`src/optimization/`)

```bash
python -m src.optimization.cli.verify_end_to_end --n 3  # CA + Gurobi smoke test
```

### Legacy code (still in `OLD/`)

Run from `OLD/`, because `constants.py` there derives `ROOT_DIR` from
`parents[1]`, so `DATA_DIR` resolves to `OLD/data`:

```bash
cd OLD && poetry run python -m src.entrypoints.run_powerset_experiment
```

Note `OLD/src/constants.py` points `PATH_DATA_PIXEL` at
`OLD/data/pixels/input_pixels.xlsx`, which does not exist — the only copy is
`data/raw_pixel/input_pixels.xlsx`, so `get_pixels()` fails there. The ported
`src/core/constants.py` already points at the right place.

### Lint

Line length is **130** everywhere: black, isort, pylint and flake8. `.flake8` used to
say 120, so black reformatted to 130 and flake8 then rejected what black had just
written; that is fixed.

`flake8` is not a dev dependency — it only exists inside pre-commit's own venv, so
`poetry run flake8` fails with "Command not found". Use pre-commit:

```bash
poetry run black . && poetry run isort .
poetry run pre-commit run --all-files   # black, isort, flake8, detect-secrets
```

**There are no tests.** Verification is the `analyze` report plus
`verify_end_to_end`.

## Architecture

A one-way pipeline. Understanding it matters more than any individual file:

```
raw events  →  monthly panel  →  shape params  →  scenarios  →  CA  →  Gurobi LP  →  results  →  HTML
data/raw_*    data/scenarios/   shape_params    generated/    optimization/         results/   optimization/
              panel_monthly     .json                         routing/  models/                reports/
└──────────────────── src/pipeline/ ─────────────────────┘└──────── src/optimization/ ────────────────────┘
```

The arrow between `generated/` and the CA is the package boundary: `pipeline` writes
those JSONs, `optimization` reads them, and neither imports the other.

The pivotal design decision: **routing costs are computed analytically before the
solve.** The Continuous Approximation turns pixel geometry (area, customer density,
drop size) into a cost and fleet size per
`(facility, pixel, vehicle, period, scenario)`, so the LP never enumerates routes —
it only chooses assignments over precomputed costs. A bug in the CA does not crash
anything; it silently shifts every downstream number.

**Two echelons:** the DC serves pixels directly with large vehicles
(`first_echelon_truck`), satellites serve pixels with small vehicles (`van`). Nine
candidate satellites, 161 pixels, 12 periods, La Paz.

**Configuration split:** `core/constants.py` holds paths, grid geometry and regime
targets; `core/config.py` holds the vehicle parameter dicts. `core/inputs.py` is the
only module that reads the raw files, and it is shared on purpose: `pipeline` needs the
pixel grid to fit shape parameters, `optimization` needs all of it to build an
`Instance`.

**Model family:** `optimization/models/base.py` owns the shared formulation plus a
registry of optional blocks (variables, objective components, constraints), each with a
flag. A variant is a thin subclass that sets `DEFAULT_FEATURES` and implements the
blocks it enables; it never restates the base. Two things the registry enforces because
both used to be silent failure modes: each objective block declares whether it is
averaged by `1/N` (installation cost is not), and `solve()` records `Status` and
`is_optimal` next to the objective. Ablation is `disabled_blocks`, which names blocks
rather than flags, so one constraint can be dropped while its variables stay; unknown
names raise. When porting a model from `OLD/`, implement its blocks — do not copy the
build/solve scaffolding.

The scenario model itself is documented in
`data/scenarios/ESCENARIOS_DOCUMENTACION.md` — read it before touching
`src/pipeline/`.

## Conventions and traps

- **Two demand scales, and mixing them inflates costs ~3×.** *Model demand* is
  `Σ (stop × drop)` — one representative delivery round in the period, ~45k/period
  in 2022. *Raw items* is the monthly count, ~149k/month, because a customer is
  served several times a month. Regime targets and everything in the contract are in
  model demand.
- **The scenario contract is four fields** per pixel: `id_pixel`, `stop`, `drop`,
  `demand`, arrays of length 12. `id_scenario` and `type` are written but never read
  — identity comes from the filename. `k`, `lon`, `lat`, `area_surface` come from
  `input_pixels.xlsx`, not the scenario file.
- **`stop >= 1` and `drop > 0` are hard invariants.** The CA only writes cost keys
  for `demand > 0` and `BaseSAAModel._obj_routing_facilities` indexes them directly
  (not `.get(key, 0)`, which is what the `OLD/` capacitated model did — that priced a
  missing key at zero instead of failing), so one
  zero pixel-period is a `KeyError` at solve time; `drop` is a divisor and `density`
  sits under a `sqrt` in a divisor, so a zero there is a `ZeroDivisionError` first.
- **The pixel set must equal `input_pixels.xlsx` exactly.** A pixel present in the
  scenario but missing from the grid is dropped with a warning and the run continues
  with fewer pixels.
- **12 periods, fixed.** `N_PERIODS` in `constants.py`; the CA loop and the models
  both assume it. `Instance` now raises if asked for anything else.
- **Result JSON keys are stringified Python tuples**, e.g.
  `"('ABAROA', 'B-106', 4, '1')"`. Parse with `ast.literal_eval`, never by splitting;
  the scenario index inside the tuple is a *string*.
- **Cost components do not sum to `objective`**: scenario-dependent terms are
  averaged by `1/N`, installation cost is not.
- **CA expressions carry `# [unit]` comments** — keep them and keep them correct.
  Some are inconsistent (`intra_tour_time_per_customer` is annotated
  `[hour/customer]` at its definition and `[hour/sqrt(customer)]` where it is used).
  Dimensional errors are the dominant failure mode there.
- **`area_surface` is an area in km²** — verified as the count of merged ~1 km² grid
  cells, matching all 161 pixels exactly.
- **`Status` is recorded with the objective** — `BaseSAAModel.solve()` returns
  `status`, `is_optimal` and `objective_value = None` when there is no feasible
  solution. The `OLD/` models did not inspect it, so a time-limit incumbent looked
  like an optimum; do not reintroduce that when porting.
- Pylint permits single-capital names (`X`, `W`, `Y`, `N`, `Q`, `i`, `k`, `t`) so code
  can mirror the formulation's notation. Use the README's symbols.
- **What is versioned:** `.gitignore` no longer excludes `data/` wholesale. Fitted
  shape parameters, the scenario documentation and the small raw inputs are tracked;
  the 192 MB demand CSV, the derived panel, generated scenarios and `results/` are
  not.

## Subagents and skills

`.claude/agents/`: `model-formulation-reviewer` (Gurobi code vs. the README
formulation, now `src/optimization/models/`), `ca-auditor` (CA dimensional analysis,
now `src/optimization/routing/`), `results-analyst` (mining result JSONs without
flooding context), `refactor-migrator` (porting `OLD/src` modules with
numerical-equivalence proof).

`.claude/skills/viz-report/`: house style for the interactive HTML reports (Plotly
via CDN, method box + insight box per section, insights computed from the data).
