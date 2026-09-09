# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

Two-echelon (DC → satellites → pixels) last-mile network design under demand
uncertainty, solved with Gurobi over scenarios whose routing costs come from a
Continuous Approximation. Research code for a paper.

The repo is mid-refactor on `feature/refactor-and-cleaning`. The **scenario pipeline
has been rebuilt** in `src/` and is working end to end; the **optimization
experiments have not been ported yet**.

Ported and working in `src/`: `constants.py`, `config.py`, `utils/`, `data/`,
`scenarios/`, `analysis/scenarios.py`, `routing_tools/`,
`models/uncapacitated_saa_model.py`, `entrypoints/scenarios/`.

Still only in `OLD/src/`: the capacitated / flex / extended models, the powerset and
experiment drivers, `visualization/`, and `analysis/ca_factor_analysis.py`.

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

### Scenario pipeline (current, in `src/`)

```bash
python -m src.entrypoints.scenarios.build_panel            # raw events -> monthly panel
python -m src.entrypoints.scenarios.fit_params --n 50      # panel -> shape_params.json
python -m src.entrypoints.scenarios.generate --all --n 50  # params -> scenarios per regime
python -m src.entrypoints.scenarios.analyze                # validation report (exit != 0 on failure)
python -m src.entrypoints.scenarios.verify_end_to_end --n 3  # CA + Gurobi smoke test
```

`--n` must match between `fit_params` and `generate`: the regime multipliers are
calibrated against the exact seed sequence the generator consumes.

### Legacy code (still in `OLD/`)

Run from `OLD/`, because `constants.py` there derives `ROOT_DIR` from
`parents[1]`, so `DATA_DIR` resolves to `OLD/data`:

```bash
cd OLD && poetry run python -m src.entrypoints.run_powerset_experiment
```

Note `OLD/src/constants.py` points `PATH_DATA_PIXEL` at
`OLD/data/pixels/input_pixels.xlsx`, which does not exist — the only copy is
`data/raw_pixel/input_pixels.xlsx`, so `get_pixels()` fails there. The ported
`src/constants.py` already points at the right place.

### Lint

Line length is **130** (black, isort, pylint); flake8 is at 120.

```bash
poetry run black . && poetry run isort . && poetry run flake8
poetry run pre-commit run --all-files   # black, isort, flake8, detect-secrets
```

**There are no tests.** Verification is the `analyze` report plus
`verify_end_to_end`.

## Architecture

A one-way pipeline. Understanding it matters more than any individual file:

```
raw events  →  monthly panel  →  shape params  →  scenarios  →  CA  →  Gurobi LP  →  results  →  HTML
data/raw_*    data/scenarios/   shape_params    generated/    routing_  models/     results/   analysis/
              panel_monthly     .json                         tools/
```

The pivotal design decision: **routing costs are computed analytically before the
solve.** The Continuous Approximation turns pixel geometry (area, customer density,
drop size) into a cost and fleet size per
`(facility, pixel, vehicle, period, scenario)`, so the LP never enumerates routes —
it only chooses assignments over precomputed costs. A bug in the CA does not crash
anything; it silently shifts every downstream number.

**Two echelons:** the DC serves pixels directly with large vehicles
(`first_echelon_truck`), satellites serve pixels with small vehicles (`van`). Nine
candidate satellites, 161 pixels, 12 periods, La Paz.

**Configuration split:** `constants.py` holds paths, grid geometry and regime
targets; `config.py` holds the vehicle parameter dicts.

The scenario model itself is documented in
`data/scenarios/ESCENARIOS_DOCUMENTACION.md` — read it before touching
`src/scenarios/`.

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
  for `demand > 0` and `uncapacitated_saa_model.py` indexes them directly, so one
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
- **Check `model.Status` before recording an objective.** The models record one
  without inspecting it, so a time-limit incumbent looks like an optimum.
- Pylint permits single-capital names (`X`, `W`, `Y`, `N`, `Q`, `i`, `k`, `t`) so code
  can mirror the formulation's notation. Use the README's symbols.
- **What is versioned:** `.gitignore` no longer excludes `data/` wholesale. Fitted
  shape parameters, the scenario documentation and the small raw inputs are tracked;
  the 192 MB demand CSV, the derived panel, generated scenarios and `results/` are
  not.

## Subagents and skills

`.claude/agents/`: `model-formulation-reviewer` (Gurobi code vs. the README
formulation), `ca-auditor` (CA dimensional analysis), `results-analyst` (mining
result JSONs without flooding context), `refactor-migrator` (porting `OLD/src`
modules with numerical-equivalence proof).

`.claude/skills/viz-report/`: house style for the interactive HTML reports (Plotly
via CDN, method box + insight box per section, insights computed from the data).
