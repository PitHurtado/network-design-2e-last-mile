# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

Two-echelon (DC → satellites → pixels) last-mile network design under demand
uncertainty, solved with Gurobi over scenarios whose routing costs come from a
Continuous Approximation. Research code for a paper.

Branch `feature/refactor-and-cleaning`: the OOP refactor is done. `src/` is six packages
with a one-way dependency graph, every artifact is versioned (candidate → validate →
promote), and golden tests pin the numbers. Official versions today: params `p1`;
scenarios `v1` (value-identical to the pre-refactor `shape_params` v3 / scenarios `v3`)
and `v2` (= `v1` + the `capacity` set); satellite capacity table `f1` (from `v2`).
No official optimization run exists yet (`r1` is pending).

```
tools ← core ← { scenarios , optimization ← calibration }      visualization ← { tools, core }
```

- `tools/` — generic: JSON I/O + hashing, paths, logging, `Check`/`Validator`,
  `ArtifactStore`, `Manifest`, `Promoter`, argparse helpers. Knows nothing of the study.
- `core/` — shared domain: constants, vehicle configs, entities, raw-input readers
  (`inputs.py`), grid geometry (`grid.py`) and the on-disk scenario contract
  (`contract.py`: `ScenarioLayout`, `scenario_id`, `SCENARIO_SETS`, `DEPENDENCE_METHODS`).
- `scenarios/` — `fitting/` (panel, marginals, `RegimeCalibrator`, `ParamsFitter`),
  `generation/` (`ScenarioGenerator`, `DependenceStrategy` subclasses, seeds,
  `ScenarioSetWriter`), `validation/` (round-trip, contract checks, report metrics),
  `params.py` (`ShapeParams`), `reports.py` (builders), `stages.py`, `cli.py`.
- `optimization/` — `instance.py` (`InstanceSpec`, `InstanceBuilder`, `Instance`),
  `routing/` (CA), `models/` (block registry), `capacity/` (satellite capacity analysis:
  tariffs, `LevelMethod` strategies, `CapacityTable`), `experiments/` (`ExperimentRunner`,
  `ResultStore`, flexibility / evaluation / benchmark), `metrics/`, `reports.py`,
  `stages.py` (`FacilityStage`, `RunStage`), `verify.py`, `cli.py`.
- `calibration/` — SAA sample-size study: interfaces only, every method raises
  `NotImplementedError`.
- `visualization/` — `html.py` toolkit, `components/labels.py`, one renderer per report
  (`scenarios/*_report.py`, `results/*_report.py`): a dataclass + `render(data, path)`.

`scenarios` and `optimization` never import each other; they meet on disk through
`core/contract.py`. `tests/test_architecture.py` enforces the whole graph.

Still only in `OLD/src/`: the powerset and best-per-size drivers, `visualization/`
solution maps, `analysis/ca_factor_analysis.py`. `capacitated.py` and `flex.py` cover
`OLD`'s capacitated, capacitated-flex and extended models (levels and costs now come from
`input_facilities.xlsx`, not `OLD`'s hard-coded Proposal-A tables).

**`OLD/` is gitignored and untracked** — it exists only on this machine (plus an
`OLD.zip` beside it) and cannot be recovered from git. It is the reference
implementation for everything not yet ported. Never modify or delete it; copy out of it.

## Commands

Dependencies are managed with Poetry. A Gurobi license is required for any solve
(academic license, expires 2027-04-22). **The user runs `poetry lock` / `poetry install`
themselves — do not run them.** `poetry.lock` is stale relative to `pyproject.toml`
(python `^3.12`, pytest, `package-mode = true` with the `scenarios` / `optimize` /
`calibrate` scripts). Until it is reinstalled, use the module form:

```bash
poetry run python -m src.scenarios <subcommand>     # == `scenarios <subcommand>`
poetry run python -m src.optimization <subcommand>  # == `optimize <subcommand>`
poetry run python -m src.calibration <subcommand>   # == `calibrate <subcommand>`
```

### Scenarios

```bash
scenarios panel build                 # raw events -> data/interim/panel_monthly.csv (+ panel_source.json)
scenarios params fit --n 100          # -> cp-* candidate (fit + regime calibration on 100 validation streams)
scenarios params recalibrate --from p1
scenarios generate --params p1        # -> cv-*: optimization 30, validation 100, capacity 100, expected, annual_expected
scenarios compare --params p1 --regimes normal --n 100   # -> cc-* (exploratory, never promoted)
scenarios explore v1
scenarios validate <ref> | promote <ref> | list | show <ref>
```

### Optimization

```bash
optimize capacity analyze --scenarios v2 [--levels percentiles-a|percentiles-b|fixed-grid]   # -> cf-*
optimize flexibility --scenarios v2 --facilities f1 [--model flex|capacitated|uncapacitated] [--regimes ...] [--flexibilities ...] [--cases ...] [--threads 1 --seed 0]
optimize evaluate --run r1            # fixed-Y recourse on validation (with r1's facilities table)
optimize benchmark --scenarios v2 --facilities f1   # RP_100 for the theoretical VSS
optimize report <run|fN> [--benchmark <run>]
optimize verify --scenarios v2 --n 3  # CA + Gurobi smoke test
optimize validate <ref> | promote <ref> | list | show <ref>    # ref: cf-*/fN or cr-*/rN
```

### Tests

```bash
poetry run python -m unittest discover -s tests -t .      # ~1 min; or pytest once installed
GOLDEN_SKIP_SOLVE=1 poetry run python -m unittest discover -s tests -t .   # no Gurobi
poetry run python -m tests.golden.capture <name>          # re-capture ONE golden, deliberately
```

- `tests/golden/`: G1 fit, G2 recalibrate, G3 generate, G4 comparison, G5 spatial,
  G6 report figures + visible text + CSVs, G7 CA costs/fleet, G8 solves (Threads=1,
  Seed=0, WorkLimit — deterministic), G9 capacity analysis. Producers call the API;
  `expected/` pins outputs.
  Re-capturing is a deliberate act: say so in the commit, with the before/after.
- Fixtures: `tests/golden/fixtures/shape_params.json` is tracked; the panel copy and
  `fixtures/results/` (archived v3 runs, for the report goldens) are local only.
- `test_lifecycle.py` (promotion conditions, byte-level reproduction) and
  `test_architecture.py` (the dependency graph).

### Lint

Line length is **130** everywhere: black, isort, pylint and flake8. `flake8` is not a dev
dependency; use pre-commit (`poetry run pre-commit run --all-files`) or its cached venv.
The remaining E501 are long HTML strings in the renderers.

### Legacy code (still in `OLD/`)

Run from `OLD/` (its `constants.py` derives `ROOT_DIR` from `parents[1]`):
`cd OLD && poetry run python -m src.entrypoints.run_powerset_experiment`.
`OLD/src/constants.py` points `PATH_DATA_PIXEL` at a file that does not exist; the only
copy is `data/raw_pixel/input_pixels.xlsx`.

## Versioning

Every stage produces an artifact; every command writes a **candidate** into a sandbox;
only `promote` creates an official, immutable version.

```
data/params/p<N>/         shape_params.json, panel_monthly.csv (copy, gitignored), manifest.json, validation.json
data/scenarios/v<N>/      <regime>/<set>/{scenario_*.json (gitignored), manifest.json}, manifest.json, validation.json, reports/
data/facilities/f<N>/     capacity.json, peak_fleet.csv, assignment.json, manifest.json, validation.json, reports/
results/runs/r<N>/        <experiment>/<v>/<flex>/<regime>[/<case>]/<leaf>.json, manifest.json, validation.json, reports/
data/sandbox/{params,scenarios,facilities,comparisons}/  results/sandbox/runs/   candidates: cp-/cv-/cf-/cc-/cr-<timestamp>
data/_archive/2026-09-25/  results/_archive/2026-09-25/                everything before versioning (v2, v3, vcompare*)
```

- `manifest.json`: command + resolved config, sha256 of every raw input, parents (id +
  digests), seeds, git commit/dirty, library versions, content digest, stage details.
- Content digest excludes `manifest.json`, `validation.json` and `reports/`, so reports
  can be regenerated inside an official version.
- `promote` refuses unless: validated and passed; content unchanged since; parents
  official and unchanged; produced from a clean tree at the current HEAD and the tree is
  still clean; and (params, scenarios, facilities) re-executing the command is
  byte-identical. Committing anything after generating a candidate moves HEAD, so the
  candidate must be regenerated before it can be promoted.
  Runs are not re-executed (time-limited MIPs are not deterministic); a leaf whose key
  (scenarios + model + solver + policy + regime + case) exists in an official run is
  copied instead of re-solved.
- Only the identity of official versions is in git (manifests, validation, params).

## Architecture

```
raw events → panel → params (pN) → scenarios (vN) → CA → Gurobi → runs (rN) → reports
└──────────── src/scenarios ─────────────┘   └──────────── src/optimization ───────────┘
```

The pivotal design decision: **routing costs are computed analytically before the
solve.** The Continuous Approximation turns pixel geometry (area, customer density,
drop size) into a cost and fleet size per `(facility, pixel, vehicle, period, scenario)`,
so the LP never enumerates routes. A bug in the CA does not crash anything; it silently
shifts every downstream number.

**Two echelons:** the DC serves pixels directly with large vehicles
(`first_echelon_truck`), satellites serve pixels with small vehicles (`van`). Nine
candidate satellites, 161 pixels, 12 periods, La Paz.

**Model family:** `UncapacitatedSAAModel ⊂ CapacitatedSAAModel ⊂ FlexSAAModel`, each a
subclass of the previous that only *adds* blocks (base + toggleable blocks + preset
subclasses, decided 2026-09-09). A variant declares `NAME` (registers it in `MODELS`,
selectable with `--model`), a `Features` dataclass extending its parent's, and `BLOCKS`:
only the blocks it introduces; `__init_subclass__` builds `CATALOGUE`, and `before=`
pins a block's position because **the variable/constraint order Gurobi sees changes
results under a work or time limit** (`tests/test_models.py` pins flex's order). To
change a block's formulation, override its method (flex overrides `_installed_capacity`
to put the capacity on Z); redeclaring a block raises. Operating policies are
`OperationPolicy` subclasses (`policies.py`), registered by name. Averaged objective
blocks return `{scenario: expr}`, so the objective, `scenario_costs()` and `decisions()`
all read the same expressions. Fixing Y for an evaluation is its own block
(`fix_installation`, enabled by `fixed_installation=`). Ablation is `disabled_blocks`
(names are unique; the objective block is `operation_cost`, the variables `operation`);
unknown names raise. When porting a model from `OLD/`, implement its blocks — do not copy
the build/solve scaffolding.

**Scenario generation:** `ScenarioGenerator.draw` multiplies expected demand by the
shocks of a `DependenceStrategy` (`SpatialJoint`, `Independent`, `HistoricalBootstrap`,
`MeanShocks`, `MedianShocks`). The order in which a strategy consumes the RNG is part of
the contract (documented in `dependence.py`): scenarios regenerate bit-for-bit from seeds.

The scenario model is documented in `data/scenarios/ESCENARIOS_DOCUMENTACION.md` — read
it before touching `src/scenarios/`.

## Conventions and traps

- **Two demand scales, and mixing them inflates costs ~3×.** *Model demand* is
  `Σ (stop × drop)` — one representative delivery round, ~45k/period in 2022. *Raw
  items* is the monthly count, ~149k/month. Regime targets and the contract are in
  model demand.
- **The scenario contract is four fields** per pixel: `id_pixel`, `stop`, `drop`,
  `demand`, arrays of length 12. `id_scenario` and `type` are written but never read.
  `k`, `lon`, `lat`, `area_surface` come from `input_pixels.xlsx`.
- **Scenario ids carry no version** (`normal-optimization-001`): the version is the
  directory. Promotion renames a directory; it never rewrites a file.
- **`stop >= 1` and `drop > 0` are hard invariants.** The CA only writes cost keys for
  `demand > 0` and the models index them directly (not `.get(key, 0)`), so one zero
  pixel-period is a `KeyError` at solve time; `drop` and `density` are divisors.
- **Satellite levels and costs come only from a facilities table `fN`.** Capacitated
  models (`USES_CAPACITY`) refuse to run without `--facilities`; `input_facilities.xlsx`
  only supplies location and sourcing cost (its 0..12 grid and costs, identical for all 9
  satellites and ~5× below the tariff table, are not used). `InstanceBuilder` without a
  table keeps the Excel values — only the uncapacitated model and the goldens rely on it.
  Levels: `percentiles-a` of the peak fleet on the `capacity` set, regimes pooled
  (a user decision: one table for every regime). Costs: `data/raw_facility/tariffs.json`,
  operating cost `OPEX(q) × [0.70 + 0.30 × seasonal factor]`.
- **The pixel set must equal `input_pixels.xlsx` exactly.** A pixel missing from the grid
  is dropped with a warning.
- **12 periods, fixed.** `N_PERIODS`; `InstanceSpec` accepts only 12 or the one-period
  `annual_expected` set, whose variable costs are scaled by `horizon_weight = 12`.
- **Regime calibration uses the validation streams and the persisted generator**
  (`ScenarioGenerator.from_params`), so validation sets hit their targets exactly.
- **Result JSON keys**: decisions are lists of row dicts. Where tuple-keyed dicts appear
  (legacy `OLD/` results) the keys are stringified tuples: parse with `ast.literal_eval`.
- **Cost components**: `result.json` components are scaled like the objective
  (`horizon_weight / N`), so they sum to `objective`. The archived v3 runs predate that
  fix: their `annual_expected` components are 1/12 of the true value.
- **`evaluation.json`** records `source_run`, `source_result` (relative to that run) and
  `source_solve`; `evaluate_one` checks that installation + mean second-stage cost equals
  the objective.
- **CA expressions carry `# [unit]` comments** — keep them correct. Some are
  inconsistent (`intra_tour_time_per_customer`). Dimensional errors are the dominant
  failure mode there.
- **`area_surface` is an area in km²** (count of merged ~1 km² cells).
- **Neighbour rings are cumulative**: `pixel_neighbor_pairs(ring=k)` is graph distance ≤ k.
- Pylint permits single-capital names (`X`, `W`, `Y`, `N`, `Q`, `i`, `k`, `t`) so code
  can mirror the formulation's notation.

## Subagents and skills

`.claude/agents/`: `model-formulation-reviewer` (Gurobi code vs. the README formulation,
`src/optimization/models/`), `ca-auditor` (CA dimensional analysis,
`src/optimization/routing/`), `results-analyst` (mining run artifacts without flooding
context), `refactor-migrator` (porting `OLD/src` modules with numerical-equivalence proof).

`.claude/skills/viz-report/`: house style for the HTML reports (Plotly via CDN, method
box + insight box per section, insights computed from the data) and the
renderer / builder split of `src/visualization/`.
