"""Experiment drivers: specs and payloads per experiment, over one `ExperimentRunner` and `ResultStore`.

    flexibility.py            regime × policy × case (expected / optimization / annual_expected)
    flexibility_evaluation.py fixed-Y recourse on the validation scenarios (out-of-sample VSS)
    validation_benchmark.py   RP on the validation scenarios (theoretical VSS)

Versioning (candidate runs, promotion, leaf reuse) is `src.optimization.stages`. Still only
in `OLD/src/entrypoints/`: the powerset sweeps and the best-per-size re-solves.
"""
