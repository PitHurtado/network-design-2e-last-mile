"""Shared domain: parameters, entities, raw-input readers and the scenario contract.

`src.scenarios` and `src.optimization` both depend on this package and never on each
other; they meet on disk, through `src.core.contract`. Checked by
`tests/test_architecture.py`.
"""
