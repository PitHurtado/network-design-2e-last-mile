"""Random streams of the simulated scenario sets.

Streams are set-specific but regime-independent, so scenario `i` of `low`, `normal` and
`high` shares its shocks and the regimes are directly comparable. The same property
pairs the dependence methods of a comparison.
"""

import numpy as np

SET_CODES = {"optimization": 30, "validation": 100}
SEED_SCHEME = "SeedSequence([seed_base, set_code]).spawn(index); set_code optimization=30, validation=100"


def set_seeds(seed_base: int, scenario_set: str, n_scenarios: int) -> list[np.random.SeedSequence]:
    """One child seed per scenario of a simulated set."""
    return np.random.SeedSequence([seed_base, SET_CODES[scenario_set]]).spawn(n_scenarios)


def rng_for(seed: np.random.SeedSequence) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))
