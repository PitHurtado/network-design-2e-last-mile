"""Gurobi formulations: a base model plus toggleable blocks, one subclass per variant.

    UncapacitatedSAAModel  ⊂  CapacitatedSAAModel  ⊂  FlexSAAModel

See `base` for how a variant declares its blocks; `MODELS` maps every `NAME` to its class.
"""

from src.optimization.models.base import MODELS, BaseSAAModel, ModelFeatures
from src.optimization.models.capacitated import CapacitatedFeatures, CapacitatedSAAModel
from src.optimization.models.flex import FlexFeatures, FlexSAAModel
from src.optimization.models.policies import POLICIES, OperationPolicy, policy_for
from src.optimization.models.uncapacitated import UncapacitatedSAAModel


def model_class(name: str) -> type[BaseSAAModel]:
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; expected one of {sorted(MODELS)}.")
    return MODELS[name]


__all__ = [
    "MODELS",
    "BaseSAAModel",
    "ModelFeatures",
    "UncapacitatedSAAModel",
    "CapacitatedSAAModel",
    "CapacitatedFeatures",
    "FlexSAAModel",
    "FlexFeatures",
    "OperationPolicy",
    "POLICIES",
    "policy_for",
    "model_class",
]
