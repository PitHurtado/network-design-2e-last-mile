"""Uncapacitated SAA model: distribution cost only.

The base formulation with every optional block off — no installation decision, no
capacity, no operating level. It is the lower bound of the family and the model the
scenario contract is smoke-tested against, so it stays the thinnest possible subclass.
"""

from src.optimization.models.base import BaseSAAModel, ModelFeatures


class UncapacitatedSAAModel(BaseSAAModel):
    """Two-echelon assignment over precomputed CA routing costs, no facility decisions."""

    NAME = "uncapacitated"
    DEFAULT_FEATURES = ModelFeatures()
