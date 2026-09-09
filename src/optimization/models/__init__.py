"""Gurobi formulations, as a base model plus toggleable blocks.

`base.BaseSAAModel` owns everything the variants share; each variant is a thin
subclass that declares which optional blocks it enables. See `base` for the
block catalogue.
"""
