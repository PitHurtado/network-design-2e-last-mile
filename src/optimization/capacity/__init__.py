"""Satellite capacity analysis: from a scenario version to the capacity table the models read.

tariffs.py   versioned tariff table (OPEX and installation per level, extrapolation)
levels.py    how capacity levels are chosen from the peak-fleet distribution (strategies)
table.py     `CapacityTable`: levels and costs per satellite, the `fN` artifact's content
analysis.py  nearest-satellite assignment -> CA fleet -> peak fleet -> levels + costs
"""
