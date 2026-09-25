"""Demand scenarios: raw delivery events -> monthly panel -> shape params -> scenario sets.

    fitting/     panel, marginals, regime multipliers: everything estimated from history
    generation/  the generator and the writers of versioned scenario sets
    spatial.py   pixel geometry and the spatial correlation model

Everything here runs before any solve. The handoff to `src.optimization` is on disk,
through `src.core.contract`; the four-field scenario contract is documented in
`data/scenarios/ESCENARIOS_DOCUMENTACION.md`.
"""
