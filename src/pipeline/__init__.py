"""Preprocessing: raw delivery events -> monthly panel -> shape params -> scenarios.

Everything here runs before any solve. The handoff to `src.optimization` is on disk,
not in code: the generated scenario JSONs under `data/scenarios/generated/<regime>/`,
whose four-field contract is documented in `data/scenarios/ESCENARIOS_DOCUMENTACION.md`.
"""
