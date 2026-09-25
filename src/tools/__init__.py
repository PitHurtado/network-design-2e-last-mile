"""Generic utilities with no knowledge of the study: JSON I/O, hashing, paths, logging, CLI helpers.

The bottom of the dependency graph: `src.tools` imports nothing else from `src`, and every
other package may import it.
"""
