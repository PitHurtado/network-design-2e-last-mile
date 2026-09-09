"""Shared foundation: paths, parameters, domain entities, logging and raw-input readers.

Both `src.pipeline` and `src.optimization` depend on this package; neither depends on
the other. That direction is the whole point of the split, so it is worth checking:

    grep -rn "^from src\\.optimization" src/pipeline   # must print nothing
    grep -rn "^from src\\.pipeline"     src/optimization   # must print nothing
"""
