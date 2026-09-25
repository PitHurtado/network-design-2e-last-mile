"""The package dependency graph, enforced.

    tools <- core <- {scenarios, optimization <- calibration};  visualization <- {tools, core}

`scenarios` and `optimization` never import each other: they meet on disk, through
`src.core.contract`. `scenarios` / `optimization` may import `visualization` to render.
"""

import ast
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
ALLOWED = {
    "tools": set(),
    "core": {"tools"},
    "visualization": {"tools", "core"},
    "scenarios": {"tools", "core", "visualization"},
    "optimization": {"tools", "core", "visualization"},
    "calibration": {"tools", "core", "optimization"},
}


def imported_packages(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return {name.split(".")[1] for name in names if name.startswith("src.") and len(name.split(".")) > 1}


class ArchitectureTests(unittest.TestCase):
    def test_every_package_is_known(self):
        packages = {p.name for p in SRC.iterdir() if p.is_dir() and not p.name.startswith("__")}
        self.assertEqual(packages - set(ALLOWED), set())

    def test_imports_follow_the_graph(self):
        violations = []
        for package, allowed in ALLOWED.items():
            for path in (SRC / package).rglob("*.py"):
                for imported in imported_packages(path) - allowed - {package}:
                    violations.append(f"{path.relative_to(SRC)} imports src.{imported}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
