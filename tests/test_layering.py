"""The packages depend on each other in one direction only."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "mingpu"

#: package or file under mingpu -> what its modules must not import when they run
FORBIDDEN = {
    "storage": ("mingpu.cli", "mingpu.planner", "mingpu.plotting"),
    "planner": ("mingpu.cli", "mingpu.plotting", "matplotlib"),
    "planner/define_search_space": (
        "mingpu.planner.execute_search",
        "mingpu.planner.workflow",
        "mingpu.planner.objectives.result",
        "mingpu.storage",
        "calculon",
        "duckdb",
    ),
    "planner/execute_search": ("mingpu.planner.objectives", "mingpu.planner.workflow"),
    "planner/objectives": (
        "mingpu.planner.execute_search",
        "mingpu.planner.define_search_space",
        "mingpu.planner.workflow",
        "calculon",
    ),
    "planner/objectives/rules.py": ("mingpu",),
}
#: file under mingpu -> what it may import in spite of FORBIDDEN
ALLOWED = {"planner/define_search_space/restrictions.py": ("calculon",)}


def _type_checking_nodes(tree: ast.Module) -> set[ast.AST]:
    """The nodes inside ``if TYPE_CHECKING:`` blocks, which do not run."""
    blocks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and getattr(node.test, "id", None) == "TYPE_CHECKING"
    ]
    return {inner for block in blocks for inner in ast.walk(block)}


def _runtime_imports(path: Path, root: Path = SRC) -> set[str]:
    """The absolute names a file imports when it runs."""
    package = ("mingpu", *path.relative_to(root).parts[:-1])
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skipped = _type_checking_nodes(tree)
    found = set()
    for node in ast.walk(tree):
        if node in skipped:
            continue
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                found.add(node.module)
                continue
            base = package[: len(package) - (node.level - 1)]
            if node.module:
                found.add(".".join((*base, node.module)))
            else:
                found.update(".".join((*base, alias.name)) for alias in node.names)
    return found


def _violations(path: Path, banned: tuple[str, ...], root: Path = SRC) -> list[str]:
    names = _runtime_imports(path, root)
    return [n for n in names for b in banned if n == b or n.startswith(f"{b}.")]


def _files(target: str) -> list[Path]:
    path = SRC / target
    return [path] if path.suffix else list(path.rglob("*.py"))


def test_packages_do_not_import_upwards(tmp_path):
    for target, banned in FORBIDDEN.items():
        for path in _files(target):
            allowed = ALLOWED.get(path.relative_to(SRC).as_posix(), ())
            bad = _violations(path, tuple(b for b in banned if b not in allowed))
            assert not bad, f"{path.relative_to(SRC)} imports {sorted(bad)}"

    # the guard itself resolves relative imports and skips TYPE_CHECKING blocks
    module = tmp_path / "planner" / "define_search_space" / "x.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "from ..execute_search.outer_search import a\nfrom ...storage import b\nimport calculon\n"
    )
    assert _runtime_imports(module, tmp_path) == {
        "mingpu.planner.execute_search.outer_search",
        "mingpu.storage",
        "calculon",
    }
    assert _violations(module, ("mingpu.planner.execute_search",), tmp_path)
    module.write_text("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import calculon\n")
    assert not _violations(module, ("calculon",), tmp_path)
