"""The package claims requires-python >= 3.9. Check that it means it.

Ported from pyHRG, which shipped 0.5.0 with ``max_iters: int | None`` and did
not import at all on 3.9: PEP 604 in an annotation is a runtime expression
before 3.10. It was invisible there because development runs on 3.12, and CI
caught it only after the tag. plGeoAdaptels is clean today — this is what keeps
it clean, and it is not idle: the package is one ``from __future__ import annotations`` away from the
same class of failure the moment anyone writes a modern-looking annotation,
and nothing else in the suite would say so.

Copyright (C) 2025 Igor Pawelec. Licence: GPLv3.
"""

import ast
import pathlib
import unittest

PKG = pathlib.Path(__file__).resolve().parent.parent / "plgeoadaptels"
SOURCES = sorted(PKG.rglob("*.py"))


def _annotations(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.arg, ast.AnnAssign)) and node.annotation:
            yield node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns:
            yield node.returns


class TestPython39Compat(unittest.TestCase):

    def test_sources_found(self):
        # Guards the whole file: a bad glob would make every test below pass
        # over an empty list.
        self.assertGreaterEqual(len(SOURCES), 3, [p.name for p in SOURCES])
        self.assertIn("__init__.py", [p.name for p in SOURCES])

    def test_no_pep604_without_future_import(self):
        """``X | Y`` in an annotation is a TypeError on 3.9."""
        for path in SOURCES:
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src, str(path))
            deferred = any(
                isinstance(n, ast.ImportFrom) and n.module == "__future__"
                and any(a.name == "annotations" for a in n.names)
                for n in tree.body
            )
            if deferred:
                continue
            for ann in _annotations(tree):
                for node in ast.walk(ann):
                    self.assertNotIsInstance(
                        node.op if isinstance(node, ast.BinOp) else None,
                        ast.BitOr,
                        f"{path.name}:{getattr(node, 'lineno', '?')} uses PEP 604 "
                        f"`X | Y` in an annotation. That is evaluated at runtime "
                        f"before 3.10. Use typing.Optional/Union, or put "
                        f"`from __future__ import annotations` first in the file.",
                    )

    def test_future_imports_are_compilable(self):
        """``from __future__`` must precede every statement but the docstring.

        ast.parse does *not* enforce this -- the first attempt at the fix
        above placed the import after three ``__author__`` assignments, the
        AST scan reported it as clean, and the module failed to import on
        every version of Python. compile() is what enforces it.
        """
        for path in SOURCES:
            src = path.read_text(encoding="utf-8")
            try:
                compile(src, str(path), "exec")
            except SyntaxError as e:
                self.fail(f"{path.name}:{e.lineno}: {e.msg}")


if __name__ == "__main__":
    unittest.main()
