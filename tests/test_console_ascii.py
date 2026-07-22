"""User-facing messages must encode on the console they print to.

plGeoAdaptels runs on Polish Windows, where the default console encoding is
cp1250, not UTF-8. A string sent to ``print``, ``sys.std*.write``,
``warnings.warn``, ``logging`` or ``raise`` is encoded with that codec on
the way out; a character it cannot represent -- ``->`` written as U+2192,
``x`` as U+00D7, ``>=`` as U+2265 -- raises UnicodeEncodeError at the moment
of printing. Because the CLI catches ValueError (which UnicodeEncodeError
subclasses) and prints ``str(e)``, the crash was even reclassified as a
"user error" with a cryptic charmap message.

This is a static check over the source rather than a run, so it covers every
message, not only the ones some other test happens to exercise. It flags a
string literal only when it is an argument to one of those sinks or the
argument of a ``raise`` -- non-ASCII in docstrings and comments never reaches
the console and is fine.

The audit that added this found the bug in three of the four sink kinds
across the family: the block that reads plgeoadaptels.__init__ shows the same style
elsewhere, so it is worth keeping the whole package honest.

Copyright (C) 2025 Igor Pawelec. Licence: GPLv3.
"""

import ast
import pathlib
import unittest

PKG = pathlib.Path(__file__).resolve().parent.parent / "plgeoadaptels"
SOURCES = sorted(PKG.rglob("*.py"))

# Names that put their string argument on a byte stream encoded with the
# console codec. `write` covers sys.stdout/sys.stderr.write; the logging
# level names cover logger.info(...) and friends.
SINKS = {"print", "write", "warn", "warning", "error", "info", "debug",
         "critical", "exception"}


def _offending(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    out = []
    for node in ast.walk(tree):
        ctx = None
        if isinstance(node, ast.Call):
            fn = node.func
            name = (fn.id if isinstance(fn, ast.Name)
                    else fn.attr if isinstance(fn, ast.Attribute) else "")
            if name in SINKS:
                ctx = name
        elif isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            ctx = "raise"
        if ctx is None:
            continue
        for s in ast.walk(node):
            if isinstance(s, ast.Constant) and isinstance(s.value, str):
                bad = sorted({hex(ord(c)) for c in s.value if ord(c) > 127})
                if bad:
                    out.append((s.lineno, ctx, bad, s.value[:50]))
    return out


class TestConsoleAscii(unittest.TestCase):

    def test_sources_found(self):
        # Without this an empty glob would make the check below pass over
        # nothing.
        self.assertGreaterEqual(len(SOURCES), 3, [p.name for p in SOURCES])

    def test_no_non_ascii_in_user_facing_strings(self):
        problems = []
        for path in SOURCES:
            for lineno, ctx, bad, text in _offending(path):
                problems.append(f"{path.name}:{lineno} [{ctx}] {bad} {text!r}")
        self.assertEqual(problems, [], "non-ASCII in a message that gets "
                         "printed or raised; on a cp1250 console this is a "
                         "UnicodeEncodeError, not decoration:\n  " +
                         "\n  ".join(problems))


if __name__ == "__main__":
    unittest.main()
