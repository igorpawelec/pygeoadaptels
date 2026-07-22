"""The version must be one number, not four that agree by hand.

GeoPalette shipped 0.5.0 and 0.6.0 with ``__version__`` still reading 0.4.0,
and rHRG's CITATION.cff sat two releases behind DESCRIPTION. Both times the
bump edited the places someone remembered instead of searching for the old
number, and nothing was checking. Nothing was checking here either.

What this compares needs care. ``plgeoadaptels.__version__`` is read from the
installed distribution's metadata, with a string literal as the fallback::

    try:
        __version__ = _version("plgeoadaptels")
    except Exception:
        __version__ = "0.8.0"

So the *runtime* value describes the environment, not this checkout — while
writing this, an editable install in the conda env reported 0.7.0 against a
source tree at 0.8.0, which is correct behaviour and would make a runtime
comparison fail for a reason that has nothing to do with the release. The
literal is the part that goes stale silently, and it is what anyone running
from an unbuilt checkout sees, so the literal is what is checked here.

Copyright (C) 2025 Igor Pawelec. Licence: GPLv3.
"""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / "plgeoadaptels" / "__init__.py"


def _pyproject_version():
    m = re.search(r'(?m)^version\s*=\s*"([^"]+)"',
                  (ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert m, "no version in pyproject.toml"
    return m.group(1)


def _literal_versions():
    """Every ``__version__ = "..."`` written out in the source."""
    return re.findall(r'__version__\s*=\s*"([^"]+)"',
                      PKG.read_text(encoding="utf-8"))


def test_fallback_literals_match_pyproject():
    literals = _literal_versions()
    assert literals, (
        f"no literal __version__ found in {PKG.name}. If the fallback was "
        f"removed, delete this test; if it was renamed, fix the pattern — "
        f"an empty result must not read as a pass."
    )
    expected = _pyproject_version()
    for got in literals:
        assert got == expected, (
            f"{PKG.name} falls back to {got}, pyproject.toml says {expected}. "
            f"That literal is what an unbuilt checkout reports."
        )


def test_citation_matches_pyproject():
    p = ROOT / "CITATION.cff"
    if not p.exists():
        pytest.skip("no CITATION.cff")
    m = re.search(r'(?m)^version:\s*"?([^"\n]+)"?\s*$', p.read_text(encoding="utf-8"))
    assert m, "no version in CITATION.cff"
    assert m.group(1).strip() == _pyproject_version(), (
        f"CITATION.cff says {m.group(1).strip()}, pyproject.toml says "
        f"{_pyproject_version()}. The CITATION is what a DOI cites."
    )


def test_changelog_documents_this_version():
    """A release with no changelog entry is a release nobody can read."""
    version = _pyproject_version()
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(rf"(?m)^#+\s*\[?{re.escape(version)}\]?", text), (
        f"CHANGELOG.md has no heading for {version}."
    )


def test_the_checks_can_fail():
    """Guards the three above: a pattern that never matches passes silently."""
    assert re.match(r"^\d+\.\d+\.\d+", _pyproject_version()), _pyproject_version()
    assert _pyproject_version() != "0.0.0"
