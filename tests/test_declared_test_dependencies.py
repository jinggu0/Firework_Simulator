"""Optional-looking test dependencies must still be declared.

`pytest.importorskip` is the right tool for a dependency that genuinely may be
absent, but it turns an undeclared one into a silent skip: the suite reports
green while the checks it gates never run. That has now happened twice here —
pyproj was declared but uninstalled, and jsonschema was installed only because
unrelated packages happened to pull it in.

So the rule is not "never use importorskip". It is that whatever it guards must
appear in a requirements file, so a documented `pip install -r` produces an
environment where every test actually executes.
"""

from __future__ import annotations

import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
TESTS = REPOSITORY_ROOT / "tests"
IMPORT_OR_SKIP = re.compile(r"""importorskip\(\s*["']([\w.]+)["']""")

#: Import name to distribution name. Anything guarded by `importorskip` has to
#: be listed here; an unmapped module fails the test rather than passing
#: vacuously, which is what would let the next undeclared dependency through.
MODULE_DISTRIBUTIONS = {
    "jsonschema": "jsonschema",
    "pyproj": "pyproj",
    "shapefile": "pyshp",
}


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_distributions() -> set[str]:
    declared: set[str] = set()
    for requirements in REPOSITORY_ROOT.glob("requirements*.txt"):
        for line in requirements.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            # `-r other.txt` includes are followed by the glob above, so the
            # transitive case needs no special handling here.
            if not line or line.startswith("-"):
                continue
            declared.add(_normalise(re.split(r"[<>=!~\[;]", line, maxsplit=1)[0]))
    return declared


def _guarded_modules() -> dict[str, set[str]]:
    guarded: dict[str, set[str]] = {}
    for path in sorted(TESTS.glob("test_*.py")):
        for module in IMPORT_OR_SKIP.findall(path.read_text(encoding="utf-8")):
            guarded.setdefault(module, set()).add(path.name)
    return guarded


def test_the_scan_finds_the_known_guards() -> None:
    # Guards the regex itself: if it silently stopped matching, every other
    # assertion here would pass over an empty set.
    guarded = _guarded_modules()

    assert "jsonschema" in guarded
    assert "pyproj" in guarded
    assert len(guarded) >= 3


def test_every_import_or_skip_module_is_mapped() -> None:
    unmapped = sorted(set(_guarded_modules()) - set(MODULE_DISTRIBUTIONS))

    assert not unmapped, (
        "these modules are guarded by importorskip but have no distribution "
        f"mapping, so nothing checks that they are declared: {unmapped}"
    )


def test_every_import_or_skip_dependency_is_declared() -> None:
    declared = _declared_distributions()
    undeclared = {
        module: sorted(files)
        for module, files in _guarded_modules().items()
        if _normalise(MODULE_DISTRIBUTIONS[module]) not in declared
    }

    assert not undeclared, (
        "these test dependencies are not declared in any requirements file, so "
        f"a clean environment skips the tests that need them: {undeclared}"
    )


def test_requirements_parsing_ignores_includes_and_specifiers() -> None:
    declared = _declared_distributions()

    assert "pytest" in declared
    assert "pyshp" in declared
    assert "numpy" in declared
    # `-r requirements.txt` lines must not be read as a package called "-r".
    assert not any(name.startswith("-") for name in declared)
