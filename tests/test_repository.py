"""Repository-level invariants: the shipped examples, the self-check and the program entry point.

The committed ``examples/*.svg`` are the project's published byte-identity reference, so they
are checked here without needing any input images (which keeps the guard alive in CI, where
``inputs/`` may be absent).
"""
from __future__ import annotations

import subprocess
import sys

import pytest


def test_committed_examples_are_byte_identical(repo, md5file, examples_md5):
    for rel, want in examples_md5.items():
        path = repo / rel
        assert path.is_file(), f"{rel} is missing from the repository"
        got = md5file(path)
        if want.startswith("__"):
            pytest.fail(
                f"{rel} has no recorded md5 (placeholder {want}); run "
                "tools/refresh_anchors.py after regenerating the examples")
        assert got == want, f"{rel} changed: expected {want}, got {got}"


def test_program_entry_point_is_thin(repo):
    """SVG_tracer.py must stay a launcher; the implementation lives in svg_tracer/."""
    text = (repo / "SVG_tracer.py").read_text(encoding="utf-8")
    assert "from svg_tracer.cli import main" in text
    assert len(text.splitlines()) < 80, "SVG_tracer.py grew back into a monolith"
    package = repo / "svg_tracer"
    for name in ("__init__.py", "__main__.py", "cli.py", "state.py", "geometry.py", "tensor.py",
                 "segment.py", "gradient.py", "refine.py", "contours.py", "strokes.py", "edges.py",
                 "svg_out.py", "shade.py", "gpu.py"):
        assert (package / name).is_file(), f"svg_tracer/{name} is missing"


def test_no_stale_historical_module_name(repo):
    """The program was renamed; nothing in the release surface should still refer to the old name.

    The name is reconstructed here from two pieces so that this file does not match its own search.
    """
    old = "logo" + "_trace"
    names = ("SVG_tracer.py", "pyproject.toml", "README.md", "selfcheck.py", "svgzip.py",
             ".github/workflows/ci.yml", "tools/refresh_anchors.py", "tools/measure_examples.py")
    for name in names:
        path = repo / name
        if not path.is_file():
            continue
        assert old not in path.read_text(encoding="utf-8"), \
            f"{name} still refers to the historical module name"


@pytest.mark.slow
def test_selfcheck_passes(repo):
    r = subprocess.run([sys.executable, "selfcheck.py"], cwd=str(repo), timeout=1800,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert r.stdout.rstrip().endswith("✓ self-check passed"), r.stdout[-1500:]
