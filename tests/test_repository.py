"""Repository-level invariants: the shipped examples, the self-check and the split itself.

The committed ``examples/*.svg`` are the project's published byte-identity reference, so they
are checked here without needing any input images (which keeps the guard alive in CI, where
``inputs/`` is gitignored and therefore absent).
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
        assert got == want, f"{rel} changed: expected {want}, got {got}"


def test_shim_is_thin(repo):
    """logo_trace.py must stay a shim; the implementation lives in svg_tracer/."""
    text = (repo / "logo_trace.py").read_text(encoding="utf-8")
    assert "from svg_tracer import" in text
    assert len(text.splitlines()) < 80, "logo_trace.py grew back into a monolith"
    package = repo / "svg_tracer"
    for name in ("__init__.py", "cli.py", "state.py", "geometry.py", "tensor.py", "segment.py",
                 "gradient.py", "refine.py", "contours.py", "strokes.py", "edges.py",
                 "svg_out.py", "shade.py", "gpu.py"):
        assert (package / name).is_file(), f"svg_tracer/{name} is missing"


@pytest.mark.slow
def test_selfcheck_passes(repo):
    r = subprocess.run([sys.executable, "selfcheck.py"], cwd=str(repo), timeout=1800,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert r.stdout.rstrip().endswith("✓ 自检全部通过"), r.stdout[-1500:]
