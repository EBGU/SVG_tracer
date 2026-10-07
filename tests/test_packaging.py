"""Packaging: the split package and the documented console script must stay installable.

Added together with the ``pyproject.toml`` release-hygiene change, so a source checkout whose
tests pass also has a correct package/entry-point declaration.
"""
from __future__ import annotations

import re
import subprocess


def _pyproject(repo) -> str:
    return (repo / "pyproject.toml").read_text(encoding="utf-8")


def test_package_and_modules_declared(repo):
    text = _pyproject(repo)
    assert re.search(r'packages\s*=\s*\[\s*"svg_tracer"\s*\]', text), \
        "the svg_tracer package is not declared in [tool.setuptools]"
    for module in ("logo_trace", "gpu_backend", "cuda_backend", "svg_slim", "svgzip", "selfcheck"):
        assert re.search(r'"%s"' % module, text), f"{module} is no longer installed"


def test_console_script_and_runtime_dependencies_declared(repo):
    text = _pyproject(repo)
    assert re.search(r'svg-tracer\s*=\s*"logo_trace:main"', text), "the console script changed"
    deps = re.search(r"dependencies\s*=\s*\[(.*?)\]", text, re.S)
    assert deps, "no runtime dependencies declared"
    for dep in ("numpy", "scipy", "scikit-image", "Pillow"):
        assert dep.lower() in deps.group(1).lower(), f"{dep} is not declared"


def test_license_file_matches_pyproject(repo):
    text = _pyproject(repo)
    assert re.search(r'license\s*=\s*\{\s*text\s*=\s*"MIT"\s*\}', text), "license metadata changed"
    lic = repo / "LICENSE"
    assert lic.is_file(), "LICENSE is missing"
    body = lic.read_text(encoding="utf-8")
    assert body.startswith("MIT License")
    assert "WITHOUT WARRANTY OF ANY KIND" in body


def test_console_script_runs_when_installed(repo):
    """`svg-tracer --version` works when the project is installed (skipped otherwise)."""
    try:
        r = subprocess.run(["svg-tracer", "--version"], cwd=str(repo), timeout=120,
                           capture_output=True, text=True)
    except FileNotFoundError:
        import pytest
        pytest.skip("svg-tracer is not installed in this environment (pip install -e .)")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "logotrace 1.0.0"
