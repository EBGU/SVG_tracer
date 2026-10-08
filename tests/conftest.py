"""Shared fixtures and helpers for the SVG_tracer test suite.

The suite has two tiers:

* the fast tier (default) runs the documented command lines on small 96 px thumbnails of the
  shipped inputs -- or on a synthetic image when ``inputs/`` is absent -- and checks the CLI
  contract, well-formed XML output and byte-level determinism;
* the ``slow`` tier (``-m slow``) reproduces the documented example commands at full size and
  compares the output against the recorded md5 anchors.  It needs the images under ``inputs/`` and
  skips loudly when they are missing (e.g. in a pruned checkout).

Every test writes its artifacts into pytest's temporary directory, never into the repository.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SVG_TRACER = REPO / "SVG_tracer.py"
INPUTS = REPO / "inputs"

# The tool is imported in-process by the API tests; make the checkout importable the
# same way selfcheck.py does.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Flags shared by every documented example run (see README "Examples").
COMMON = ["--gpu", "auto", "--jobs", "64", "--no-preview"]

# md5 anchors.  Each entry is (id, md5, argv, required inputs).
# ``tools/refresh_anchors.py`` rewrites these digests (and the EXAMPLES_MD5 map below) whenever the
# shipped examples are deliberately regenerated; `--check` verifies them without writing.
ANCHORS = [
    dict(
        name="openai_scale1",
        md5="857bed7ba8d6637c27731a8f068a51a8",
        args=["--in", "inputs/openai.png", "--preset", "logo", "--scale", "1", *COMMON],
        inputs=["openai.png"],
    ),
    dict(
        name="water_lilies_scale1",
        md5="36c561471c0ad7c4c7f6da0e3cc72cdc",
        args=["--in", "inputs/water_lilies.jpg", "--preset", "painting", "--scale", "1", *COMMON],
        inputs=["water_lilies.jpg"],
    ),
    dict(
        name="wave_scale1",
        md5="ba28414137b129a62ce4661d743d0109",
        args=["--in", "inputs/wave.jpg", "--preset", "painting", "--scale", "1", *COMMON],
        inputs=["wave.jpg"],
    ),
    dict(
        name="apple_scale4_auto",
        md5="bad941615655185c1e0a1d04ae7311bc",
        args=["--in", "inputs/apple.png", "--preset", "logo", "--scale", "4",
              *COMMON, "--auto-gradient"],
        inputs=["apple.png"],
    ),
]

# The md5 of every example SVG that is committed to the repository.
# Kept in sync by tools/refresh_anchors.py (never hand-edit the digests).
EXAMPLES_MD5 = {
    "examples/apple_traced_scale1.svg": "4e3ac97c7dbbba5c953d32fc8a4e3ea5",
    "examples/apple_traced_scale4.svg": "bad941615655185c1e0a1d04ae7311bc",
    "examples/openai_traced_scale1.svg": "857bed7ba8d6637c27731a8f068a51a8",
    "examples/water_lilies_traced_scale1.svg": "36c561471c0ad7c4c7f6da0e3cc72cdc",
    "examples/wave_traced_scale1.svg": "ba28414137b129a62ce4661d743d0109",
}

THUMB_PX = 96


def md5(path) -> str:
    """Streaming md5 of a file."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pytest_generate_tests(metafunc):
    """Parametrize every test that asks for the ``anchor`` fixture over ANCHORS."""
    if "anchor" in metafunc.fixturenames:
        metafunc.parametrize("anchor", ANCHORS, ids=[a["name"] for a in ANCHORS])


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: full-size byte-identity anchors that need the inputs/ images")


def run_trace(args, out, timeout=3600, env_extra=None, cwd=None):
    """Run ``python SVG_tracer.py <args> --out <out>`` and return the CompletedProcess."""
    cmd = [sys.executable, str(SVG_TRACER), *map(str, args), "--out", str(out)]
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(cmd, cwd=str(cwd or REPO), env=env, timeout=timeout,
                          capture_output=True, text=True)


@pytest.fixture
def trace():
    """The :func:`run_trace` helper as a fixture."""
    return run_trace


@pytest.fixture
def md5file():
    """The :func:`md5` helper as a fixture."""
    return md5


@pytest.fixture(scope="session")
def repo():
    """The repository root (the directory that holds SVG_tracer.py)."""
    return REPO


@pytest.fixture(scope="session")
def examples_md5():
    """md5 of every example SVG committed to the repository."""
    return dict(EXAMPLES_MD5)


def _crop_or_synth(src: Path, dst: Path, size: int = THUMB_PX) -> Path:
    """A small whole-image thumbnail of ``src`` (or a synthetic stand-in when it is missing).

    A thumbnail rather than a tile crop: a 96 px crop of a flat logo can be a single colour, which
    exercises only the degenerate one-region path (see tests/test_degenerate.py) instead of the
    multi-region behaviour these smoke and determinism tests are meant to cover.
    """
    from PIL import Image
    if src.is_file():
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = size / max(w, h)
            if scale < 1.0:
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                               Image.LANCZOS)
            im.save(dst)
        return dst
    return synth_image(dst, size)


def synth_image(dst: Path, size: int = THUMB_PX) -> Path:
    """A small hard-edged figure with one smooth ramp, so every layer has something to do."""
    import numpy as np
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (size, size), (250, 249, 246))
    d = ImageDraw.Draw(im)
    d.ellipse([size * 0.08, size * 0.08, size * 0.60, size * 0.60], fill=(226, 92, 28))
    d.line([0, size - 1, size - 1, 0], fill=(32, 32, 40), width=max(2, size // 32))
    d.rectangle([size * 0.62, size * 0.62, size * 0.95, size * 0.94], fill=(18, 58, 200))
    a = np.asarray(im).astype(np.float32)
    y0, y1 = int(size * 0.62), int(size * 0.94)
    x0, x1 = int(size * 0.62), int(size * 0.95)
    t = np.linspace(0.0, 1.0, x1 - x0, dtype=np.float32)[None, :, None]
    a[y0:y1, x0:x1] = (1 - t) * np.array([18, 58, 200], np.float32) + t * np.array([40, 190, 220], np.float32)
    Image.fromarray(a.astype(np.uint8)).save(dst)
    return dst


@pytest.fixture(scope="session")
def small_image(tmp_path_factory):
    """Factory: a 96 px thumbnail of a shipped input, or a synthetic image when it is absent."""
    cache = tmp_path_factory.mktemp("crops")

    def make(name: str) -> Path:
        return _crop_or_synth(INPUTS / name, cache / name)

    return make


@pytest.fixture
def require_inputs():
    """Skip loudly when the byte-identity anchors cannot run because inputs/ is missing."""
    def check(anchor):
        missing = [n for n in anchor["inputs"] if not (INPUTS / n).is_file()]
        if missing:
            pytest.skip(
                "inputs/ is missing -- byte-identity anchor %r NOT checked; "
                "restore %s and re-run with '-m slow' to verify byte-for-byte output"
                % (anchor["name"], ", ".join("inputs/" + m for m in missing)))
    return check
