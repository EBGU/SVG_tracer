"""Determinism: the same command on the same input must give the same bytes, every time.

Byte-identical output is the project's headline promise, and the pipeline is parallel (``--jobs``
fork workers, per-region seeds for the stroke layer, a GPU path that must agree with the CPU
path).  These checks rerun one small command and compare md5s; the full-size variants are the
byte-identity anchors in ``test_anchors.py``.
"""
from __future__ import annotations

import pytest

DETERMINISM_CASES = [
    ("serial", "water_lilies.jpg", ["--preset", "painting", "--scale", "1", "--jobs", "1"]),
    ("parallel", "water_lilies.jpg", ["--preset", "painting", "--scale", "1", "--jobs", "8"]),
    ("logo_gpu", "apple.png", ["--preset", "logo", "--scale", "2", "--jobs", "4",
                               "--auto-gradient"]),
]


@pytest.mark.parametrize("name,src,extra", DETERMINISM_CASES, ids=[c[0] for c in DETERMINISM_CASES])
def test_same_command_twice_same_md5(trace, small_image, md5file, tmp_path, name, src, extra):
    img = small_image(src)
    outs = []
    for i in (1, 2):
        out = tmp_path / f"{name}_{i}.svg"
        r = trace(["--in", img, "--gpu", "auto", "--no-preview", *extra], out)
        assert r.returncode == 0, r.stderr[-3000:]
        outs.append(out)
    assert outs[0].stat().st_size > 512
    assert md5file(outs[0]) == md5file(outs[1])
