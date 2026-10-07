"""CLI smoke tests: every documented command line still parses and produces valid output.

These run on a 96 px thumbnail (or a synthetic stand-in) so the whole module costs a few
seconds, while still driving every preset, route and output mode that the README documents.
The full-size byte-identity checks live in ``test_anchors.py``.
"""
from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET

import pytest

# (id, input, extra argv) -- the four example commands from README "Examples".
EXAMPLE_COMMANDS = [
    ("apple", "apple.png", ["--preset", "logo", "--scale", "4", "--auto-gradient"]),
    ("openai", "openai.png", ["--preset", "logo", "--scale", "1"]),
    ("water_lilies", "water_lilies.jpg", ["--preset", "painting", "--scale", "1"]),
    ("wave", "wave.jpg", ["--preset", "painting", "--scale", "1"]),
]

# Other command lines documented in README "Common recipes" / "Presets".
DOCUMENTED_COMMANDS = [
    ("no_strokes_gzip", ["--preset", "logo", "--no-strokes", "--gzip"]),
    ("colors_baseline", ["--method", "colors", "--fit", "cr", "--snap-mode", "off"]),
    ("edges_baseline", ["--method", "edges", "--fit", "bezier"]),
    ("watershed_route", ["--method", "watershed", "--scale", "2"]),
    ("seg_flat", ["--preset", "logo", "--seg", "flat"]),
    ("compress_slim", ["--preset", "logo", "--compress", "slim"]),
    ("edge_ridges", ["--preset", "logo", "--edge-mode", "energy"]),
    ("legacy_off_paths", ["--preset", "logo", "--no-grad-radial", "--no-merge-grad"]),
    ("detail_layer", ["--preset", "painting", "--detail-chroma", "2", "--no-strokes",
                      "--no-adaptive-refine"]),
    ("labels_cache", ["--preset", "logo", "--scale", "2"]),
]


def _check_svg(path):
    assert path.is_file() and path.stat().st_size > 512, f"{path} missing or suspiciously small"
    root = ET.parse(path).getroot()          # raises on malformed XML
    assert root.tag.endswith("svg")
    text = path.read_text(encoding="utf-8")
    assert 'id="regions"' in text and "<defs>" in text
    return root


@pytest.mark.parametrize("name,src,extra", EXAMPLE_COMMANDS, ids=[c[0] for c in EXAMPLE_COMMANDS])
def test_example_command_smoke(trace, small_image, tmp_path, name, src, extra):
    img = small_image(src)
    out = tmp_path / f"{name}.svg"
    r = trace(["--in", img, "--gpu", "auto", "--jobs", "8", "--no-preview", *extra], out)
    assert r.returncode == 0, r.stderr[-3000:]
    _check_svg(out)


@pytest.mark.parametrize("name,extra", DOCUMENTED_COMMANDS,
                         ids=[c[0] for c in DOCUMENTED_COMMANDS])
def test_documented_flags_smoke(trace, small_image, tmp_path, name, extra):
    img = small_image("apple.png")
    out = tmp_path / f"{name}.svg"
    args = ["--in", img, "--gpu", "auto", "--jobs", "8", "--no-preview", *extra]
    if name == "labels_cache":
        args += ["--labels-cache", str(tmp_path / "labels.npy")]
    r = trace(args, out)
    assert r.returncode == 0, r.stderr[-3000:]
    _check_svg(out)


def test_preview_and_debug_outputs(trace, small_image, tmp_path):
    img = small_image("openai.png")
    out, prev, dbg = tmp_path / "a.svg", tmp_path / "a_preview.png", tmp_path / "a_debug.png"
    r = trace(["--in", img, "--gpu", "auto", "--jobs", "8", "--preset", "logo", "--scale", "1",
               "--preview", prev, "--debug", dbg], out)
    assert r.returncode == 0, r.stderr[-3000:]
    _check_svg(out)
    assert prev.is_file() and dbg.is_file()


def test_quiet_run_prints_no_progress(trace, small_image, tmp_path):
    img = small_image("openai.png")
    out = tmp_path / "q.svg"
    r = trace(["--in", img, "--gpu", "auto", "--jobs", "1", "--no-preview", "--quiet",
               "--preset", "logo", "--scale", "1"], out)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "结构张量" not in r.stdout          # progress lines are gated by log()
    _check_svg(out)


def test_svgz_matches_svg(trace, small_image, tmp_path):
    img = small_image("openai.png")
    out = tmp_path / "z.svg"
    r = trace(["--in", img, "--gpu", "auto", "--jobs", "1", "--no-preview", "--preset", "logo",
               "--scale", "1", "--gzip"], out)
    assert r.returncode == 0, r.stderr[-3000:]
    with gzip.open(str(out) + "z", "rt", encoding="utf-8") as fh:
        assert fh.read() == out.read_text(encoding="utf-8")


def test_missing_input_fails_cleanly(trace, tmp_path):
    r = trace(["--in", tmp_path / "nope.png", "--no-preview"], tmp_path / "x.svg")
    assert r.returncode != 0
    assert "找不到输入文件" in (r.stderr + r.stdout)
