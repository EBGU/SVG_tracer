"""Byte-identity anchors: the documented commands must reproduce their recorded md5 exactly.

This is the project's hard invariant, so the checks are deliberately literal: the documented
command line, the recorded md5, nothing normalised.  One full-size run per anchor covers both
the byte-identity (b/e) and the well-formed-XML (d) requirement; the module is marked ``slow``
because those runs take about 65/20/8/170 seconds.

The anchors need the gitignored images under ``inputs/`` and skip loudly when those are
absent -- the md5s of the committed ``examples/*.svg`` are checked without any inputs in
``test_repository.py``, so a checkout without inputs still guards the shipped artifacts.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest


@pytest.mark.slow
def test_anchor_md5_and_xml(trace, require_inputs, md5file, anchor, tmp_path):
    """The documented command reproduces the recorded md5 and emits well-formed XML."""
    require_inputs(anchor)
    out = tmp_path / f"{anchor['name']}.svg"
    r = trace(anchor["args"], out)
    assert r.returncode == 0, r.stderr[-3000:]

    got = md5file(out)
    assert got == anchor["md5"], (
        f"{anchor['name']}: output is not byte-identical\n"
        f"  expected {anchor['md5']}\n  got      {got}\n"
        f"  command  python logo_trace.py {' '.join(map(str, anchor['args']))}")

    text = out.read_text(encoding="utf-8")
    root = ET.fromstring(text)               # raises on malformed XML
    assert root.tag.endswith("svg")
    assert 'id="regions"' in text and "<defs>" in text
