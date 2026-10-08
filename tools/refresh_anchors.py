#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Refresh the recorded md5 anchors in ``tests/conftest.py``.

The repository publishes ``examples/*.svg`` as its byte-identity reference and pins every one of them
with an md5:

* ``ANCHORS``       -- the recorded md5 of each full-size command that reproduces a shipped example;
* ``EXAMPLES_MD5``  -- the digests of the committed examples themselves.

When the traced output legitimately changes, both lists have to be refreshed together with the files,
otherwise ``tests/test_anchors.py`` and ``tests/test_repository.py`` keep failing.  This script reads
the current ``examples/*.svg``, recomputes every digest and rewrites the two lists in place.

It is deliberately literal: it rewrites only the 32-hex-digit digests (or the ``__NAME__``
placeholders the lists may start out with), so the surrounding file is never reflowed.

    python tools/refresh_anchors.py            # rewrite the digests
    python tools/refresh_anchors.py --check    # report only, exit 1 if anything is stale
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CONFTEST = os.path.join(REPO, "tests", "conftest.py")
EXAMPLES = "examples"
DIGEST = r"[0-9a-fA-F]{32}"

# The ``__NAME__`` placeholders the lists may start out with, mapped to their example file.
PLACEHOLDERS = {
    "__APPLE1__": "apple_traced_scale1.svg",
    "__APPLE4__": "apple_traced_scale4.svg",
    "__OPENAI__": "openai_traced_scale1.svg",
    "__WATERLILIES__": "water_lilies_traced_scale1.svg",
    "__WAVE__": "wave_traced_scale1.svg",
}

# every file under examples/ that is pinned by a digest, in the order the README lists it
EXAMPLE_NAMES = [
    "apple_traced_scale1.svg",
    "apple_traced_scale4.svg",
    "openai_traced_scale1.svg",
    "water_lilies_traced_scale1.svg",
    "wave_traced_scale1.svg",
]


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def example_for_args(args: list[str]) -> str | None:
    """Map an anchor's argv back to the shipped example it reproduces (``--in`` plus scale)."""
    src, scale = None, "4"  # 4.0 is the CLI default for --scale
    for i, tok in enumerate(args):
        if tok == "--in" and i + 1 < len(args):
            src = os.path.basename(args[i + 1])
        elif tok == "--scale" and i + 1 < len(args):
            scale = args[i + 1]
    if src is None:
        return None
    if src != "apple.png":
        return f"{EXAMPLES}/{os.path.splitext(src)[0]}_traced_scale1.svg"
    return f"{EXAMPLES}/apple_traced_scale{scale}.svg"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Refresh the md5 anchors in tests/conftest.py")
    ap.add_argument("--check", action="store_true",
                    help="report the current state and exit 1 when anything is stale or unfilled")
    args = ap.parse_args(argv)

    text = open(CONFTEST, encoding="utf-8").read()
    stale, report = [], []

    # 1) the EXAMPLES_MD5 map: one digest per committed example
    for name in EXAMPLE_NAMES:
        rel = f"{EXAMPLES}/{name}"
        path = os.path.join(REPO, rel)
        if not os.path.isfile(path):
            sys.exit(f"[error] missing {rel}; the shipped examples must all be present")
        digest = md5(path)
        report.append(f"{rel:48s} {digest}")
        pattern = re.compile(r'("' + re.escape(rel) + r'":\s*")' + DIGEST + r'(")')
        if not pattern.search(text):
            sys.exit(f"[error] tests/conftest.py has no EXAMPLES_MD5 entry for {rel}")
        if pattern.search(text).group(0) != f'"{rel}": "{digest}"':
            stale.append(rel)
        text = pattern.sub(lambda m: m.group(1) + digest + m.group(2), text, count=1)

    # 2) the ANCHORS list: map each recorded command back to the example it reproduces
    head, _, rest = text.partition("ANCHORS = [")
    anchors, sep, tail = rest.partition("\n]")
    for entry in re.finditer(r"dict\((.*?)\n    \),", anchors, re.S):
        block = entry.group(1)
        name = re.search(r'name="([^"]+)"', block)
        argv = re.search(r"args=\[(.*?)\]", block, re.S)
        if not name or not argv:
            continue
        rel = example_for_args(re.findall(r'"([^"]*)"', argv.group(1)))
        if rel is None or not os.path.isfile(os.path.join(REPO, rel)):
            continue
        digest = md5(os.path.join(REPO, rel))
        old = re.search(r'md5="(' + DIGEST + r')"', block)
        if old and old.group(1) != digest:
            stale.append(f"ANCHORS[{name.group(1)}] -> {rel}")
        anchors = anchors.replace(block, re.sub(r'md5="' + DIGEST + r'"', f'md5="{digest}"', block), 1)
    text = head + "ANCHORS = [" + anchors + sep + tail

    # 3) any leftover placeholder is filled from its example
    for tok, name in PLACEHOLDERS.items():
        if tok in text:
            digest = md5(os.path.join(REPO, EXAMPLES, name))
            stale.append(tok)
            report.append(f"{tok:48s} {digest}  ({EXAMPLES}/{name})")
            text = text.replace(tok, digest)

    print("\n".join(report))
    if args.check:
        if stale:
            print(f"[check] {len(stale)} stale or unfilled anchor(s): " + ", ".join(stale))
            return 1
        print("[check] every recorded md5 matches the shipped examples")
        return 0
    open(CONFTEST, "w", encoding="utf-8").write(text)
    print(f"updated {os.path.relpath(CONFTEST, REPO)} ({len(stale)} change(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
