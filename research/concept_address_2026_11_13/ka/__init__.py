# -*- coding: utf-8 -*-
"""Kavram Adresi (KA) — the concept-address study package.

Importing this package puts the frozen CEV-11 study folder and the repository root on ``sys.path`` so that
``cev11`` (imported, never edited), ``src``, ``model_management`` and ``tokenizer_management`` resolve.
"""
from __future__ import annotations

import pathlib
import sys

KA_DIR = pathlib.Path(__file__).resolve().parents[1]
REPO = KA_DIR.parents[1]
CEV11_DIR = REPO / "research" / "living_learning_paradigm_2026_11_13"

for _p in (str(CEV11_DIR), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Record format version. ka/2: DONE.json describes the whole world directory (complete phases, required phases),
# gate indicators cover the scored sequence, FT records carry no wall-clock time. Records of another version are
# refused by the selection, the analysis and the record validation.
KA_VERSION = "ka/2"

__all__ = ["KA_DIR", "REPO", "CEV11_DIR", "KA_VERSION"]
