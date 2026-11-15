# -*- coding: utf-8 -*-
"""Seeing: an image as a perceived thing with its aspects (vision 10: any sense; an adapter, no core code).

No trained network: plain arithmetic on the pixels. An image becomes
  * one item on the sense channel "goz": its gist (a 64-bit difference hash), the same image is the same thing;
  * facets on the channel "gorunum" (they accompany the item, they are not part of a sentence):
      - 8 position-specific bytes of the difference hash (coarse shape and layout of light and dark),
      - the colour of each cell of a 4 x 4 grid (8 hues when saturated, else dark / mid / light),
      - the 3 dominant colours of the whole image (position-free),
      - the dominant edge orientation and the edge density.
Two photos of the same face, object or scene share many facets; the core's facet similarity (F18) lets a new image
join the categories of the images it resembles, and lets the most similar members speak loudest.
"""
from __future__ import annotations

import math
from typing import List, Tuple

EYE, LOOK = "goz", "gorunum"


def _palette(r: float, g: float, b: float) -> str:
    mx, mn = max(r, g, b), min(r, g, b)
    light = (mx + mn) / 2.0
    if mx - mn < 40:                                      # unsaturated: a grey level
        return "koyu" if light < 85 else ("orta" if light < 170 else "acik")
    if mx == r:
        h = ((g - b) / (mx - mn)) % 6
    elif mx == g:
        h = (b - r) / (mx - mn) + 2
    else:
        h = (r - g) / (mx - mn) + 4
    hue = int(((h * 60.0) % 360) // 45)                   # 8 hues of 45 degrees
    return "ton%d" % hue


def features(path: str) -> Tuple[str, List[str]]:
    """(gist symbol, facet symbols) of an image file."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    small = img.resize((32, 32))
    px = list(small.getdata())
    gray9 = img.convert("L").resize((9, 8))
    g = list(gray9.getdata())
    bits = [1 if g[r * 9 + c] > g[r * 9 + c + 1] else 0 for r in range(8) for c in range(8)]
    value = 0
    for b in bits:
        value = (value << 1) | b
    facets = ["dh%d=%02x" % (i, (value >> (8 * (7 - i))) & 0xFF) for i in range(8)]
    for row in range(4):                                  # colour layout
        for col in range(4):
            cell = [px[(row * 8 + y) * 32 + col * 8 + x] for y in range(8) for x in range(8)]
            r = sum(p[0] for p in cell) / 64.0
            gg = sum(p[1] for p in cell) / 64.0
            b = sum(p[2] for p in cell) / 64.0
            facets.append("h%d%d=%s" % (row, col, _palette(r, gg, b)))
    counts = {}
    for p in px:
        k = _palette(*p)
        counts[k] = counts.get(k, 0) + 1
    for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:3]:
        facets.append("renk=%s" % k)
    lum = [0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2] for p in px]
    hist = [0.0, 0.0, 0.0, 0.0]
    total = 0.0
    for y in range(1, 31):
        for x in range(1, 31):
            gx = lum[y * 32 + x + 1] - lum[y * 32 + x - 1]
            gy = lum[(y + 1) * 32 + x] - lum[(y - 1) * 32 + x]
            mag = math.hypot(gx, gy)
            if mag < 1e-6:
                continue
            angle = math.degrees(math.atan2(gy, gx)) % 180
            hist[int(((angle + 22.5) % 180) // 45)] += mag
            total += mag
    names = ("yatay", "capraz1", "dikey", "capraz2")
    facets.append("kenar=%s" % names[max(range(4), key=lambda i: hist[i])])
    facets.append("yogunluk=%d" % min(3, int(total / (30 * 30 * 40))))
    facets += _figure(img)
    return "g%016x" % value, facets


def _figure(img) -> List[str]:
    """Position- and size-free aspects of what stands out from the background (the border's typical colour): how
    much of its bounding box it fills, its width to height, and its own dominant colour."""
    small = img.resize((64, 64))
    px = list(small.getdata())
    border = [px[i] for i in range(64)] + [px[63 * 64 + i] for i in range(64)] + \
        [px[i * 64] for i in range(64)] + [px[i * 64 + 63] for i in range(64)]
    bg = tuple(sorted(c[k] for c in border)[len(border) // 2] for k in range(3))
    mask = [sum(abs(p[k] - bg[k]) for k in range(3)) > 60 for p in px]
    xs = [i % 64 for i, m in enumerate(mask) if m]
    ys = [i // 64 for i, m in enumerate(mask) if m]
    if len(xs) < 20:
        return ["figur=yok"]
    w, h = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
    fill = len(xs) / float(w * h)
    aspect = w / float(h)
    counts = {}
    for i, m in enumerate(mask):
        if m:
            k = _palette(*px[i])
            counts[k] = counts.get(k, 0) + 1
    colour = max(counts, key=counts.get)
    return ["doluluk=%d" % min(9, int(fill * 10)), "oran=%s" % ("yatik" if aspect > 1.25 else "dik" if aspect < 0.8 else "esit"),
            "figur_renk=%s" % colour]
