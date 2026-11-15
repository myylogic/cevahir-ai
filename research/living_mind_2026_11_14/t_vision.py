# -*- coding: utf-8 -*-
"""Seeing and recognising: generated images, a few named, new ones recognised by resemblance (F18).

    python research/living_mind_2026_11_14/t_vision.py --seeds 0 1 2 --taught 3 --tested 10

Three kinds of drawing (a red disc "top", a blue square "kutu", a green triangle "çatı") at random size, position and
background brightness. The mind watches `taught` images of each kind named ("bu ne ?" -> "bu bir <ad> ."), twice. Then
for `tested` new images of each kind it is asked "bu ne" in a chat; right if its answer contains the name. Control:
the same life with resemblance off (F18 disabled). Images are drawn with Pillow and never seen by the core as pixels:
only the adapter's features (gorme.py) reach it.
"""
import argparse, json, os, random, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from PIL import Image, ImageDraw  # noqa: E402
from zihin import Mind, Config  # noqa: E402
from zihin.__main__ import Session, CHANNEL  # noqa: E402
from zihin.gorme import features  # noqa: E402

KINDS = {"top": ((210, 30, 30), "disc"), "kutu": ((30, 60, 210), "square"), "çatı": ((30, 170, 40), "triangle")}


def draw(kind, rng, path):
    colour, shape = KINDS[kind]
    bg = rng.randint(170, 250)
    img = Image.new("RGB", (128, 128), (bg, bg, bg))
    d = ImageDraw.Draw(img)
    size = rng.randint(40, 90)
    x, y = rng.randint(5, 123 - size), rng.randint(5, 123 - size)
    jitter = tuple(max(0, min(255, c + rng.randint(-25, 25))) for c in colour)
    if shape == "disc":
        d.ellipse([x, y, x + size, y + size], fill=jitter)
    elif shape == "square":
        d.rectangle([x, y, x + size, y + size], fill=jitter)
    else:
        d.polygon([(x + size / 2, y), (x, y + size), (x + size, y + size)], fill=jitter)
    img.save(path)


def life(seed, taught, tested, resemblance, folder):
    rng = random.Random(seed)
    cfg = Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                         backoff=True, resemblance=resemblance)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    lessons = []
    for kind in KINDS:
        for i in range(taught):
            p = os.path.join(folder, "t_%d_%s_%d.png" % (seed, kind, i))
            draw(kind, rng, p)
            lessons.append((kind, p))
    for _ in range(2):
        rng.shuffle(lessons)
        for kind, p in lessons:
            s.view = features(p)
            s.watch("bu ne ?", "bu bir %s ." % kind)
    right, answers = 0, []
    for kind in KINDS:
        for i in range(tested):
            p = os.path.join(folder, "n_%d_%s_%d.png" % (seed, kind, i))
            draw(kind, rng, p)
            s.view = features(p)
            said = s.reply("bu ne")
            s.close_turn()
            ok = kind in said.split()
            right += ok
            answers.append((kind, said))
    return right / float(tested * len(KINDS)), answers[:6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--taught", type=int, default=3)
    ap.add_argument("--tested", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "t_vision.jsonl"))
    args = ap.parse_args()
    folder = tempfile.mkdtemp()
    for seed in args.seeds:
        for resemblance in (True, False):
            acc, sample = life(seed, args.taught, args.tested, resemblance, folder)
            row = {"seed": seed, "resemblance": resemblance, "taught_per_kind": args.taught,
                   "tested_per_kind": args.tested, "accuracy": acc, "sample": sample}
            with open(args.out, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
