# -*- coding: utf-8 -*-
"""t_school.py: one mind taught seven subjects, examined on what it was never asked (section 41).

    python research/living_mind_2026_11_22/t_school.py mind [--seeds 3] [--separate]
    python research/living_mind_2026_11_22/t_school.py transformer [--seeds 3] [--steps 8000]
    python research/living_mind_2026_11_22/t_school.py draw    (the drawings and the melody one mind made, as files)

Asked by the owner: assume everything built is one mind; teach it mathematics, physics, biology, chemistry and
coding, and drawing and sound, and test it. The lessons below are short Turkish sentences read (told) and dialogues
watched (someone asks, someone answers), as a teacher gives them; every subject is taught to the same mind, the
lessons of all subjects interleaved in one shuffled order, three times. Nothing of a subject is written into the
mind: the tables here are the teacher's, and the mind only lives what the teacher says.

The examination asks only what was never asked during the lessons, of three kinds:

* **deduction** (biology, physics): facts read apart ("kartal bir kuştur .", "kuşlar uçar .") and the question asked
  of two members of each class; asked of the three never asked.
* **composition** (chemistry, coding, drawing): parts answered alone and in some combinations; asked of combinations
  never shown, and of a part shown only alone (as SCAN's add-primitive split; Lake & Baroni 2018).
* **relation** (mathematics, music): a lived sequence (counting, the scale) and a question answered with what
  follows in it; asked of the places never asked. Addition of two numbers never added is asked too, as a control:
  nothing lived says it (a lookup cannot do it either), and both learners are expected to fail it.

What was asked during the lessons is asked again (recall), on the same copies.

Learners: ``mind``: the program's whole mind (``VARIANTS["full"]``) with a window of 12 items (``knob F``, as the SCAN
runs: an answer here has at most 6 words); ``--separate``: each subject taught to a mind of its own (does one mind
holding seven subjects lose anything?). ``transformer``: a decoder-only Transformer trained from scratch on exactly
the same lessons (each told sentence as a sequence, each dialogue as "question ⏎ answer ."), the recipe of
``t_compose.py``, until its answers to the lessons' questions are exact (reported). Exact match of the whole answer.

``draw``: the drawings (move sequences: ileri = one step, sağ / sol = a quarter turn) and the melody (notes) the
mind produced for questions never asked, rendered by a fixed renderer to SVG and WAV files in ``results/school/``:
the mind makes the sequence, the renderer only turns it into pixels and sound.

Appends to ``results/school.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import sys
import time
import wave
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import torch  # noqa: E402

OUT = os.path.join(HERE, "results", "school.jsonl")
DRAWN = os.path.join(HERE, "results", "school")
WINDOW = 12

# ------------------------------------------------------------------------------------------------ the teacher's tables
ANIMALS = {"memeli": ("memelidir", "memeliler", "emzirir", ["kedi", "köpek", "inek", "fil", "at"]),
           "kuş": ("kuştur", "kuşlar", "uçar", ["serçe", "kartal", "güvercin", "baykuş", "martı"]),
           "balık": ("balıktır", "balıklar", "yüzer", ["hamsi", "sazan", "levrek", "alabalık", "palamut"]),
           "sürüngen": ("sürüngendir", "sürüngenler", "sürünür", ["yılan", "kertenkele", "timsah", "kaplumbağa",
                                                                 "iguana"])}
MATTER = {"katı": ("katıdır", "katılar", "erir", ["buz", "tereyağı", "mum", "çikolata", "kurşun"]),
          "sıvı": ("sıvıdır", "sıvılar", "kaynar", ["su", "süt", "alkol", "kahve", "zeytinyağı"]),
          "gaz": ("gazdır", "gazlar", "genleşir", ["hava", "helyum", "buhar", "duman", "azot"])}
ELEMENTS = {"h": "hidrojen", "o": "oksijen", "c": "karbon", "n": "nitrojen", "s": "kükürt", "na": "sodyum",
            "cl": "klor", "fe": "demir", "k": "potasyum", "ca": "kalsiyum"}
ONLY_ALONE = ["fe", "k", "ca"]                               # elements shown only alone (as SCAN's "jump")
HELD_PAIRS = [("s", "o"), ("n", "h"), ("c", "s"), ("na", "o"), ("h", "s"), ("cl", "o")]
VARIABLES = ["a", "b", "x", "y", "sayi", "isim", "liste", "toplam", "sonuc", "deger"]
NEW_VARIABLES = ["sonuc", "deger", "toplam"]                 # never used in a lesson
NOTES = ["do", "re", "mi", "fa", "sol", "la", "si"]
SHAPES = {"çizgi": ["ileri", "ileri"], "köşe": ["ileri", "sağ"], "dönüş": ["sağ", "sağ"],
          "basamak": ["ileri", "sol", "ileri"], "kanca": ["ileri", "sol"]}
NEW_SHAPE = "kanca"                                          # shown only alone
DRAW = {"{x} çiz": lambda x: x, "iki kez {x} çiz": lambda x: x + x}
DRAW2 = {"{x} sonra {y} çiz": lambda x, y: x + y}


def subjects(seed: int) -> Dict[str, Tuple[list, list]]:
    """{subject: (lessons, examination)}: a lesson is ("tell", sentence) or ("watch", question, answer); a question of
    the examination is (kind, question, the right answer)."""
    rng = random.Random(seed)
    out = {}
    for name, table, frame in (("biyoloji", ANIMALS, "{x} ne yapar ?"), ("fizik", MATTER, "{x} ısınınca ne olur ?")):
        lessons, exam = [], []
        for copula, plural, prop, members in table.values():
            for m in members:
                lessons.append(("tell", "%s bir %s ." % (m, copula)))
            lessons += [("tell", "%s %s ." % (plural, prop))] * 2
            for m in members[:2]:                            # asked of two; the three others never
                lessons.append(("watch", frame.format(x=m), "%s %s ." % (m, prop)))
                exam.append(("recall", frame.format(x=m), "%s %s" % (m, prop)))
            for m in members[2:]:
                exam.append(("deduction", frame.format(x=m), "%s %s" % (m, prop)))
        out[name] = (lessons, exam)

    lessons, exam = [], []                                   # chemistry: what a compound is made of
    for sym, el in ELEMENTS.items():
        lessons.append(("watch", "%s nedir ?" % sym, "%s ." % el))
        exam.append(("recall", "%s nedir ?" % sym, el))
    taught = [s for s in ELEMENTS if s not in ONLY_ALONE]
    for a in taught:
        for b in taught:
            if a == b:
                continue
            q, ans = "%s %s neyden oluşur ?" % (a, b), "%s %s" % (ELEMENTS[a], ELEMENTS[b])
            if (a, b) in HELD_PAIRS:
                exam.append(("composition: a new pair", q, ans))
            else:
                lessons.append(("watch", q, ans + " ."))
    for new in ONLY_ALONE:
        for other in rng.sample(taught, 3):
            for a, b in ((new, other), (other, new)):
                exam.append(("composition: an element seen only alone", "%s %s neyden oluşur ?" % (a, b),
                             "%s %s" % (ELEMENTS[a], ELEMENTS[b])))
    out["kimya"] = (lessons, exam)

    lessons, exam = [], []                                   # mathematics: counting, plus one, plus anything
    numbers = [str(i) for i in range(1, 21)]
    lessons += [("tell", " ".join(numbers) + " .")] * 3
    asked = set(rng.sample(range(1, 19), 9))
    for i in range(1, 20):
        q, ans = "%d sonrası nedir ?" % i, str(i + 1)
        if i in asked:
            lessons.append(("watch", q, ans + " ."))
            exam.append(("recall", q, ans))
        else:
            exam.append(("relation: what follows", q, ans))
    plus = set(rng.sample(range(1, 19), 9))
    for i in range(1, 20):
        q, ans = "%d artı 1 kaç ?" % i, str(i + 1)
        if i in plus:
            lessons.append(("watch", q, ans + " ."))
        else:
            exam.append(("relation: plus one", q, ans))
    pairs = [(a, b) for a in range(2, 8) for b in range(2, 8)]
    rng.shuffle(pairs)
    for a, b in pairs[:18]:
        lessons.append(("watch", "%d artı %d kaç ?" % (a, b), "%d ." % (a + b)))
    for a, b in pairs[18:26]:
        exam.append(("control: a sum never added", "%d artı %d kaç ?" % (a, b), str(a + b)))
    out["matematik"] = (lessons, exam)

    lessons, exam = [], []                                   # coding (Python)
    old = [v for v in VARIABLES if v not in NEW_VARIABLES]
    for v in old:
        lessons.append(("watch", "%s yazdır" % v, "print ( %s ) ." % v))
        lessons.append(("watch", "%s uzunluğu" % v, "len ( %s ) ." % v))
        exam.append(("recall", "%s yazdır" % v, "print ( %s )" % v))
    for v in NEW_VARIABLES:
        exam.append(("composition: a new name", "%s yazdır" % v, "print ( %s )" % v))
        exam.append(("composition: a new name", "%s uzunluğu" % v, "len ( %s )" % v))
    pair_list = [(a, b) for a in old for b in old if a != b]
    rng.shuffle(pair_list)
    for a, b in pair_list[:12]:
        lessons.append(("watch", "%s ile %s toplamı" % (a, b), "%s + %s ." % (a, b)))
    for a, b in pair_list[12:18]:
        exam.append(("composition: a new pair", "%s ile %s toplamı" % (a, b), "%s + %s" % (a, b)))
    out["kodlama"] = (lessons, exam)

    lessons, exam = [], []                                   # music: the scale and three notes up from a note
    lessons += [("tell", " ".join(NOTES + ["do"]) + " .")] * 3
    for i, n in enumerate(NOTES[:5]):
        q, ans = "%s den üç nota" % n, " ".join(NOTES[i:i + 3])
        if n in ("do", "mi", "sol"):
            lessons.append(("watch", q, ans + " ."))
            exam.append(("recall", q, ans))
        else:
            exam.append(("relation: three notes up", q, ans))
    out["müzik"] = (lessons, exam)

    lessons, exam = [], []                                   # drawing: shapes and two ways to combine them
    old_shapes = [s for s in SHAPES if s != NEW_SHAPE]
    lessons.append(("watch", "%s çiz" % NEW_SHAPE, " ".join(SHAPES[NEW_SHAPE]) + " ."))
    held = {("iki kez {x} çiz", "basamak"), ("{x} sonra {y} çiz", ("dönüş", "çizgi"))}
    for form, make in DRAW.items():
        for sh in old_shapes:
            q, ans = form.format(x=sh), " ".join(make(SHAPES[sh]))
            if (form, sh) in held:
                exam.append(("composition: a new combination", q, ans))
            else:
                lessons.append(("watch", q, ans + " ."))
        if form != "{x} çiz":
            exam.append(("composition: a shape seen only alone", form.format(x=NEW_SHAPE),
                         " ".join(make(SHAPES[NEW_SHAPE]))))
    for form, make in DRAW2.items():
        for a in old_shapes:
            for b in old_shapes:
                if a == b:
                    continue
                q, ans = form.format(x=a, y=b), " ".join(make(SHAPES[a], SHAPES[b]))
                if (form, (a, b)) in held:
                    exam.append(("composition: a new combination", q, ans))
                else:
                    lessons.append(("watch", q, ans + " ."))
        for b in old_shapes[:2]:
            exam.append(("composition: a shape seen only alone", form.format(x=NEW_SHAPE, y=b),
                         " ".join(make(SHAPES[NEW_SHAPE], SHAPES[b]))))
    out["çizim"] = (lessons, exam)
    return out


def course(seed: int, chosen=None) -> List[tuple]:
    """Every lesson of the chosen subjects, three times, each time in one shuffled order (the subjects interleaved)."""
    rng = random.Random(1000 + seed)
    taught = [lesson for name, (lessons, _) in subjects(seed).items() if chosen is None or name in chosen
              for lesson in lessons]
    out = []
    for _ in range(3):
        block = list(taught)
        rng.shuffle(block)
        out += block
    return out


# ------------------------------------------------------------------------------------------------ the mind
def raise_mind(seed: int, chosen=None):
    import zihin.__main__ as chat
    from zihin import IMITATED, Mind
    from zihin.__main__ import CHANNEL, Session
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = chat.VARIANTS["full"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    torch.manual_seed(seed)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    mind.knob["F"] = WINDOW
    s = Session(mind)
    for lesson in course(seed, chosen):
        if lesson[0] == "tell":
            s.company(None)
            s.listen(lesson[1], IMITATED)
            s.end_of_turn(IMITATED, tone=".")
            mind.step([])
        else:
            s.watch(lesson[1], lesson[2])
    s.close_turn()
    return s


def examine_mind(s, exam) -> List[Tuple[str, str, str, str]]:
    from zihin.__main__ import Session
    out = []
    for kind, q, ans in exam:
        said = Session(s.mind.clone()).reply(q, limit=20)
        out.append((kind, q, ans, said))
    return out


def run_mind(seed: int, separate: bool) -> dict:
    t0 = time.process_time()
    exams = subjects(seed)
    answers = {}
    if separate:
        for name, (_, exam) in exams.items():
            answers[name] = examine_mind(raise_mind(seed, {name}), exam)
    else:
        s = raise_mind(seed)
        for name, (_, exam) in exams.items():
            answers[name] = examine_mind(s, exam)
    return {"answers": answers, "cpu_s": round(time.process_time() - t0, 1)}


# ------------------------------------------------------------------------------------------------ the Transformer
def run_transformer(seed: int, steps: int) -> dict:
    import torch.nn as nn
    torch.manual_seed(seed)
    body = lambda text: [w for w in text.lower().split() if w not in (".", "?")]   # noqa: E731 (the mind's words)
    seqs = []
    for lesson in course(seed):
        if lesson[0] == "tell":
            seqs.append(body(lesson[1]) + ["."])
        else:
            seqs.append(body(lesson[1]) + ["⏎"] + body(lesson[2]) + ["."])
    exams = subjects(seed)
    words = {w for s in seqs for w in s}
    for _, exam in exams.values():
        for _, q, ans in exam:
            words |= set(body(q)) | set(ans.split())
    vocab = sorted(words | {"<pad>", "⏎", "."})
    ix = {w: i for i, w in enumerate(vocab)}
    L = max(len(s) for s in seqs) + 2

    class Net(nn.Module):
        def __init__(self, d=64, h=4, layers=3):
            super().__init__()
            self.emb, self.pos = nn.Embedding(len(vocab), d), nn.Embedding(L + 16, d)
            block = nn.TransformerEncoderLayer(d, h, 4 * d, dropout=0.1, batch_first=True)
            self.body = nn.TransformerEncoder(block, layers)
            self.out = nn.Linear(d, len(vocab))

        def forward(self, x):
            n = x.shape[1]
            mask = torch.triu(torch.full((n, n), float("-inf")), 1)
            return self.out(self.body(self.emb(x) + self.pos(torch.arange(n)), mask=mask))

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    data = torch.full((len(seqs), L), ix["<pad>"])
    for i, s in enumerate(seqs):
        data[i, :len(s)] = torch.tensor([ix[w] for w in s])
    t0 = time.process_time()
    for _ in range(steps):
        batch = data[torch.randint(len(seqs), (32,))]
        logits = net(batch[:, :-1])
        loss = nn.functional.cross_entropy(logits.reshape(-1, len(vocab)), batch[:, 1:].reshape(-1),
                                           ignore_index=ix["<pad>"])
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()

    def decode(q):
        seq = [ix[w] for w in body(q) + ["⏎"]]
        out = []
        for _ in range(16):
            nxt = int(net(torch.tensor([seq]))[0, -1].argmax())
            if vocab[nxt] in (".", "<pad>"):
                break
            out.append(vocab[nxt])
            seq.append(nxt)
        return " ".join(out)
    answers, learned, taught = {}, 0, 0
    with torch.no_grad():
        for lesson in set(l for l in course(seed) if l[0] == "watch"):
            taught += 1
            learned += decode(lesson[1]) == " ".join(body(lesson[2]))
        for name, (_, exam) in exams.items():
            answers[name] = [(kind, q, ans, decode(q) or "(sessiz)") for kind, q, ans in exam]
    return {"answers": answers, "cpu_s": round(time.process_time() - t0, 1), "train_exact": learned,
            "train_n": taught, "loss": round(float(loss.detach()), 4)}


# ------------------------------------------------------------------------------------------------ scoring and files
def score(answers) -> Dict[str, Dict[str, List[int]]]:
    """{subject: {kind: [right, asked]}}."""
    out = {}
    for name, rows in answers.items():
        for kind, _, ans, said in rows:
            cell = out.setdefault(name, {}).setdefault(kind, [0, 0])
            cell[0] += said == ans
            cell[1] += 1
    return out


def svg(moves: List[str], title: str) -> str:
    """A fixed turtle renderer: ileri = one step forward, sağ / sol = a quarter turn right / left."""
    x, y, heading = 0.0, 0.0, 0
    points = [(x, y)]
    for m in moves:
        if m == "ileri":
            x += [1, 0, -1, 0][heading]
            y += [0, 1, 0, -1][heading]
            points.append((x, y))
        elif m == "sağ":
            heading = (heading + 1) % 4
        elif m == "sol":
            heading = (heading - 1) % 4
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    lo_x, lo_y = min(xs) - 1, min(ys) - 1
    w, h = max(xs) - lo_x + 1, max(ys) - lo_y + 1
    path = " ".join("%s%.0f,%.0f" % ("M" if i == 0 else "L", (px - lo_x) * 40, (py - lo_y) * 40)
                    for i, (px, py) in enumerate(points))
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d"><title>%s</title>'
            '<path d="%s" fill="none" stroke="black" stroke-width="6" stroke-linecap="round"/></svg>\n'
            % (w * 40, h * 40, title, path))


def melody(notes: List[str], path: str) -> None:
    """A fixed renderer: each note half a second of a sine tone (C major from middle C)."""
    freq = dict(zip(NOTES, (261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88)))
    rate = 22050
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        for n in notes:
            hz = freq.get(n)
            for i in range(rate // 2):
                env = min(1.0, i / 400.0, (rate // 2 - i) / 400.0)
                value = 0.4 * env * math.sin(2 * math.pi * hz * i / rate) if hz else 0.0
                f.writeframes(struct.pack("<h", int(value * 32767)))


def draw(seed: int = 0) -> None:
    s = raise_mind(seed)
    os.makedirs(DRAWN, exist_ok=True)
    from zihin.__main__ import Session
    exams = subjects(seed)
    made = []
    for kind, q, ans in exams["çizim"][1]:
        said = Session(s.mind.clone()).reply(q, limit=20)
        name = q.replace(" ", "_")
        with open(os.path.join(DRAWN, name + ".svg"), "w", encoding="utf-8") as f:
            f.write(svg(said.split(), "%s -> %s" % (q, said)))
        made.append({"question": q, "said": said, "right": said == ans, "kind": kind})
    for kind, q, ans in exams["müzik"][1]:
        said = Session(s.mind.clone()).reply(q, limit=20)
        melody(said.split(), os.path.join(DRAWN, q.replace(" ", "_") + ".wav"))
        made.append({"question": q, "said": said, "right": said == ans, "kind": kind})
    for row in made:
        print(json.dumps(row, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("who", choices=["mind", "transformer", "draw"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--separate", action="store_true")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    if args.who == "draw":
        draw()
        return
    total, first, extra = {}, None, []
    for seed in range(args.seeds):
        got = run_mind(seed, args.separate) if args.who == "mind" else run_transformer(seed, args.steps)
        for name, kinds in score(got["answers"]).items():
            for kind, (right, n) in kinds.items():
                cell = total.setdefault(name, {}).setdefault(kind, [0, 0])
                cell[0] += right
                cell[1] += n
        if first is None:
            first = {name: [r for r in rows if r[2] != r[3]][:4] for name, rows in got["answers"].items()}
        extra.append({k: v for k, v in got.items() if k != "answers"})
    label = {"mind": "one whole mind, every subject" if not args.separate else "a whole mind for each subject",
             "transformer": "a Transformer trained from scratch on the same lessons"}[args.who]
    lines = ["== %s: %d seeds (right / asked) ==" % (label, args.seeds)]
    for name, kinds in total.items():
        lines.append("  " + name + ": " + "; ".join("%s %d/%d" % (k, r, n) for k, (r, n) in kinds.items()))
    lines.append("  runs: %s" % json.dumps(extra, ensure_ascii=False))
    for line in lines:
        print(line, flush=True)
    row = {"test": "school", "who": args.who, "separate": args.separate, "seeds": args.seeds, "window": WINDOW,
           "scores": total, "runs": extra, "first_wrong": first, "summary": lines}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
