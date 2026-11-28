# -*- coding: utf-8 -*-
"""SCAN (Lake and Baroni 2018; Loula, Baroni and Lake 2018) written out in the layout of the published repository,
generated here from its grammar. Nothing is downloaded.

    python scan_files.py OUT_FOLDER

The published files are not kept in this repository, and this study was told to download no dataset. The earlier
study generated all 20,910 commands from SCAN's grammar (research/living_mind_2026_11_15/t_compose.py) and compared
that generation with the published files as sets (research/living_mind_2026_11_22/results/scan_official.jsonl,
"scan_official_verify"): the commands with their answers, and the training and test files of add-jump and length,
are the same sets. This file writes the same generation with the published token names.

The four template splits are written by a rule inferred from the published line counts, which that record keeps
(train / test: around right 15,225 / 4,476; jump around right 18,528 / 1,173; opposite right 15,225 / 4,476;
right 15,225 / 4,476). Every pair of counts is met by one rule: commands that hold "turn" with the template's tail
("turn around right", "turn opposite right", "turn right") are in neither file; the test file holds the commands in
which a primitive stands with the tail (for "jump around right", only jump); the training file holds the rest. The
counts are asserted below. That the sets themselves equal the published ones cannot be checked here without the
published files; for add-jump and length the earlier study checked it.

Lines are written in the grammar's order, not the published files' order, so a seeded sample of a test file is a
sample of the same set but not the same commands as a sample of the published file.
"""
from __future__ import annotations

import itertools
import os
import sys

PRIMITIVES = {"walk": "I_WALK", "run": "I_RUN", "look": "I_LOOK", "jump": "I_JUMP"}
TURN = {"left": "I_TURN_LEFT", "right": "I_TURN_RIGHT"}
FILES = {"add_jump": ("add_prim_split/tasks_train_addprim_jump.txt", "add_prim_split/tasks_test_addprim_jump.txt"),
         "length": ("length_split/tasks_train_length.txt", "length_split/tasks_test_length.txt"),
         "around_right": ("template_split/tasks_train_template_around_right.txt",
                          "template_split/tasks_test_template_around_right.txt"),
         "jump_around_right": ("template_split/tasks_train_template_jump_around_right.txt",
                               "template_split/tasks_test_template_jump_around_right.txt"),
         "opposite_right": ("template_split/tasks_train_template_opposite_right.txt",
                            "template_split/tasks_test_template_opposite_right.txt"),
         "right": ("template_split/tasks_train_template_right.txt", "template_split/tasks_test_template_right.txt")}
COUNTS = {"add_jump": (13204, 7706), "length": (16990, 3920), "around_right": (15225, 4476),
          "jump_around_right": (18528, 1173), "opposite_right": (15225, 4476), "right": (15225, 4476)}
TAILS = {"around_right": ("around", "right"), "jump_around_right": ("around", "right"),
         "opposite_right": ("opposite", "right"), "right": ("right",)}


def commands():
    """All 20,910 commands with their answers: C -> S and S | S after S | S; S -> V twice | V thrice | V;
    V -> U | D | U opposite D | U around D | turn opposite D | turn around D; D -> U left | U right | turn left |
    turn right; U -> walk | look | run | jump."""
    v = [([u], [a]) for u, a in PRIMITIVES.items()]
    for d, t in TURN.items():
        v.append((["turn", d], [t]))
        v.append((["turn", "opposite", d], [t, t]))
        v.append((["turn", "around", d], [t] * 4))
        for u, a in PRIMITIVES.items():
            v += [([u, d], [t, a]), ([u, "opposite", d], [t, t, a]), ([u, "around", d], [t, a] * 4)]
    sentences = []
    for c, a in v:
        sentences += [(c, a), (c + ["twice"], a * 2), (c + ["thrice"], a * 3)]
    out = list(sentences)
    for (c1, a1), (c2, a2) in itertools.product(sentences, sentences):
        out += [(c1 + ["and"] + c2, a1 + a2), (c1 + ["after"] + c2, a2 + a1)]
    return [(tuple(c), tuple(a)) for c, a in out]


def holds(command: tuple, part: tuple) -> bool:
    return any(command[i:i + len(part)] == part for i in range(len(command) - len(part) + 1))


def split(name: str, every):
    if name == "add_jump":
        return ([d for d in every if "jump" not in d[0] or d[0] == ("jump",)],
                [d for d in every if "jump" in d[0] and d[0] != ("jump",)])
    if name == "length":
        return [d for d in every if len(d[1]) <= 22], [d for d in every if len(d[1]) >= 24]
    tail = TAILS[name]
    verbs = ("jump",) if name == "jump_around_right" else tuple(PRIMITIVES)
    kept = [d for d in every if not holds(d[0], ("turn",) + tail)]
    test = [d for d in kept if any(holds(d[0], (u,) + tail) for u in verbs)]
    asked = set(test)
    return [d for d in kept if d not in asked], test


def write(folder: str) -> dict:
    every = commands()
    assert len(every) == len(set(every)) == 20910
    rows = {}

    def put(path, pairs):
        path = os.path.join(folder, path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for c, a in pairs:
                f.write("IN: %s OUT: %s\n" % (" ".join(c), " ".join(a)))

    put("tasks.txt", every)
    for name, (train_file, test_file) in FILES.items():
        train, test = split(name, every)
        assert (len(train), len(test)) == COUNTS[name], (name, len(train), len(test))
        assert not set(c for c, _ in train) & set(c for c, _ in test)
        put(train_file, train)
        put(test_file, test)
        rows[name] = [len(train), len(test)]
    return rows


if __name__ == "__main__":
    print(write(sys.argv[1]))
