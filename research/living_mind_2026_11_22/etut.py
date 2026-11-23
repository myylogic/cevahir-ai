# -*- coding: utf-8 -*-
"""etut.py: a study session (etüt) with a saved mind, as a teacher holds one.

    python research/living_mind_2026_11_22/etut.py --mind IN.zihin --save OUT.zihin
                   [--read PATH ...] [--lessons FILE ...] [--rounds 3] [--kapat silent_claims] [--report out.json]

Asked by the owner: grow the mind raised on his own data by studying with it, every subject, as a teacher would.

The mind is loaded and given the program's whole mind (``configure``, as ``python -m zihin --load`` does), with the
mechanisms of ``--kapat`` kept off (``silent_claims`` for a large mind: section 32). Then, in order:

1. ``--read``: files or folders the mind reads as the program reads them (``!oku``: documents, ``.json`` dialogues).
2. ``--lessons``: each lesson is a text file, one line a sentence (told by the teacher) or ``question<TAB>answer``
   (a dialogue watched); ``#`` lines are notes; ``?question<TAB>answer`` lines are never taught, only asked at the
   end (what the lesson did not say: does the mind reach it?). A lesson is studied as an etüt:
   * taught: every sentence and dialogue of it lived once (``!ogret``);
   * asked: every dialogue's question, in a shuffled order, answered by the mind itself (it lives the question and
     its own answer). A right answer is confirmed (``!dogru``). A wrong or a missing one is taken back exactly (the
     moments of the attempt retracted from the ledger kept from the lesson on: the mind's own act is the surest
     evidence it holds, and a wrong one kept would outweigh the teacher's correction, which was found by studying)
     and the right answer is shown (the dialogue watched again);
   * asked again, round after round, until every answer is right or ``--rounds`` run out.
   The examination lines are asked last, on copies (nothing is learned from them).
3. The mind is saved to ``--save`` (never over ``--mind``), after the reading too (``--save`` with ``.okuma``).

``--report``: what was studied and how it went (counts only: the right answers of each round, of the examination).
The lessons are the teacher's (written for this, public knowledge); a mind raised on private data stays where it is.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import torch  # noqa: E402


def lesson_lines(path: str):
    """(the lines to teach, the dialogues to ask, the examination dialogues)."""
    taught, asked, exam = [], [], []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line.startswith("?"):
                q, a = line[1:].split("\t", 1)
                exam.append((q.strip(), a.strip()))
                continue
            taught.append(line)
            if "\t" in line:
                q, a = line.split("\t", 1)
                asked.append((q.strip(), a.strip()))
    return taught, asked, exam


def study(s, path: str, rounds: int, seed: int, say=print, show: bool = False) -> dict:
    from zihin.__main__ import Session
    taught, asked, exam = lesson_lines(path)
    name = os.path.splitext(os.path.basename(path))[0]
    t0 = time.time()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        tmp.write("\n".join(taught) + "\n")
    try:
        say("  %s: %s" % (name, s.teach(tmp.name).strip()))
    finally:
        os.remove(tmp.name)
    from zihin.__main__ import FINAL

    def trimmed(text):                                   # its words, without the closing punctuation
        words = s.words(text)
        while words and words[-1] in FINAL:
            words = words[:-1]
        return words

    def judge(said, answer):
        """(right: it began with the whole answer; went on: it said more than the answer after it)."""
        want, words = trimmed(answer), s.words(said)
        right = said != "(sessiz)" and words[:len(want)] == want
        return right, right and any(w not in FINAL for w in words[len(want):])

    def pause():                                         # a silence between two questions (it ends the episode)
        for _ in range(2):
            s.mind.step([])
    mind = s.mind
    if not mind.S.keep_log:                              # a ledger of the moments from here on, so that an attempt
        mind.S.keep_log = True                           # marked wrong can be taken back exactly
        if not isinstance(mind.S.log, dict):
            mind.S.log = {}
    rng = random.Random(seed)
    scores, went_on = [], []
    for r in range(rounds):
        order = list(asked)
        rng.shuffle(order)
        right = longer = 0
        for q, a in order:
            pause()
            first = mind.t + 1
            said = s.reply(q, limit=40)
            ok, went = judge(said, a)
            if ok and not went:
                right += 1
                s.feedback(True)
            else:
                if ok:
                    right += 1                           # right, but it did not stop there: taken back too, and the
                    longer += 1                          # answer shown as it ends
                if show:
                    say("      %s -> %s   (doğrusu: %s)" % (q, said, a))
                for m in range(mind.t, first - 1, -1):   # the attempt marked wrong is taken back exactly (its own
                    if m in mind.S.log:                  # act would otherwise stand as the surest evidence it has)
                        mind.retract(m)
                s.open_turn, s.answered = False, None
                s.watch(q, a)                            # and the right answer is shown
        scores.append(right)
        went_on.append(longer)
        say("    tur %d: %d / %d (doğru olup cevaptan sonra konuşmayı sürdüren: %d)" % (r + 1, right, len(asked), longer))
        if right == len(asked):
            break
    passed = 0
    for q, a in exam:
        said = Session(s.mind.clone()).reply(q, limit=40)
        passed += judge(said, a)[0]
    if exam:
        say("    sınav (derste geçmeyen): %d / %d" % (passed, len(exam)))
    return {"lesson": name, "lines": len(taught), "asked": len(asked), "rounds": scores, "went_on": went_on,
            "exam": [passed, len(exam)],
            "seconds": round(time.time() - t0, 1), "moments": s.mind.t}


def review(s, paths, say=print, undo: bool = True) -> dict:
    """Every dialogue of the lessons asked again, in the order of the lessons, once all of them were studied: what
    stayed. With ``undo`` each attempt is taken back exactly afterwards (nothing is learned from the review; on a
    large mind this is slow); without it the mind is not saved after the review (``--review-only``)."""
    from zihin.__main__ import FINAL
    mind = s.mind
    if undo and not mind.S.keep_log:
        mind.S.keep_log = True
        if not isinstance(mind.S.log, dict):
            mind.S.log = {}

    def trimmed(text):
        words = s.words(text)
        while words and words[-1] in FINAL:
            words = words[:-1]
        return words
    out = {}
    for path in paths:
        _, asked, _ = lesson_lines(path)
        right = 0
        for q, a in asked:
            for _ in range(2):
                mind.step([])
            first = mind.t + 1
            said = s.reply(q, limit=40)
            want, words = trimmed(a), s.words(said)
            right += said != "(sessiz)" and words[:len(want)] == want
            if undo:
                for m in range(mind.t, first - 1, -1):
                    if m in mind.S.log:
                        mind.retract(m)
            s.open_turn, s.answered = False, None
        name = os.path.splitext(os.path.basename(path))[0]
        out[name] = [right, len(asked)]
        say("  genel tekrar %s: %d / %d" % (name, right, len(asked)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mind", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--read", nargs="*", default=[])
    ap.add_argument("--lessons", nargs="*", default=[])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--kapat", default="silent_claims")
    ap.add_argument("--report", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="print each wrong answer (on the console only)")
    ap.add_argument("--review", nargs="*", default=None, help="lessons asked again at the end (default: --lessons)")
    ap.add_argument("--checkpoint", action="store_true", help="save after the reading too (.okuma.zihin)")
    ap.add_argument("--review-only", action="store_true", help="only ask the lessons again; nothing is saved")
    ap.add_argument("--ac", default="", help="mechanisms to switch on by name (comma separated)")
    args = ap.parse_args()
    if os.path.abspath(args.save) == os.path.abspath(args.mind):
        raise SystemExit("--save must not be --mind: the mind studied with is kept as it was")
    torch.set_num_threads(1)
    import zihin.__main__ as chat
    from zihin.__main__ import Session
    from zihin.persist import load, save
    t0 = time.time()
    mind = load(args.mind, ledger=False)
    chat.configure(mind, "full", [k for k in args.ac.split(",") if k], [k for k in args.kapat.split(",") if k],
                   loaded=True)
    for k in [k for k in args.ac.split(",") if k]:
        if hasattr(mind.cfg, k):
            setattr(mind.cfg, k, True)
    if args.review_only:
        s = Session(mind)
        print(json.dumps(review(s, args.review or args.lessons, undo=False), ensure_ascii=False), flush=True)
        return
    s = Session(mind)
    print("yüklendi: %d an, %d birim, %d bağ (%.0f s)" % (mind.t, mind.S.n_units, mind.S.n_ties, time.time() - t0),
          flush=True)
    report = {"mind": os.path.basename(args.mind), "saved": os.path.basename(args.save), "moments_before": mind.t,
              "ties_before": mind.S.n_ties, "read": [], "lessons": []}
    for path in args.read:
        t1 = time.time()
        print(s.read_path(path).strip(), flush=True)
        report["read"].append({"path": os.path.basename(path.rstrip("/")), "seconds": round(time.time() - t1, 1),
                               "moments": mind.t})
    if args.read and args.checkpoint:
        save(mind, args.save.replace(".zihin", "") + ".okuma.zihin")
        print("okuma sonrası kaydedildi: %d an, %d bağ" % (mind.t, mind.S.n_ties), flush=True)
    for i, path in enumerate(args.lessons):
        report["lessons"].append(study(s, path, args.rounds, args.seed + i, show=args.show))
        sys.stdout.flush()
    again = args.lessons if args.review is None else args.review
    if again:
        report["review"] = review(s, again)
    save(mind, args.save)
    report.update({"moments_after": mind.t, "ties_after": mind.S.n_ties, "seconds": round(time.time() - t0, 1)})
    print("kaydedildi: %s (%d an, %d bağ)" % (args.save, mind.t, mind.S.n_ties), flush=True)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
