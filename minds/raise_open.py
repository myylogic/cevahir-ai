"""Raise the open mind of release v7 from nothing, on the repository's public material only.

    python minds/raise_open.py OUT_FOLDER

The program's whole mind (``VARIANTS["full"]``, every mechanism of the full configuration on, silent claims too) is
created empty and studied with ``research/living_mind_2026_11_22/etut.py``, in this order:

1. the dialogue and text files of the first experiments (``research/living_mind_2026_11_14/data/``);
2. the seven school subjects of ``research/living_mind_2026_11_22/t_school.py`` (``subjects(0)``), written out as
   lessons;
3. the 24 lessons of public knowledge (``research/living_mind_2026_11_22/dersler/``).

Each lesson is taught, its dialogues asked in a shuffled order, a wrong attempt taken back exactly and the answer
shown, round after round (3 at most); every dialogue is asked again at the end. The mind is saved as
``OUT_FOLDER/cevahir-open.zihin`` with its ledger; the released file is the same mind saved without it
(``persist.load(..., ledger=False)`` then ``persist.save``), compressed.
"""
import glob
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "research", "living_mind_2026_11_22"))
import torch  # noqa: E402

torch.set_num_threads(1)
torch.manual_seed(0)
import zihin.__main__ as chat  # noqa: E402
from zihin import Mind  # noqa: E402
from zihin.__main__ import CHANNEL  # noqa: E402
from zihin.persist import save  # noqa: E402
import t_school  # noqa: E402


def main() -> None:
    out = sys.argv[1]
    os.makedirs(os.path.join(out, "lessons"), exist_ok=True)
    for name, (lessons, _) in t_school.subjects(0).items():          # the school's seven subjects, as lessons
        with open(os.path.join(out, "lessons", "school_%s.txt" % name), "w", encoding="utf-8") as f:
            f.write("# %s (research/living_mind_2026_11_22/t_school.py, subjects(0))\n" % name)
            for lesson in lessons:
                f.write((lesson[1] + "\n") if lesson[0] == "tell" else (lesson[1] + "\t" + lesson[2] + "\n"))
    mind = Mind(chat.VARIANTS["full"](out))
    mind.dil = "tr"
    mind.S.log, mind.S.keep_log, mind.cfg.keep_log, mind.cfg.log_path = {}, False, False, None
    mind.add_channel(CHANNEL, "act")
    save(mind, os.path.join(out, "empty.zihin"))
    data = os.path.join(REPO, "research", "living_mind_2026_11_14", "data")
    lessons = [os.path.join(data, name) for name in ("first_dialogues.tsv", "feedback_dialogues.tsv",
                                                     "school_text.txt", "school_questions.tsv", "imagination_text.txt")]
    lessons += sorted(glob.glob(os.path.join(out, "lessons", "school_*.txt")))
    lessons += sorted(glob.glob(os.path.join(REPO, "research", "living_mind_2026_11_22", "dersler", "*.txt")))
    with open(os.path.join(out, "sources.json"), "w", encoding="utf-8") as f:
        json.dump(lessons, f, indent=1)
    t0 = time.time()
    done = subprocess.run([sys.executable, os.path.join(REPO, "research", "living_mind_2026_11_22", "etut.py"),
                           "--mind", os.path.join(out, "empty.zihin"), "--save", os.path.join(out, "cevahir-open.zihin"),
                           "--lessons", *lessons, "--rounds", "3", "--kapat", "",
                           "--report", os.path.join(out, "study.json")])
    print("done", done.returncode, round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
