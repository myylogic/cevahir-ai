"""Raise the open Cevahir from nothing, on the repository's public material only (living/cevahir.py).

    python minds/raise_cevahir.py OUT_FOLDER

The same lessons, in the same order, as the open mind of release v7 (raise_open.py, sources.json): the dialogue and
text files of the first experiments, the seven school subjects of t_school.py written out as lessons, and the 24
lessons of public knowledge. Cevahir (Zihin's whole mind and the cortex, meeting in the harmony) is created empty and
studied with research/living_intelligence_2026_11_24/m3_etut.py: each lesson taught, its questions answered by
Cevahir in conversation, a right answer confirmed, a wrong one forgotten exactly and the answer shown, round after
round (3 at most), then every dialogue asked again. It is kept compressed in ``OUT_FOLDER/cevahir`` (Cevahir.save),
with the study's report (counts only) in ``OUT_FOLDER/study_cevahir.json``.
"""
import glob
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "research", "living_mind_2026_11_22"))


def main() -> None:
    import t_school
    out = sys.argv[1]
    os.makedirs(os.path.join(out, "lessons"), exist_ok=True)
    for name, (lessons, _) in t_school.subjects(0).items():          # the school's seven subjects, as lessons
        with open(os.path.join(out, "lessons", "school_%s.txt" % name), "w", encoding="utf-8") as f:
            f.write("# %s (research/living_mind_2026_11_22/t_school.py, subjects(0))\n" % name)
            for lesson in lessons:
                f.write((lesson[1] + "\n") if lesson[0] == "tell" else (lesson[1] + "\t" + lesson[2] + "\n"))
    data = os.path.join(REPO, "research", "living_mind_2026_11_14", "data")
    lessons = [os.path.join(data, name) for name in ("first_dialogues.tsv", "feedback_dialogues.tsv",
                                                     "school_text.txt", "school_questions.tsv", "imagination_text.txt")]
    lessons += sorted(glob.glob(os.path.join(out, "lessons", "school_*.txt")))
    lessons += sorted(glob.glob(os.path.join(REPO, "research", "living_mind_2026_11_22", "dersler", "*.txt")))
    t0 = time.time()
    done = subprocess.run([sys.executable, os.path.join(REPO, "research", "living_intelligence_2026_11_24",
                                                        "m3_etut.py"),
                           "--lessons", *lessons, "--rounds", "3", "--save", os.path.join(out, "cevahir"),
                           "--report", os.path.join(out, "study_cevahir.json")])
    print("done", done.returncode, round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
