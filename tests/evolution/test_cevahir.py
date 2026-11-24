# -*- coding: utf-8 -*-
"""Cevahir M3: Zihin as a voice of the harmony, and the prefrontal decision (living/cevahir.py, living/_core.c)."""
import random
import shutil

import pytest

pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler")

CFG = dict(depth=8, episode=True, minlen=12, episode_bits=20, mix=True, words=True, two_inputs=True,
           tree_prior=True, recent=6, lines=True)


@pytest.fixture(scope="module")
def FastCortex(tmp_path_factory):
    import os
    os.environ.setdefault("LIVING_CACHE", str(tmp_path_factory.mktemp("living_cache")))
    from living.fast import FastCortex
    return FastCortex


def _words(seed=0, n=3000):
    rng = random.Random(seed)
    vocab = ["".join(rng.choice("abcdefghijklmnop") for _ in range(rng.randint(3, 7))) for _ in range(60)]
    return [rng.choice(vocab) for _ in range(n)], vocab, rng


def test_a_silent_voice_leaves_the_harmony_bit_identical(FastCortex):
    toks, _, _ = _words()
    text = " ".join(toks)
    plain = FastCortex(**CFG)
    voiced = FastCortex(voice=True, **CFG)
    a, bits_a = plain.stream(text, bits=True)
    total = 0.0
    for i, t in enumerate(toks):                      # chunk by chunk, the voice set and silent each time
        voiced.set_voice([], [], 0.0)
        b, _ = voiced.stream((" " if i else "") + t)
        total += b
    assert total == pytest.approx(a, rel=0, abs=1e-9)


def test_a_voice_that_knows_is_followed_and_one_that_does_not_is_ignored(FastCortex):
    toks, vocab, rng = _words(1, 4000)

    def run(kind):
        m = FastCortex(voice=kind is not None, **CFG)
        total, n = 0.0, 0
        for i, t in enumerate(toks):
            chunk = (" " if i else "") + t
            if kind == "right":
                m.set_voice([chunk], [1.0], 1.0)
            elif kind == "random":
                m.set_voice([(" " if i else "") + rng.choice(vocab)], [1.0], 1.0)
            b, _ = m.stream(chunk)
            total += b
            n += len(chunk)
        return total / n

    alone, right, wrong = run(None), run("right"), run("random")
    assert right < alone / 10
    assert wrong < alone * 1.05


def test_the_voice_follows_what_is_lived(FastCortex):
    m = FastCortex(voice=True, **CFG)
    m.stream("one two three ")
    m.set_voice([" walk", " work", " run"], [1.0, 1.0, 1.0], 1.0)
    assert m.voice_info() == (3, 0, 0)
    m.stream(" w")
    assert m.voice_info() == (2, 0, 2)
    m.stream("alk")
    assert m.voice_info() == (0, 1, 5)


def _world():
    prims = {"walk": "WALK", "run": "RUN", "look": "LOOK", "zap": "ZAP"}
    templates = [("{} twice", "{0} {0}"), ("{} thrice", "{0} {0} {0}"), ("{} left", "LTURN {0}"),
                 ("{} right", "RTURN {0}"), ("{} and {}", "{0} {1}")]
    dialogues = [(p, a) for p, a in prims.items()]
    for tpl, ans in templates:
        for p, a in prims.items():
            if p == "zap":
                continue
            if tpl.count("{}") == 2:
                for q, b in prims.items():
                    if q != "zap":
                        dialogues.append((tpl.format(p, q), ans.format(a, b)))
            else:
                dialogues.append((tpl.format(p), ans.format(a)))
    rng = random.Random(0)
    life = []
    for _ in range(2):
        d = list(dialogues)
        rng.shuffle(d)
        life += d
    tests = [(tpl.format("zap"), ans.format("ZAP")) for tpl, ans in templates if tpl.count("{}") == 1]
    tests += [("zap and walk", "ZAP WALK"), ("walk and zap", "WALK ZAP")]
    return life, tests


def test_cevahir_says_what_zihin_generalised_where_the_harmony_alone_cannot(FastCortex):
    from living.cevahir import Cevahir
    life, tests = _world()
    c = Cevahir(variant="speaking", switches={"wholes": True, "told_apart": True}, window=13)
    for q, a in life:
        c.watch(q, a + " .")
    c.session.close_turn()
    said = c.answers([q for q, _ in tests], way="every", workers=2)
    exact = {way: sum(c.session.body(s[way]) == c.session.words(a) for s, (_, a) in zip(said, tests))
             for way in ("zihin", "harmony", "cevahir")}
    assert exact["zihin"] == len(tests)
    assert exact["cevahir"] == len(tests)
    assert exact["harmony"] == 0
    assert all(s["cevahir"] == a for s, (_, a) in zip(said, tests))      # written as the answers were: "ZAP ZAP"
    assert c.counts["unaligned"] == 0


def test_a_kept_cevahir_answers_and_lives_as_the_one_kept(FastCortex, tmp_path):
    from living.cevahir import Cevahir
    life, tests = _world()
    c = Cevahir(variant="speaking", switches={"wholes": True, "told_apart": True}, window=13)
    for q, a in life[:len(life) // 2]:
        c.watch(q, a + " .")
    folder = str(tmp_path / "cevahir")
    c.save(folder)
    d = Cevahir.load(folder)
    rest = life[len(life) // 2:]
    for q, a in rest:
        assert c.watch(q, a + " .") == d.watch(q, a + " .")          # the same bits, dialogue by dialogue
    c.session.close_turn()
    d.session.close_turn()
    qs = [q for q, _ in tests]
    assert c.answers(qs, way="every", workers=2) == d.answers(qs, way="every", workers=2)
    import json
    import os
    with open(os.path.join(folder, "cevahir.json"), "a", encoding="utf-8") as f:
        f.write(" ")
    with pytest.raises(IOError):
        Cevahir.load(folder)


def test_feedback_is_kept_by_the_amygdala_and_a_wrong_answer_is_not_lived_as_right(FastCortex):
    from living.cevahir import Cevahir
    life, tests = _world()
    c = Cevahir(variant="speaking", switches={"wholes": True, "told_apart": True}, window=13)
    for q, a in life:
        c.watch(q, a + " .")
    c.session.close_turn()
    lived = c.cortex.size()["lived"]
    answer = c.converse("zap twice")
    assert answer == "ZAP ZAP"
    assert c.cortex.size()["lived"] == lived                         # the turn waits to be judged
    leads = list(c.last_leads)
    c.feedback(False)
    after_bad = c.cortex.size()["lived"]
    assert after_bad == lived + len("\nzap twice\t")                 # a wrong answer is not lived as said rightly
    for who, cell in leads:
        assert c.outcome_all[(who, cell)] == 1 and c.outcome_good[(who, cell)] == 0
    if leads:
        who, cell = leads[0]
        before = c.trust_zihin(cell)
        c.outcome_all[(who, cell)] += 5                               # five more answers led by it, judged wrong
        assert (c.trust_zihin(cell) < before) == (who == "zihin")
    c.converse("zap thrice")
    c.feedback(True)
    assert c.cortex.size()["lived"] == after_bad + len("\nzap thrice\tZAP ZAP ZAP")


def test_a_language_never_met_opens_a_channel_is_watched_and_then_spoken_noise_never(FastCortex, tmp_path):
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "research", "living_intelligence_2026_11_24"))
    from m1_channels import alien_language
    from living.cevahir import Cevahir
    c = Cevahir(variant="speaking")
    c.read("Kuşlar uçar. Balıklar yüzer. Kediler miyavlar. Köpekler havlar. " * 20)
    alien = alien_language(12000, "pua", sentences=3)
    rng = random.Random(5)
    noise = "".join(chr(0xF000 + rng.randrange(22)) for _ in range(12000))
    names = set()
    for i in range(0, 12000, 256):
        names.add(c.perceive(alien[i:i + 256]))
    seen_noise = set()
    for i in range(0, 12000, 256):
        seen_noise.add(c.perceive(noise[i:i + 256]))
    assert len(names) == 1 and c.channel not in names                 # one new channel for the new script
    assert len(seen_noise) == 1 and not (seen_noise & names) and c.channel not in seen_noise
    a, n = names.pop(), seen_noise.pop()
    from living.evidence import SPEAKING_LINE
    assert c.structure(a) >= SPEAKING_LINE and c.structure(n) < SPEAKING_LINE
    assert c.say_on(n, noise[-256:]) == ""                            # noise is watched, never spoken
    assert c.say_on(a, alien[-256:]) != ""
    assert c.perceive("Kuşlar uçar. ") == c.channel                    # the known language stays on its channel
    folder = str(tmp_path / "kept")
    c.save(folder)
    d = Cevahir.load(folder)
    assert d.structure(a) == c.structure(a) and d.say_on(a, alien[-256:]) == c.say_on(a, alien[-256:])


def test_cevahir_sleeps_forgets_one_off_contexts_and_still_answers(FastCortex):
    from living.cevahir import Cevahir
    life, tests = _world()
    c = Cevahir(variant="speaking", switches={"wholes": True, "told_apart": True}, window=13)
    for q, a in life:
        c.watch(q, a + " .")
    c.session.close_turn()
    nodes = c.cortex.size()["nodes"]
    night = c.sleep(dream=False)
    assert night["forgotten"][c.channel] > 0 and c.cortex.size()["nodes"] < nodes
    said = c.answers([q for q, _ in tests], way="every", workers=2)
    assert sum(c.session.body(s["cevahir"]) == c.session.words(a) for s, (_, a) in zip(said, tests)) == len(tests)


def test_a_word_first_met_in_the_question_is_said_back_as_it_was_written(FastCortex):
    from living.cevahir import Cevahir
    rng = random.Random(0)
    names = ["Ali", "Veli", "Ayşe", "Fatma", "Mehmet", "Zeynep", "Can", "Ece", "Deniz", "Selin", "Kerem", "Elif"]
    places = ["evde", "okulda", "parkta", "bahçede"]
    life = [("%s nerede ?" % n, "%s %s ." % (n, rng.choice(places))) for _ in range(3) for n in names]
    rng.shuffle(life)
    c = Cevahir(variant="full")
    for q, a in life:
        c.watch(q, a)
    c.session.close_turn()
    said = c.answers(["Zorbek nerede ?", "Yıldız nerede ?"], way="every", workers=2)
    for s, name in zip(said, ["Zorbek", "Yıldız"]):
        if s["zihin"] != "(sessiz)":
            assert c.session.words(s["cevahir"]) == c.session.words(s["zihin"])
            assert s["cevahir"].startswith(name)                        # the case as it was heard


def test_cevahir_forgets_an_entry_exactly_and_can_no_longer_say_it(FastCortex):
    from living.cevahir import Cevahir
    life, _ = _world()
    c = Cevahir(variant="speaking", ledger=True)
    d = Cevahir(variant="speaking", ledger=True)
    for q, a in life[:30]:
        c.watch(q, a + " .")
        d.watch(q, a + " .")
    c.watch("şifre nedir ?", "şifre kırmızı elma .")
    c.watch("şifre nedir ?", "şifre kırmızı elma .")
    assert c.cortex.digest() != d.cortex.digest()
    got = c.forget(len(c.entries) - 1)
    got2 = c.forget(len(c.entries) - 2)
    assert got["forgotten"] and got2["forgotten"]
    assert c.cortex.digest() == d.cortex.digest()                     # the counts of a mind that never heard it
    said = c.answers(["şifre nedir ?"], way="every", workers=1)[0]
    assert all("kırmızı" not in said[w].lower() for w in said)


def test_the_hippocampus_answers_an_utterance_that_differs_in_two_places_from_every_one_lived(FastCortex):
    from living.analogy import Episodes
    people = ["Ali", "Veli", "Ayşe", "Can", "Ece", "Zeynep"]
    things = ["elma", "kitap", "top", "kalem", "çanta"]
    ep = Episodes()
    rng = random.Random(3)
    lived = set()
    for _ in range(60):
        p, q, t = rng.choice(people), rng.choice(people), rng.choice(things)
        lived.add((p, q, t))
        ep.live([p, "verdi", q, "'", "ya", t, "."], ["ver", "(", p, ",", q, ",", t, ")"])
    asked = [(p, q, t) for p in people for q in people for t in things if (p, q, t) not in lived][:30]
    right = 0
    for p, q, t in asked:
        plan, e, n = ep.plan([p, "verdi", q, "'", "ya", t, "."])
        right += plan == ["ver", "(", p, ",", q, ",", t, ")"]
    assert right == len(asked)
    i = len(ep) - 1                                                    # a dialogue taken back leaves no trace
    before = dict(ep.with_w)
    words = len(set(ep.said[i]))
    ep.forget(i)
    assert sum(before.values()) - sum(ep.with_w.values()) == words     # each of its words, once
    assert all(i not in ids for ids in ep.index.values())


def test_where_one_place_frames_cannot_answer_the_record_lets_the_analogy_lead(FastCortex):
    from living.cevahir import Cevahir
    people = ["Ali", "Veli", "Ayşe", "Can", "Ece", "Zeynep", "Deniz", "Selin"]
    lemma = {"elmayı": "elma", "kitabı": "kitap", "topu": "top", "kalemi": "kalem", "çantayı": "çanta"}
    rng = random.Random(1)
    seen, life = set(), []
    for _ in range(90):
        p, q = rng.sample(people, 2)
        t = rng.choice(sorted(lemma))
        seen.add((p, q, t))
        life.append(("%s %s %s'ya verdi ." % (p, t, q), "ver ( %s , %s , %s )" % (p, q, lemma[t])))
    asked = [("%s %s %s'ya verdi ." % (p, t, q), "ver ( %s , %s , %s )" % (p, q, lemma[t]))
             for p in people for q in people for t in sorted(lemma) if p != q and (p, q, t) not in seen][:12]
    c = Cevahir(variant="speaking")
    for q, a in life:
        c.watch(q, a)
    c.session.close_turn()
    said = c.answers([q for q, _ in asked], way="every", workers=2)
    exact = {w: sum(s[w] == a for s, (_, a) in zip(said, asked)) for w in ("zihin", "analogy", "cevahir")}
    assert exact["analogy"] == len(asked) and exact["cevahir"] == len(asked)    # written as the answers were
    assert exact["zihin"] < len(asked)
    assert c.trust_analogy() > 0.5 and all(s["thought"] == "analogy" for s in said)
