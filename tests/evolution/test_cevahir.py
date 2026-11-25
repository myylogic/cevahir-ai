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


def test_a_plan_is_said_to_its_end_though_a_token_of_it_was_never_written(FastCortex):
    """A plan by parts may count a place beyond any lived ("x _ 24" where answers went to 9): its token has no
    written form, and the plan is still said to its end, the token as it is held."""
    from living.cevahir import Cevahir
    c = Cevahir(variant="full")
    for n in range(1, 10):
        c.watch("kaç %d ?" % n, "x _ %d ." % n)
    c.session.close_turn()
    plan = ["x", "_", "24", "ve", "x", "_", "3"]
    said = c.speak("kaç 24 ?\t", plan, [1.0] * len(plan), "\t", most=400)
    assert c.session.words(said)[:len(plan)] == plan


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


def test_two_roles_that_swap_places_are_filled_at_once(FastCortex):
    from living.analogy import Episodes
    people = ["Ali", "Veli", "Ayşe", "Can", "Ece", "Zeynep"]
    ep = Episodes()
    for verb, lemma in (("koştu", "koş"), ("uyudu", "uyu")):      # each name's relation: the name itself
        for p in people:
            ep.live([p, verb, "."], [lemma, "(", p, ")"])
    ep.live(["Ali", "'", "nin", "babası", "Veli", "."], ["baba", "(", "Veli", ",", "Ali", ")"])   # the only neighbour
    plan, e, n = ep.plan(["Veli", "'", "nin", "babası", "Ali", "."])
    assert plan == ["baba", "(", "Ali", ",", "Veli", ")"]              # one place at a time gave "( Ali , Ali )"
    assert n == 1


def _nested(rng, depth, gave=False):
    """A small world of nested parts: "Ali saw the cup on the mat in the box ." and its answer, which names the places
    of what was said (a form like COGS's, its words and marks made up here)."""
    nouns = ["cat", "dog", "box", "cup", "mat", "bed", "car", "hat", "pen", "jar"]
    name, (verb, lemma) = rng.choice(["Ali", "Can", "Ece", "Deniz"]), rng.choice([("saw", "see"), ("took", "take")])
    if gave:
        verb, lemma = rng.choice([("gave", "give"), ("sold", "sell"), ("sent", "send")])
    things = [rng.choice(nouns) for _ in range(depth + 1)]
    preps = [rng.choice(["on", "in", "beside"]) for _ in range(depth)]
    said, places = [name, verb, "the", things[0]], [3]
    for p, n in zip(preps, things[1:]):
        said += [p, "the", n]
        places.append(len(said) - 1)
    answer = []
    for n, i in zip(things, places):
        answer += ["*", n, "(", "x", "_", str(i), ")", ";"]
    answer += [lemma, ".", "agent", "(", "x", "_", "1", ",", name, ")", "AND",
               lemma, ".", "theme", "(", "x", "_", "1", ",", "x", "_", "3", ")"]
    if gave:
        to = rng.choice(["Ali", "Can", "Ece", "Deniz"])
        said += ["to", to]
        answer += ["AND", lemma, ".", "recipient", "(", "x", "_", "1", ",", to, ")"]
    for k, p in enumerate(preps):
        answer += ["AND", things[k], ".", "nmod", ".", p, "(", "x", "_", str(places[k]), ",", "x", "_",
                   str(places[k + 1]), ")"]
    return said + ["."], answer


def test_words_stand_for_each_other_only_if_their_relations_play_alike_parts_in_answers(FastCortex):
    from living.analogy import Episodes
    rng = random.Random(4)
    ep = Episodes()
    for _ in range(200):
        ep.live(*_nested(rng, rng.choice([0, 1, 2]), gave=rng.random() < 0.3))
    assert ep.relation("to") == "recipient" and ep.relation("on") == "on"
    assert not ep.alike("recipient", "on")                            # after a verb, and after "nmod"
    assert ep.alike("on", "in") and ep.alike("cup", "mat") and ep.alike("see", "take")


def test_parts_answer_nesting_deeper_than_any_lived(FastCortex):
    from living.analogy import Episodes
    from living.evidence import SPEAKING_LINE
    rng = random.Random(0)
    ep = Episodes()
    for _ in range(400):
        ep.live(*_nested(rng, rng.choice([0, 1, 1, 2, 2])))               # one or two parts inside, never three
    asked = [_nested(rng, depth) for depth in (3, 4, 6) for _ in range(5)]
    right = 0
    for said, answer in asked:
        plan, e, n = ep.plan(said)
        right += plan == answer and e >= SPEAKING_LINE
        assert plan is None or e < SPEAKING_LINE or plan == answer          # silent, never wrong
        assert ep.plan(said, parts=False)[0] is None                       # no lived utterance is that long
    assert right >= 13


def test_a_part_that_cannot_be_taken_out_is_exchanged_for_one_lived_in_its_place(FastCortex):
    from living.analogy import Episodes
    from living.evidence import SPEAKING_LINE
    names, things = ["Ali", "Can", "Ece", "Deniz", "Selin", "Mert"], ["cup", "box", "pen", "hat", "jar", "map"]
    def saw(rng):                                    # "Ali saw Can ." or "Ali saw the cup ." (or took ... home)
        a, (verb, lemma, end) = rng.choice(names), rng.choice([("saw", "see", []), ("took", "take", ["home"])])
        if rng.random() < 0.5:
            n = rng.choice(names)
            return [a, verb, n] + end + ["."], [lemma, ".", "agent", "(", "x", "_", "1", ",", a, ")", "AND", lemma,
                                                 ".", "theme", "(", "x", "_", "1", ",", n, ")"]
        t = rng.choice(things)
        return [a, verb, "the", t] + end + ["."], ["*", t, "(", "x", "_", "3", ")", ";", lemma, ".", "agent", "(",
                                                    "x", "_", "1", ",", a, ")", "AND", lemma, ".", "theme", "(",
                                                    "x", "_", "1", ",", "x", "_", "3", ")"]
    def gave(rng, name):                                                 # a name is never given in what is lived
        a, r = rng.choice(names), rng.choice(names)
        head = ["give", ".", "agent", "(", "x", "_", "1", ",", a, ")", "AND", "give", ".", "theme", "(", "x", "_",
                "1", ","]
        tail = [")", "AND", "give", ".", "recipient", "(", "x", "_", "1", ",", r, ")"]
        if name:
            n = rng.choice(names)
            return [a, "gave", n, "to", r, "."], head + [n] + tail
        t = rng.choice(things)
        return [a, "gave", "the", t, "to", r, "."], ["*", t, "(", "x", "_", "3", ")", ";"] + head + \
            ["x", "_", "3"] + tail
    rng = random.Random(0)
    ep = Episodes()
    for _ in range(300):
        ep.live(*(saw(rng) if rng.random() < 0.5 else gave(rng, False)))
    right = 0
    for _ in range(10):
        said, answer = gave(rng, True)
        assert ep.plan(said, parts=False)[0] is None or ep.plan(said, parts=False)[1] < SPEAKING_LINE
        plan, e, n = ep.plan(said)
        assert plan is None or e < SPEAKING_LINE or plan == answer          # silent, never wrong
        right += plan == answer and e >= SPEAKING_LINE
    assert right >= 8                                                    # "the cup" for the name, and back
    # A change read in one place is carried to another because such changes carried between the places lived (a
    # name for "the cup" after "saw" and after "took"): the record of exchanges read elsewhere.
    ep.ready()
    shape = ("\u25a1",)
    assert ep.exchange_record(1, shape, 2, ("the", "\u25a1"), ("the", "cup"), elsewhere=True) >= SPEAKING_LINE
    # An exchange's rest is read as its record reads the rests of its examples (the one lived, or the analogy's own at
    # the speaking line), never by parts in turn: a rest answered by parts would be believed beyond that record.
    depths = []
    inner = ep._by_parts
    def counted(*args, **kw):
        depths.append(args[3])
        return inner(*args, **kw)
    ep._by_parts = counted
    said, answer = gave(rng, True)
    got = ep._by_exchange(tuple(said), {}, [100], 0)
    assert depths == [] and (got[0] is None or got[1] < SPEAKING_LINE or got[0] == answer)


def test_a_word_that_stands_twice_is_not_filled_where_only_one_of_it_changes(FastCortex):
    """"run and run" says nothing of "run and jump": which of its two I_RUN is the one that changes cannot be told, so
    that neighbour does not vote (it would say I_JUMP I_JUMP); the neighbours whose replaced words stand once do."""
    from living.analogy import Episodes
    ep = Episodes()
    for word, act in (("walk", "I_WALK"), ("look", "I_LOOK"), ("run", "I_RUN"), ("jump", "I_JUMP")):
        ep.live([word], [act])
    ep.live(["walk", "and", "look"], ["I_WALK", "I_LOOK"])
    ep.live(["look", "and", "walk"], ["I_LOOK", "I_WALK"])
    ep.live(["run", "and", "run"], ["I_RUN", "I_RUN"])
    plan, agree, filled, d = ep._plan(["run", "and", "jump"])
    assert plan == ["I_RUN", "I_JUMP"] and agree == filled == 2


def test_a_word_keeps_what_its_token_always_had_after_it_between_the_same_neighbours(FastCortex):
    """COGS's verb classes in a small world: of some verbs the one named is the one it happens to ("theme"), of others
    the one who does it ("agent"), and "hippo" is met only with the first kind. "a hippo smiled" filled from "a hippo
    rolled" would make the hippo a theme; but "smiled", said after a noun at the end, was always answered "smile .
    agent", so that neighbour gives nothing and those that keep "smiled" answer. Where a word was never said between
    such neighbours, nothing holds a plan back."""
    from living.analogy import Episodes
    from living.evidence import SPEAKING_LINE
    happens = {"rolled": "roll", "froze": "freeze", "broke": "break"}
    does = {"smiled": "smile", "slept": "sleep", "laughed": "laugh"}
    def say(n, v):
        lemma, role = (happens[v], "theme") if v in happens else (does[v], "agent")
        return ["a", n, v, "."], [n, "(", "x", "_", "1", ")", "AND", lemma, ".", role, "(", "x", "_", "2", ",", "x",
                                  "_", "1", ")"]
    ep = Episodes()
    for n in ["dog", "cat", "girl", "boy", "lion"]:
        for v in list(happens) + list(does):
            ep.live(*say(n, v))
    for _ in range(2):
        for v in happens:
            ep.live(*say("hippo", v))
    ep.ready()
    for v in does:
        said, answer = say("hippo", v)
        plan, e, _ = ep.plan(said)
        assert plan == answer and e >= SPEAKING_LINE
    assert ep._own("smiled", "□", ".") == {1: ".", 2: "agent"}
    assert ep._own("smiled", "was", ".") == {}
    # The end of an answer counts: in "run after walk twice" the second I_WALK stood before I_RUN, the first before
    # I_WALK, and nothing came after I_RUN; nothing always came after I_WALK there, so "look after walk twice" is filled
    # from "look after run twice" (counting only what came before the end, I_RUN would have been I_WALK's own).
    ep = Episodes()
    for w in ("walk", "run", "look", "jump", "sit", "swim", "sing", "stand"):
        ep.live([w], ["I_" + w.upper()])
    ep.live(["run", "after", "walk", "twice"], ["I_WALK", "I_WALK", "I_RUN"])
    ep.live(["look", "after", "run", "twice"], ["I_RUN", "I_RUN", "I_LOOK"])
    ep.ready()
    assert ep._own("walk", ep._shape("after"), ep._shape("twice")) == {}
    plan, agree, filled, d = ep._plan(["look", "after", "walk", "twice"])
    assert plan == ["I_WALK", "I_WALK", "I_LOOK"] and agree == filled == 2


def test_only_a_word_never_met_stands_for_itself(FastCortex):
    """A name's relation is the name itself, and a word never met before stands for itself where a name stood ("Zeki
    saw Can"). A word lived again and again that never brought a token of its own ("the": the mark that comes with it
    is in more than half of all answers, so it is no word's) is no name: "the saw Can" is not answered as if "the" saw
    Can (in COGS such rests, "The liked that ...", were answered so and spent the search for parts)."""
    from living.analogy import Episodes
    names, things, verbs = ["Ali", "Can", "Ece", "Deniz", "Selin", "Mert"], ["cup", "box", "pen", "hat"], \
        {"saw": "see", "took": "take"}
    def two(v, a, b):
        return [a, v, b, "."], [verbs[v], ".", "agent", "(", "x", "_", "1", ",", a, ")", "AND", verbs[v], ".", "theme",
                                "(", "x", "_", "1", ",", b, ")"]
    def the(v, a, t):
        return [a, v, "the", t, "."], ["*", t, "(", "x", "_", "3", ")", ";", verbs[v], ".", "agent", "(", "x", "_",
                                       "1", ",", a, ")", "AND", verbs[v], ".", "theme", "(", "x", "_", "1", ",", "x",
                                       "_", "3", ")"]
    ep = Episodes()
    for n in names:
        ep.live([n], [n])
    for a in names:
        for b in names[:3]:
            if a != b:
                ep.live(*two("saw", a, b))
    for v in verbs:
        for a in names:
            for t in things:
                ep.live(*the(v, a, t))
    ep.ready()
    assert ep.relation("the") is None and ep.with_w["the"] == 48
    assert ep._plan(["the", "saw", "Can", "."])[0] is None
    assert ep._plan(["Zeki", "saw", "Can", "."])[0] == two("saw", "Zeki", "Can")[1]
    # The order lived answers keep their lists in (MODEL.md 12.6): the items ";" ends first, then the items "AND"
    # joins, each by the first place it names. A plan put together by parts is said in that order.
    assert ep._list_order() == (";", "AND")
    lived = the("took", "Ece", "pen")[1]
    misplaced = lived[8:18] + ["AND"] + lived[:7] + [";"] + lived[19:]   # "take . agent ... AND * pen ... ; ..."
    assert misplaced != lived and ep._in_order(misplaced) == lived
    moves = Episodes()                                                   # answers that name no places: no lists
    for w in ("walk", "run", "look", "jump"):
        moves.live([w], ["I_" + w.upper()])
        moves.live([w, "twice"], ["I_" + w.upper()] * 2)
    assert moves._list_order() is None and moves._in_order(["I_RUN", "I_WALK"]) == ["I_RUN", "I_WALK"]


def test_a_word_inside_and_outside_a_part_lends_no_change_where_the_answer_names_no_places(FastCortex):
    """Taking "opposite right and look" out of "look opposite right and look opposite right twice": its answer's
    I_LOOK may be either look's, so the example lends no change; where an answer names the places of what was said
    (as COGS's do), a token is told apart by its place and the example still lends one."""
    from living.analogy import Episodes
    ep = Episodes()
    for word, act in (("look", "I_LOOK"), ("jump", "I_JUMP"), ("walk", "I_WALK"), ("run", "I_RUN")):
        ep.live([word], [act])
    assert ep.relation("look") == "I_LOOK"
    o = ("look", "opposite", "right", "and", "look", "opposite", "right", "twice")
    a2 = ["I_TURN_RIGHT", "I_TURN_RIGHT", "I_LOOK"] * 3
    a2_rest = ["I_TURN_RIGHT", "I_TURN_RIGHT", "I_LOOK"] * 2
    u = ("jump", "opposite", "right", "and", "look", "opposite", "right", "twice")
    rest = ["I_TURN_RIGHT", "I_TURN_RIGHT", "I_JUMP"] * 2
    assert ep._transplant(o, 1, 4, a2_rest, a2, u, 1, rest) is None
    named = ["look", "0"] + a2                                           # the same answer naming a place
    assert ep._transplant(o, 1, 4, ["look", "0"] + a2_rest, named, u, 1, ["jump", "0"] + rest) is not None


def test_an_example_whose_part_shares_a_word_with_the_part_put_in_lends_no_change(FastCortex):
    """Exchanging "run" for "look opposite" to answer "run left after jump left": the example "look left after run",
    exchanged the same way, shows a turn fewer and its I_LOOK kept, since its "look" stays; put into the asked rest,
    that leaves "look"'s token where "run"'s belongs. An example whose part shares no word with the part put in shows
    both changes (SCAN's add jump: Cevahir said I_TURN_LEFT I_JUMP I_TURN_LEFT I_LOOK)."""
    from living.analogy import Episodes
    ep = Episodes()
    for word, act in (("walk", "I_WALK"), ("look", "I_LOOK"), ("run", "I_RUN"), ("jump", "I_JUMP")):
        ep.live([word], [act])
    u, y = ("run", "left", "after", "jump", "left"), ("look", "opposite")
    rest = ["I_TURN_LEFT", "I_JUMP", "I_TURN_LEFT", "I_TURN_LEFT", "I_LOOK"]     # look opposite left after jump left
    a2_rest = ["I_RUN", "I_TURN_LEFT", "I_TURN_LEFT", "I_LOOK"]                  # look opposite left after run
    shared, apart = ("look", "left", "after", "run"), ("walk", "left", "after", "run")
    lent = ep._transplant(shared, 0, 1, a2_rest, ["I_RUN", "I_TURN_LEFT", "I_LOOK"], u, 0, rest, instead=y)
    assert lent == ["I_TURN_LEFT", "I_JUMP", "I_TURN_LEFT", "I_LOOK"] and not ep._apart(shared[:1], y)
    lent = ep._transplant(apart, 0, 1, a2_rest, ["I_RUN", "I_TURN_LEFT", "I_WALK"], u, 0, rest, instead=y)
    assert lent == ["I_TURN_LEFT", "I_JUMP", "I_TURN_LEFT", "I_RUN"] and ep._apart(apart[:1], y)
    assert not ep._apart(("jump",), ("jump", "opposite"))               # asked: a part holding the asked one


def _phrased(rng, on_subject=0, on_object=0):
    """A small world of phrases: "the cup on the mat gave the box in the jar to the cat ." and its answer, which names
    the places of what was said and keeps its items in the order of the places they name (a form like COGS's, its
    words and marks made up here); no noun is said twice in a sentence."""
    nouns = rng.sample(["cat", "dog", "box", "cup", "mat", "bed", "car", "hat", "pen", "jar", "map", "toy"], 8)
    verb, lemma, gives = rng.choice([("saw", "see", False), ("took", "take", False), ("gave", "give", True),
                                     ("sent", "send", True)])
    said, ended, joined = [], [], []
    def noun_phrase(k):
        said.extend(["the", nouns.pop()])
        head = last = len(said) - 1
        ended.append((head, ["*", said[head], "(", "x", "_", str(head), ")"]))
        for _ in range(k):
            p = rng.choice(["on", "in", "beside"])
            said.extend([p, "the", nouns.pop()])
            n = len(said) - 1
            ended.append((n, ["*", said[n], "(", "x", "_", str(n), ")"]))
            joined.append(((last, n), [said[last], ".", "nmod", ".", p, "(", "x", "_", str(last), ",", "x", "_",
                                       str(n), ")"]))
            last = n
        return head
    subj = noun_phrase(on_subject)
    said.append(verb)
    v = len(said) - 1
    obj = noun_phrase(on_object)
    joined.append(((v, subj), [lemma, ".", "agent", "(", "x", "_", str(v), ",", "x", "_", str(subj), ")"]))
    joined.append(((v, obj), [lemma, ".", "theme", "(", "x", "_", str(v), ",", "x", "_", str(obj), ")"]))
    if gives:
        said.extend(["to", "the", nouns.pop()])
        r = len(said) - 1
        ended.append((r, ["*", said[r], "(", "x", "_", str(r), ")"]))
        joined.append(((v, r), [lemma, ".", "recipient", "(", "x", "_", str(v), ",", "x", "_", str(r), ")"]))
    said.append(".")
    answer = []
    for _, it in sorted(ended, key=lambda x: x[0]):
        answer += it + [";"]
    for k, (_, it) in enumerate(sorted(joined, key=lambda x: x[0])):
        answer += (["AND"] if k else []) + it
    return said, answer


def test_a_phrase_on_a_subject_is_answered_from_phrases_lived_on_objects(FastCortex):
    """Phrases lived only on objects; asked on a subject ("the cup on the mat saw the box ."), the first search is
    silent: the phrase's frame is one no lived part was a whole in, and no lived phrase stood in its surroundings. The
    second reads every word by its kind ("on" and "mat" by the "(" their tokens have after them, "saw" by its "."),
    takes the phrase out as one never lived there, answers the rest, and puts the phrase's change back as the items a
    phrase on an object adds, read from phrases in other surroundings, the answer said in the order of its lists."""
    from living.analogy import Episodes
    from living.evidence import SPEAKING_LINE
    rng = random.Random(0)
    ep = Episodes()
    for _ in range(600):
        ep.live(*_phrased(rng, 0, rng.choice([0, 1])))                     # phrases only on objects
    ep.ready()
    assert ep._kind("on") == ep._kind("mat") == "□(" and ep._kind("saw") == "□."
    asked = [_phrased(rng, 1, rng.choice([0, 1])) for _ in range(12)]
    right = 0
    for said, answer in asked:
        plan, e, n = ep.plan(said)
        right += plan == answer and e >= SPEAKING_LINE
        assert plan is None or e < SPEAKING_LINE or plan == answer          # silent, never wrong
        assert ep._by_parts(tuple(said), {}, [4000], 0)[0] is None          # the first search alone is silent
    assert right >= 10


def test_an_answer_is_said_in_the_order_of_the_places_its_items_name(FastCortex):
    """Every lived answer is given back from its items in any order: the ended items first, each item by the places it
    names, its numbers and the names in it (a word whose token is the word itself, said once), a name read as the
    place where it was said ("see . agent ( x _ 1 , Ali )" before "see . theme ( x _ 1 , Can )")."""
    from living.analogy import Episodes
    rng = random.Random(1)
    lived = [_phrased(rng, 0, rng.choice([0, 1])) for _ in range(200)]
    names = ["Ali", "Can", "Ece", "Deniz"]
    for a in names:
        for b in names:
            if a != b:
                lived.append(([a, "saw", b, "."], ["see", ".", "agent", "(", "x", "_", "1", ",", a, ")", "AND", "see",
                                                   ".", "theme", "(", "x", "_", "1", ",", b, ")"]))
    ep = Episodes()
    for n in names:
        ep.live([n], [n])
    for said, answer in lived:
        ep.live(said, answer)
    assert ep._list_order() == (";", "AND")
    for said, answer in lived:
        items, cur = [], []
        for t in answer:
            if t in (";", "AND"):
                items.append((cur, t))
                cur = []
            else:
                cur.append(t)
        items.append((cur, None))
        rng.shuffle(items)
        ended = [it for it, sep in items if sep == ";"]
        joined = [it for it, sep in items if sep != ";"]
        shuffled = [t for it in ended for t in it + [";"]] + \
            [t for k, it in enumerate(joined) for t in (["AND"] if k else []) + it]
        assert ep._in_order(shuffled, said) == answer


def _moves():
    """Commands of a small grammar of moves, built as SCAN builds them: an action, a way (around: four turns and acts;
    opposite: two turns and an act) and a side, said once, twice or thrice, and two of them joined by "and"."""
    acts = {"walk": "I_WALK", "run": "I_RUN", "jump": "I_JUMP", "look": "I_LOOK"}
    turns = {"left": "I_TURN_LEFT", "right": "I_TURN_RIGHT"}
    units = []
    for a, act in acts.items():
        units.append(([a], [act]))
        for side, turn in turns.items():
            units += [([a, side], [turn, act]), ([a, "opposite", side], [turn, turn, act]),
                      ([a, "around", side], [turn, act] * 4)]
    said = [(u + ([w] if w else []), ans * k) for u, ans in units for w, k in (("", 1), ("twice", 2), ("thrice", 3))]
    pairs = [(a[0] + ["and"] + b[0], a[1] + b[1]) for a in said for b in said]
    random.Random(0).shuffle(pairs)
    return said, pairs


def test_parts_stay_silent_where_a_change_depends_on_what_stands_around_it(FastCortex):
    """What "around" changes depends on what stands around it ("around right thrice" turns twelve times), and a part
    whose words have no relation of their own ("right") brings a change that depends on its place: answering by
    parts is silent there, never wrong (SCAN's around-right and length splits, in a small world)."""
    from living.analogy import Episodes
    from living.evidence import SPEAKING_LINE
    said, pairs = _moves()
    held = lambda s: any(s[i:i + 2] == ["around", "right"] for i in range(len(s) - 1))
    splits = [([d for d in said + pairs if not held(d[0])][:3000], [d for d in pairs if held(d[0])][:30]),
              ([d for d in said + pairs if len(d[1]) <= 22][:4000], [d for d in said + pairs if len(d[1]) >= 24][:30])]
    for lived, asked in splits:
        ep = Episodes()
        for s, a in lived:
            ep.live(s, a)
        ep.ready()
        for s, a in asked:
            plan, e, _ = ep.plan(s)
            assert plan is None or e < SPEAKING_LINE or plan == a                # silent, never wrong


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
