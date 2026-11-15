# -*- coding: utf-8 -*-
"""Hearing and recognising: generated voices, a few named, new ones recognised by resemblance (F18).

    python research/living_mind_2026_11_14/t_hearing.py --seeds 0 1 2 --taught 3 --tested 10

Three voices ("mehmet" low, "ayşe" middle, "zeynep" high: harmonic sounds with different ranges of the fundamental and
different formant-like spectral envelopes) at random pitch within the range, with a pitch glide, vibrato and wander,
random formant scale, spectral tilt, duration, loudness, leading and trailing silence and a little noise. The mind
hears `taught` sounds of each voice named in watched dialogues ("bu kimin sesi ?" -> "bu <ad> ."), twice. Then for
`tested` new sounds of each voice it is asked "bu kimin sesi" in a chat; right if its answer contains the name.
Control: the same life with resemblance off (F18 disabled). Sounds are synthesised with numpy and written as WAV files;
the core never sees samples, only the adapter's features (isitme.py). Each row also says how well the facets alone tell
the voices apart, measured with the core's own facet similarity at the end of the life: the mean similarity of every
new sound to the named sounds of its own voice and of the other voices, and how often its most similar named sound is
of its own voice.
"""
import argparse, json, os, random, sys, tempfile, wave
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from zihin import Mind, Config  # noqa: E402
from zihin.__main__ import Session, CHANNEL  # noqa: E402
from zihin.isitme import EAR, features  # noqa: E402
from zihin.learn import facet_similarity  # noqa: E402

RATE = 16000
VOICES = {"mehmet": ((95.0, 145.0), (650.0, 1100.0, 2450.0)),     # range of the fundamental, formants (Hz)
          "ayşe": ((180.0, 245.0), (500.0, 1850.0, 2750.0)),
          "zeynep": ((235.0, 320.0), (380.0, 2350.0, 3150.0))}


def speak(name, rng, path):
    """One utterance-like sound of a voice: a harmonic source shaped by three resonances, as a 16-bit WAV file."""
    (lo, hi), formants = VOICES[name]
    noise = np.random.default_rng(rng.getrandbits(32))
    dur = rng.uniform(0.6, 2.0)
    n = int(dur * RATE)
    t = np.arange(n) / RATE
    base = lo * (hi / lo) ** rng.random()
    wander = np.cumsum(noise.normal(0.0, 0.05, int(dur * 100) + 2))           # semitones, 100 steps a second
    semis = (rng.uniform(-2.0, 2.0) * t / dur                                     # glide
             + rng.uniform(0.1, 0.4) * np.sin(2 * np.pi * rng.uniform(4.0, 7.0) * t + rng.uniform(0, 2 * np.pi))
             + np.interp(t, np.arange(len(wander)) / 100.0, wander - wander.mean()))
    f0 = base * 2.0 ** (semis / 12.0)
    phase = 2 * np.pi * np.cumsum(f0) / RATE
    scale = rng.uniform(0.94, 1.06)
    peaks = [f * scale * rng.uniform(0.97, 1.03) for f in formants]
    widths = [b * rng.uniform(0.8, 1.25) for b in (80.0, 100.0, 150.0)]
    tilt = rng.uniform(0.8, 1.3)
    y = np.zeros(n)
    for k in range(1, int(0.45 * RATE / f0.min()) + 1):
        f = k * f0
        gain = k ** -tilt
        for p, b in zip(peaks, widths):
            gain = gain * p * p / np.sqrt((p * p - f * f) ** 2 + (b * f) ** 2)
        y += np.where(f < 0.45 * RATE, gain, 0.0) * np.sin(k * phase)
    y /= np.abs(y).max()
    attack, release = rng.uniform(0.02, 0.06), rng.uniform(0.05, 0.15)
    envelope = np.clip(np.minimum(t / attack, (dur - t) / release), 0.0, 1.0)
    depth, syllables = rng.uniform(0.0, 0.5), rng.uniform(3.0, 6.0)
    envelope *= 1.0 - depth * (0.5 - 0.5 * np.cos(2 * np.pi * syllables * t + rng.uniform(0, 2 * np.pi)))
    y *= envelope
    level = np.sqrt((y ** 2).mean())
    y = np.concatenate([np.zeros(int(rng.uniform(0.05, 0.5) * RATE)), y, np.zeros(int(rng.uniform(0.05, 0.5) * RATE))])
    y += noise.normal(0.0, level * 10.0 ** (-rng.uniform(30.0, 45.0) / 20.0), len(y))
    y *= rng.uniform(0.05, 0.9) / np.abs(y).max()
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(np.round(y * 32767).astype("<i2").tobytes())


def separation(mind, named, new):
    """How well the facets alone tell the voices apart: (name, gist) pairs of the named and the new sounds."""
    named = [(name, mind.unit_of(EAR, gist)) for name, gist in dict.fromkeys(named)]
    within, across, nearest = [], [], 0
    for name, gist in new:
        unit = mind.unit_of(EAR, gist)
        sims = [(facet_similarity(mind, unit, other), voice) for voice, other in named if other != unit]
        within += [s for s, voice in sims if voice == name]
        across += [s for s, voice in sims if voice != name]
        nearest += max(sims)[1] == name
    return {"sim_within": round(float(np.mean(within)), 3), "sim_across": round(float(np.mean(across)), 3),
            "nearest_right": nearest / float(len(new))}


def life(seed, taught, tested, resemblance, folder):
    rng = random.Random(seed)
    cfg = Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                         backoff=True, resemblance=resemblance, race=True)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    lessons = []
    for name in VOICES:
        for i in range(taught):
            p = os.path.join(folder, "t_%d_%s_%d.wav" % (seed, name, i))
            speak(name, rng, p)
            lessons.append((name, features(p)))
    for _ in range(2):
        rng.shuffle(lessons)
        for name, heard in lessons:
            s.hearing = heard
            s.watch("bu kimin sesi ?", "bu %s ." % name)
    right, answers, new = 0, [], []
    for name in VOICES:
        for i in range(tested):
            p = os.path.join(folder, "n_%d_%s_%d.wav" % (seed, name, i))
            speak(name, rng, p)
            s.hearing = features(p)
            new.append((name, s.hearing[0]))
            said = s.reply("bu kimin sesi")
            s.close_turn()
            ok = name in said.split()
            right += ok
            answers.append((name, said))
    named = [(name, heard[0]) for name, heard in lessons]
    facts = separation(mind, named, new)
    facts["distinct_gists"] = len({g for _, g in named + new})
    return right / float(tested * len(VOICES)), answers[:6], facts


def main():
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--taught", type=int, default=3)
    ap.add_argument("--tested", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "t_hearing.jsonl"))
    args = ap.parse_args()
    folder = tempfile.mkdtemp()
    for seed in args.seeds:
        for resemblance in (True, False):
            acc, sample, facts = life(seed, args.taught, args.tested, resemblance, folder)
            row = {"seed": seed, "resemblance": resemblance, "taught_per_kind": args.taught,
                   "tested_per_kind": args.tested, "accuracy": acc, "sample": sample}
            row.update(facts)
            with open(args.out, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
