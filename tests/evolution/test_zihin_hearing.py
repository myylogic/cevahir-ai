# -*- coding: utf-8 -*-
"""Zihin, the hearing adapter: a WAV file as a heard thing with its facets, and the session command that plays it.

What is pinned here
-------------------
* Features are deterministic: the same file, or a byte-identical copy, gives the same gist and facets; another sound
  gives another gist.
* PCM of 8 (unsigned), 16, 24 and 32 bits, any sample rate, mono or stereo (mixed to mono, not the first channel)
  describe the same tone alike: the same pitch band, nearly all facets shared.
* Loudness and leading silence do not change the facets; only the first 10 seconds are heard; silence is "sessiz".
* Two very different sounds (a low harmonic tone and a short noise burst) share few facets; two close tones share many.
* ``!dinle <wav>`` (alias ``!hear``) sets the session's sound, the next utterance perceives it first (the item on
  "kulak", its facets on "ses") and clears it; a view and a sound set together are perceived in the same moment.

Sounds are synthesised with numpy into pytest's temporary folder; nothing else is written.
"""
from __future__ import annotations

import re
import wave

import numpy as np
import pytest
import torch

from zihin import TOLD, Config, Mind
from zihin.__main__ import CHANNEL, Session, _one_line
from zihin.gorme import EYE, LOOK
from zihin.isitme import EAR, SOUND, features


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def save(path, samples, rate=16000, width=2) -> str:
    """Write samples in [-1, 1] ((n,) or (n, channels)) as PCM WAV; 8-bit is unsigned as the format requires."""
    y = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    channels = 1 if y.ndim == 1 else y.shape[1]
    scale = float((1 << (8 * width - 1)) - 1)
    v = np.round(y.reshape(-1) * scale).astype(np.int64)
    if width == 1:
        data = (v + 128).astype(np.uint8).tobytes()
    elif width == 3:
        v &= 0xFFFFFF
        data = np.stack([v & 255, (v >> 8) & 255, (v >> 16) & 255], axis=1).astype(np.uint8).tobytes()
    else:
        data = v.astype("<i2" if width == 2 else "<i4").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(data)
    return str(path)


def harmonic(f0, seconds, rate=16000, pad=0.2):
    """A steady harmonic tone (8 partials, amplitude 1/k) with ``pad`` seconds of silence on both sides."""
    t = np.arange(int(rate * seconds)) / rate
    y = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 9) if k * f0 < rate / 2)
    y = 0.8 * y / np.abs(y).max()
    return np.concatenate([np.zeros(int(pad * rate)), y, np.zeros(int(pad * rate))])


def overlap(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / float(len(a | b))


def test_features_are_deterministic_and_the_same_file_gives_the_same_gist(tmp_path):
    rng = np.random.default_rng(3)
    y = harmonic(140, 0.9) * (1.0 + 0.3 * np.sin(np.linspace(0, 9, int(1.3 * 16000))))
    path = save(tmp_path / "a.wav", 0.7 * y + rng.normal(0, 0.005, len(y)))
    first = features(path)
    assert features(path) == first
    copy = tmp_path / "copy.wav"
    copy.write_bytes((tmp_path / "a.wav").read_bytes())
    assert features(str(copy)) == first
    assert re.fullmatch(r"k[0-9a-f]{16}", first[0])
    assert features(save(tmp_path / "b.wav", harmonic(260, 0.6)))[0] != first[0]


@pytest.mark.parametrize("rate, width, channels", [(8000, 1, 2), (16000, 2, 1), (22050, 3, 2), (44100, 4, 1),
                                                   (48000, 2, 2)])
def test_any_pcm_width_rate_and_channel_count(tmp_path, rate, width, channels):
    reference = features(save(tmp_path / "ref.wav", harmonic(193, 1.0)))[1]   # 193 Hz: the middle of a pitch band
    y = harmonic(193, 1.0, rate)
    if channels == 2:                       # the channels differ, their mean is the tone
        other = 0.3 * np.sin(2 * np.pi * 1000.0 * np.arange(len(y)) / rate)
        y = np.stack([0.7 * y + other, 0.7 * y - other], axis=1)
    got = features(save(tmp_path / "x.wav", y, rate, width))[1]
    assert [f for f in got if f.startswith("perde")] == ["perde=14", "perde=13"]    # 8 log2(193 / 55) = 14.49
    assert overlap(got, reference) >= 0.9


def test_loudness_and_leading_silence_do_not_change_the_facets(tmp_path):
    y = harmonic(150, 0.8)
    loud = features(save(tmp_path / "loud.wav", y))
    later = np.concatenate([np.zeros(4800), 0.25 * y])        # 0.3 s: a whole number of analysis hops
    assert features(save(tmp_path / "quiet_later.wav", later))[1] == loud[1]


def test_only_the_first_ten_seconds_are_heard(tmp_path):
    rng = np.random.default_rng(0)
    y = harmonic(180, 12.0, pad=0.0)
    y[160000:] = rng.normal(0.0, 0.3, len(y) - 160000)         # what comes after 10 s would change every measure
    assert features(save(tmp_path / "long.wav", y)) == features(save(tmp_path / "ten.wav", y[:160000]))


def test_silence_is_heard_as_silence(tmp_path):
    assert features(save(tmp_path / "silent.wav", np.zeros(8000))) == ("k" + "0" * 16, ["sessiz"])
    assert features(save(tmp_path / "empty.wav", np.zeros(0))) == ("k" + "0" * 16, ["sessiz"])


def test_very_different_sounds_share_few_facets(tmp_path):
    rng = np.random.default_rng(0)
    tone = features(save(tmp_path / "tone.wav", harmonic(110, 1.0)))[1]
    burst = np.concatenate([np.zeros(3200), rng.normal(0.0, 0.15, 3200), np.zeros(3200)])
    noise = features(save(tmp_path / "noise.wav", burst))[1]
    close = features(save(tmp_path / "close.wav", harmonic(113, 1.3)))[1]
    assert overlap(tone, noise) <= 0.1
    assert overlap(tone, close) >= 0.6


def new_session() -> Session:
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True))
    mind.add_channel(CHANNEL, "act")
    return Session(mind)


def record(mind) -> list:
    """The events of every moment the mind lives from now on, in order."""
    seen, step = [], mind.step

    def recorded(events=()):
        events = list(events)
        seen.append(events)
        return step(events)

    mind.step = recorded
    return seen


@pytest.mark.parametrize("command", ["dinle", "hear"])
def test_the_session_command_sets_the_sound_and_the_next_utterance_hears_it_first(tmp_path, command):
    path = save(tmp_path / "ses.wav", harmonic(200, 0.7))
    session = new_session()
    mind = session.mind
    assert mind.ch.kind(mind.ch.id(EAR)) == "sense" and mind.ch.kind(mind.ch.id(SOUND)) == "facet"
    _one_line(session, "!%s %s" % (command, path), False, None, None, 0)
    assert session.hearing == features(path)
    gist, facets = session.hearing
    seen = record(mind)
    session.live("bu ses", TOLD)
    assert {(e.channel, e.symbol) for e in seen[0]} == {(EAR, gist)} | {(SOUND, f) for f in facets}
    assert [e.symbol for e in seen[1] if e.channel == CHANNEL] == ["bu"]
    assert session.hearing is None and mind.unit_of(EAR, gist) >= 0


def test_a_view_and_a_sound_are_perceived_in_the_same_first_moment():
    session = new_session()
    session.view = ("g1", ["renk=ton0"])
    session.hearing = ("k1", ["perde=8"])
    seen = record(session.mind)
    session.reply("bu ne")
    assert {(e.channel, e.symbol) for e in seen[0]} == {(EYE, "g1"), (LOOK, "renk=ton0"), (EAR, "k1"),
                                                        (SOUND, "perde=8")}
    assert session.view is None and session.hearing is None
    assert all(e.channel not in (EYE, LOOK, EAR, SOUND) for events in seen[1:] for e in events)
