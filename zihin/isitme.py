# -*- coding: utf-8 -*-
"""Hearing: a sound as a perceived thing with its aspects (vision 10: any sense; an adapter, no core code).

No trained model: plain arithmetic on the samples of a WAV file (PCM 8, 16, 24 or 32 bit, any sample rate, channels
mixed to mono, at most the first 10 seconds). Only frames within 30 dB of the loudest one are described, so leading
and trailing silence and the overall loudness do not matter. A sound becomes
  * one item on the sense channel "kulak": its gist (a 64-bit difference hash of its band energies over time: 8
    mel-like bands x 9 time segments, a bit says whether a band grew louder from one segment to the next), the same
    recording is the same thing;
  * facets on the channel "ses" (they accompany the item, they are not part of a sentence):
      - pitch: the median fundamental of the voiced frames (autocorrelation), in bands of 3 semitones,
      - spectral centroid and roll-off (85 % of the energy), in third-octave bands,
      - the share of the energy in each of 8 mel-like bands (position-specific, in 6 dB steps) and the two
        strongest bands (position-free),
      - zero-crossing rate (as a frequency), voiced fraction and duration, in coarse bands.
    Every graded measure is quantised on two grids offset by half a band: values less than half a band apart share
    at least one symbol, values a band or more apart share none.
Two recordings of the same voice or the same kind of sound share many facets; the core's facet similarity (F18) lets a
new sound join the categories of the sounds it resembles, and lets the most similar members speak loudest.
"""
from __future__ import annotations

import wave
from typing import List, Tuple

import numpy as np

EAR, SOUND = "kulak", "ses"
LIMIT = 10.0                     # seconds analysed at most
FRAME, HOP = 0.040, 0.020        # analysis frame and hop (s): a frame holds two periods of the lowest pitch
GATE = 30.0                      # frames quieter than the loudest by more than this (dB) are silence
LOW, HIGH = 60.0, 8000.0         # band range (Hz)
BANDS = 8
PITCH = (50.0, 1000.0)           # pitch search range (Hz)
VOICED = 0.5                     # autocorrelation peak above which a frame is voiced


def read(path: str, limit: float = LIMIT) -> Tuple[np.ndarray, int]:
    """Mono samples in [-1, 1] and the sample rate of a PCM WAV file (8-bit unsigned, 16/24/32-bit signed)."""
    with wave.open(path, "rb") as w:
        channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(min(w.getnframes(), int(rate * limit)))
    data = np.frombuffer(raw, dtype=np.uint8)
    if width == 1:
        x = data.astype(np.float64) - 128.0
    elif width == 2:
        x = data.view("<i2").astype(np.float64)
    elif width == 3:
        b = data.reshape(-1, 3).astype(np.int64)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(v >= 1 << 23, v - (1 << 24), v).astype(np.float64)
    elif width == 4:
        x = data.view("<i4").astype(np.float64)
    else:
        raise ValueError("unsupported sample width: %d bytes" % width)
    x /= float(1 << (8 * width - 1))
    n = len(x) // channels * channels
    return x[:n].reshape(-1, channels).mean(axis=1), rate


def _grid(name: str, x: float) -> List[str]:
    """A measure in half-band units on two grids offset by half a band (even values on one, odd on the other)."""
    return ["%s=%d" % (name, 2 * int(np.floor(x / 2.0))), "%s=%d" % (name, 2 * int(np.floor((x + 1.0) / 2.0)) - 1)]


def _mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _filters(freqs: np.ndarray) -> np.ndarray:
    """(frequencies x BANDS) overlapping triangles spaced evenly on the mel scale between LOW and HIGH."""
    edges = np.linspace(_mel(LOW), _mel(HIGH), BANDS + 2)
    m = _mel(freqs)[:, None]
    rise = (m - edges[None, :-2]) / (edges[1:-1] - edges[:-2])[None, :]
    fall = (edges[None, 2:] - m) / (edges[2:] - edges[1:-1])[None, :]
    return np.clip(np.minimum(rise, fall), 0.0, None)


def _pitch(r: np.ndarray, rate: int, lo: int, hi: int) -> Tuple[float, float]:
    """Fundamental (Hz) and periodicity of one frame from its normalised autocorrelation ``r`` (lags lo..hi): the
    first peak nearly as high as the highest (so not an octave below), refined by a parabola."""
    seg = r[lo - 1:hi + 2]
    peaks = np.flatnonzero((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:])) + lo
    if len(peaks) == 0:
        return 0.0, 0.0
    height = r[peaks]
    tau = int(peaks[np.argmax(height >= 0.85 * height.max())])
    a, b, c = r[tau - 1], r[tau], r[tau + 1]
    bend = a - 2.0 * b + c
    shift = 0.5 * (a - c) / bend if bend < 0.0 else 0.0
    return rate / (tau + shift), float(b)


def features(path: str) -> Tuple[str, List[str]]:
    """(gist symbol, facet symbols) of a WAV file."""
    x, rate = read(path)
    size, hop = max(4, int(round(FRAME * rate))), max(1, int(round(HOP * rate)))
    if len(x) < size:
        x = np.concatenate([x, np.zeros(size - len(x))])
    frames = np.lib.stride_tricks.sliding_window_view(x, size)[::hop]
    frames = frames - frames.mean(axis=1, keepdims=True)
    energy = (frames ** 2).mean(axis=1)
    if energy.max() <= 1e-12:                             # digital silence
        return "k%016x" % 0, ["sessiz"]
    active = energy >= energy.max() * 10.0 ** (-GATE / 10.0)
    idx = np.flatnonzero(active)
    first, last = int(idx[0]), int(idx[-1])
    n_fft = 1 << int(np.ceil(np.log2(2 * size)))          # twice the frame: the autocorrelation does not wrap
    window = np.hanning(size)
    power = np.abs(np.fft.rfft(frames * window, n_fft, axis=1)) ** 2
    freqs = np.arange(power.shape[1]) * rate / float(n_fft)
    bank = _filters(freqs)
    weight = energy[active] / energy[active].sum()        # loud frames say more than the noise between them

    # pitch and voicing: autocorrelation of each active frame, divided by that of the window (Boersma 1993)
    lags = np.fft.irfft(power[active], n_fft, axis=1)[:, :size]
    own = np.fft.irfft(np.abs(np.fft.rfft(window, n_fft)) ** 2, n_fft)[:size]
    lo = max(2, int(np.floor(rate / PITCH[1])))
    hi = min(size // 2, int(np.ceil(rate / PITCH[0])))
    r = (lags[:, :hi + 2] / np.maximum(lags[:, :1], 1e-300)) / (own[:hi + 2] / own[0])[None, :]
    found = [_pitch(row, rate, lo, hi) for row in r] if hi > lo + 2 else [(0.0, 0.0)] * len(r)
    voiced = np.array([strength > VOICED for _, strength in found])
    facets: List[str] = []
    if voiced.sum() >= max(2, 0.2 * len(found)):
        f0 = float(np.median([f for (f, _), v in zip(found, voiced) if v]))
        facets += _grid("perde", 12.0 * np.log2(f0 / 55.0) / 1.5)
    else:
        facets.append("perde=yok")
    facets += _grid("otumlu", float(weight[voiced].sum()) / 0.125 + 0.5)    # bounded: its ends inside a band

    # spectral envelope: the summed spectrum of the active frames (each counts with its energy), within the band range
    spectrum = power[active].sum(axis=0)
    inside = (freqs >= LOW) & (freqs <= HIGH)
    part = spectrum[inside]
    total = part.sum()
    if total > 0.0:
        centroid = float((freqs[inside] * part).sum() / total)
        rolloff = float(freqs[inside][np.searchsorted(np.cumsum(part), 0.85 * total)])
        facets += _grid("merkez", 6.0 * np.log2(max(centroid, LOW) / 100.0))
        facets += _grid("sinir", 6.0 * np.log2(max(rolloff, LOW) / 100.0))
    bands = spectrum @ bank
    share = bands / max(bands.sum(), 1e-300)
    level = np.clip(10.0 * np.log10(np.maximum(share, 1e-12)), -30.0, 0.0)
    for b in range(BANDS):
        facets += _grid("b%d" % b, (level[b] + 30.0) / 3.0 + 0.5)
    facets += ["tepe=%d" % b for b in sorted(np.argsort(-share, kind="stable")[:2].tolist())]

    sign = np.signbit(frames[active])
    crossings = (sign[:, 1:] != sign[:, :-1]).mean(axis=1)
    zcr = float((crossings * weight).sum()) * rate / 2.0  # as a frequency: a pure tone of f Hz gives f
    facets += _grid("gecis", 4.0 * np.log2(max(zcr, 1.0) / 50.0))
    duration = (last - first) * hop / float(rate) + size / float(rate)
    facets += _grid("sure", 4.0 * np.log2(duration / 0.1))

    # gist: do the band energies of the heard span grow from one ninth of it to the next?
    span = power[first:last + 1] @ bank
    bounds = np.linspace(0, len(span), 10)
    grid = np.array([span[min(int(bounds[s]), len(span) - 1):max(int(bounds[s]) + 1, int(bounds[s + 1]))].mean(axis=0)
                     for s in range(9)])
    value = 0
    for b in range(BANDS):
        for s in range(8):
            value = (value << 1) | int(grid[s + 1, b] > grid[s, b])
    return "k%016x" % value, facets
