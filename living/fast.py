# -*- coding: utf-8 -*-
"""The cortex in C, for long lives: the same store and the same bits as living/cortex.py (M2).

``FastCortex`` runs ``living/_core.c``: escape "situation", exclusion, growth m = 1, full updates (or update
exclusion). Its bits are the same floating-point numbers as the Python cortex's, symbol for symbol; tests check it.
On top of the cortex it can run the episode expert and the harmony's verdict by the rule (MODEL.md, section 5).

The C source is compiled once with the system's C compiler into a cache folder outside the repository
(``$LIVING_CACHE`` or ``~/.cache/living``), named by the source's digest.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "_core.c")
_lib = None


def _library() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    with open(SOURCE, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    cache = os.environ.get("LIVING_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "living")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, "_core_%s.so" % digest)
    if not os.path.exists(path):
        cc = os.environ.get("CC", "cc")
        fd, tmp = tempfile.mkstemp(suffix=".so", dir=cache)
        os.close(fd)
        subprocess.run([cc, "-O2", "-shared", "-fPIC", "-o", tmp, SOURCE, "-lm"], check=True)
        os.replace(tmp, path)
    lib = ctypes.CDLL(path)
    lib.core_new.restype = ctypes.c_void_p
    lib.core_new.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.core_free.argtypes = [ctypes.c_void_p]
    lib.core_reset_history.argtypes = [ctypes.c_void_p]
    lib.core_set_history.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
    lib.core_stream.restype = ctypes.c_double
    lib.core_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p]
    lib.core_stats.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.core_recall.restype = ctypes.c_uint64
    lib.core_recall.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_double,
                                ctypes.c_void_p, ctypes.c_void_p]
    lib.core_enable_lines.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.core_set_episode_lengths.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.core_say.restype = ctypes.c_uint64
    lib.core_say.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_double,
                             ctypes.c_void_p, ctypes.c_void_p]
    lib.core_enable_mix.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int]
    lib.core_stream_mix.restype = ctypes.c_double
    lib.core_stream_mix.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p]
    _lib = lib
    return lib


def codes(text: str) -> np.ndarray:
    """The Unicode code points of a text."""
    return np.frombuffer(text.encode("utf-32-le", "surrogatepass"), dtype=np.uint32)


class FastCortex:
    """A cortex in C. ``episode=True`` adds the episode expert and the verdict by the rule; ``mix=True`` replaces the
    verdict with the harmony as binary context mixing (the cortex, every context and the episode, combined by a gate
    learned online with learning rate ``lr``)."""

    def __init__(self, depth: int = 8, update: str = "all", episode: bool = False, minlen: int = 12,
                 episode_bits: int = 22, mix: bool = False, lr: float = 0.02, lr_min: float = 0.002,
                 words: bool = True, two_inputs: bool = True, tree_prior: bool = True, recent: int = 0,
                 episode_lengths: tuple = (), lines: bool = False):
        if update not in ("all", "found"):
            raise ValueError("update is 'all' or 'found'")
        self.depth = depth
        self.update = update
        self.episode = episode
        self._lib = _library()
        self._c = self._lib.core_new(depth, 1 if update == "found" else 0, 1 if episode else 0, minlen, episode_bits)
        if episode_lengths:                          # several lengths, the longest first
            lens = (ctypes.c_int * len(episode_lengths))(*sorted(episode_lengths, reverse=True))
            self._lib.core_set_episode_lengths(self._c, lens, len(episode_lengths))
        self.mix = mix
        if mix:
            self._lib.core_enable_mix(self._c, lr, lr_min, 1 if words else 0, 1 if two_inputs else 0,
                                      1 if tree_prior else 0, recent)
            if lines:
                self._lib.core_enable_lines(self._c, episode_bits)

    def __del__(self):
        if getattr(self, "_c", None):
            self._lib.core_free(self._c)
            self._c = None

    def stream(self, text: str, history: str | None = None, learn: bool = True,
               bits: bool = False) -> tuple[float, np.ndarray | None]:
        """Predict each symbol of ``text`` and (if ``learn``) learn it; continue from the core's own history unless a
        ``history`` is given. Returns the code length in bits and, if asked, the bits of each symbol."""
        if history is not None:                       # its tail: contexts, the word state, the previous symbol
            h = codes(history[-max(self.depth, 256):])
            self._lib.core_set_history(self._c, h.ctypes.data, len(h))
        x = codes(text)
        out = np.empty(len(x), dtype=np.float64) if bits else None
        run = self._lib.core_stream_mix if self.mix else self._lib.core_stream
        total = run(self._c, x.ctypes.data, len(x), 1 if learn else 0, out.ctypes.data if bits else None)
        return total, out

    def say(self, history: str, most: int = 40, line: float = 0.0) -> tuple[str, list[float]]:
        """Continue ``history`` with the most probable symbols (the harmony only), while their probability reaches
        ``line`` (0: always), at most ``most``. Nothing is learned and the mind is left as it was. Returns what was
        said and the probability of each symbol said, plus that of the first one not said (if any)."""
        if not self.mix:
            raise ValueError("saying needs the harmony (mix=True)")
        h = codes(history[-max(self.depth, 256):])
        out = np.zeros(most, dtype=np.uint32)
        probs = np.zeros(most + 1, dtype=np.float64)
        n = self._lib.core_say(self._c, h.ctypes.data, len(h), most, line, out.ctypes.data, probs.ctypes.data)
        text = out[:n].tobytes().decode("utf-32-le")
        return text, [float(p) for p in probs[:min(n + 1, most)]]

    def recall(self, history: str, most: int = 400, line: float = 0.0) -> tuple[str, bool]:
        """Recall what began as the current line began: lock onto the lived line that began so and say it to its end;
        if none, go on as say() does. Returns what was said and whether it was a lived line (locked)."""
        if not self.mix:
            raise ValueError("recall needs the harmony (mix=True)")
        h = codes(history[-max(self.depth, 4096):])
        out = np.zeros(most, dtype=np.uint32)
        locked = ctypes.c_int(0)
        n = self._lib.core_recall(self._c, h.ctypes.data, len(h), most, line, out.ctypes.data, ctypes.byref(locked))
        return out[:n].tobytes().decode("utf-32-le"), bool(locked.value)

    def size(self) -> dict:
        s = (ctypes.c_uint64 * 6)()
        self._lib.core_stats(self._c, s)
        return {"nodes": s[0], "entries": s[1], "alphabet": s[2], "lived": s[3], "situations": s[4],
                "episode_text": s[5]}
