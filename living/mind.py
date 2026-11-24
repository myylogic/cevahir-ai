# -*- coding: utf-8 -*-
"""Channels: the mind's own alphabets, new channels, watching and speaking (MODEL.md, section 4).

A channel is an origin of streams. It has its own cortex, alphabet and lexicon. Its symbols are whatever it delivers:
the characters of a text, or the atoms of a channel no one has named. The mind can be told which channel a stream
belongs to. If it is not told, it decides by the rule:

- it computes, for every channel it has, the belief that the stream is new to it: W+ = symbols of the stream that are
  not in the channel's alphabet, W = all symbols;
- if that belief reaches the speaking line for every channel, a new channel is born: a new origin with an empty store;
- otherwise the stream belongs to the channel it is least new to.

A new language in a known script is not a new channel, but it is unfamiliar. The belief that its symbols are
surprising is the rule on W+ = symbols costing more than the channel's typical bits plus 2, W = all. The mind can
report that it is watching something it does not know.

Watching and speaking. Before each symbol arrives, the mind holds its best guess and the guess's probability. It
would *say* the guess only if two things hold:

- the guess reaches the speaking line e* ~ 0.583;
- the channel has shown that it has structure. The structure belief is the rule on W+ = symbols that the context
  predicted better than a store without context would have, W = all symbols. That store is the channel's own
  order-0 store, which lives the same stream and learns its novelty the same way.

A stream without structure never passes, so the mind keeps watching it and never speaks. A language does pass, and the
mind starts to speak when its evidence allows it.
"""
from __future__ import annotations

import math

from .cortex import Cortex
from .evidence import SPEAKING_LINE, belief
from .lexicon import Lexicon, words


class Channel:
    def __init__(self, name: str, **cortex_args) -> None:
        self.name = name
        self.cortex = Cortex(**cortex_args)
        self.plain = Cortex(**{**cortex_args, "depth": 0})     # the same stream without context
        self.lexicon = Lexicon()
        self.history = ""
        self.helped = 0          # symbols the context predicted better than the store without context
        self.asked = 0           # symbols predicted
        self.bits = 0.0          # bits of everything this channel lived

    def structure(self) -> float:
        """Belief that the channel's context predicts: the rule on its record."""
        return belief(self.helped, self.asked)

    def typical_bits(self) -> float:
        return self.bits / self.asked if self.asked else 0.0


class Mind:
    def __init__(self, **cortex_args) -> None:
        self.cortex_args = cortex_args
        self.channels: dict[str, Channel] = {}
        self.born: list[str] = []                      # channels in the order they were born

    def open(self, name: str | None = None) -> Channel:
        if name is None:
            name = "channel-%d" % (len(self.channels) + 1)
        if name not in self.channels:
            self.channels[name] = Channel(name, **self.cortex_args)
            self.born.append(name)
        return self.channels[name]

    # ----------------------------------------------------------------------------------------------------------------
    def newness(self, segment: str) -> dict[str, float]:
        """For every channel: belief that ``segment`` is new to it (symbols outside its alphabet)."""
        out = {}
        for name, ch in self.channels.items():
            alpha = ch.cortex.alphabet
            out[name] = belief(sum(1 for x in segment if x not in alpha), len(segment))
        return out

    def which(self, segment: str) -> str | None:
        """The channel ``segment`` belongs to, or None if it is new to every channel (a new channel is born)."""
        new = self.newness(segment)
        if not new:
            return None
        name = min(new, key=new.get)
        return name if new[name] < SPEAKING_LINE else None

    def unfamiliar(self, name: str, segment: str) -> float:
        """Belief that ``segment``'s symbols surprise channel ``name`` (cost more than its typical bits + 2)."""
        ch = self.channels[name]
        line = ch.typical_bits() + 2.0
        h = ch.history
        surprising = 0
        for x in segment:
            p = ch.cortex.predict(h, x)
            if p <= 0 or -math.log2(p) > line:
                surprising += 1
            h = (h + x)[-ch.cortex.depth:] if ch.cortex.depth else ""
        return belief(surprising, len(segment))

    # ----------------------------------------------------------------------------------------------------------------
    def perceive(self, segment: str, channel: str | None = None, learn: bool = True,
                 record: list | None = None) -> str:
        """Live ``segment`` on ``channel`` (decided by the rule if not given); return the channel's name.

        ``record``, if given, receives per symbol (bits, said, right): ``said`` is whether the mind would have said its
        guess before the symbol came, ``right`` whether the guess was the symbol."""
        if channel is None:
            channel = self.which(segment)
        ch = self.open(channel)
        cx = ch.cortex
        for x in segment:
            h = ch.history
            if record is not None:
                guess, gp = cx.best(h)
                said = guess is not None and gp >= SPEAKING_LINE and ch.structure() >= SPEAKING_LINE
            if learn:
                b, _ = cx.stream(x, h)                 # predicted first, then learned
                plain, _ = ch.plain.stream(x)
            else:
                b = -math.log2(cx.predict(h, x))
                plain = -math.log2(ch.plain.predict("", x))
            ch.asked += 1
            ch.bits += b
            if b < plain:
                ch.helped += 1
            if record is not None:
                record.append((b, said, said and guess == x))
            ch.history = (h + x)[-cx.depth:] if cx.depth else ""
        if learn:
            for a, b in words(segment):
                ch.lexicon.live(segment[a:b])
        return ch.name

    def say(self, channel: str, prompt: str, most: int = 200) -> tuple[str, str]:
        """Continue ``prompt`` on ``channel`` as long as the mind's guess reaches the speaking line.

        Returns what was said and why it stopped ("silence" when no guess reached the line, or "length")."""
        ch = self.channels[channel]
        cx = ch.cortex
        if ch.structure() < SPEAKING_LINE:
            return "", "silence"
        h = (ch.history + prompt)[-cx.depth:] if cx.depth else ""
        out = []
        for _ in range(most):
            guess, gp = cx.best(h)
            if guess is None or gp < SPEAKING_LINE:
                return "".join(out), "silence"
            out.append(guess)
            h = (h + guess)[-cx.depth:] if cx.depth else ""
        return "".join(out), "length"
