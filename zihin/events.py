# -*- coding: utf-8 -*-
"""Event interface of Zihin (design C1): everything the mind meets or does is an event on a channel.

An event is (time, channel, symbol, provenance) plus an optional intensity in [0, 1]. Time is the moment in
which the event is handed to the mind, so it is not stored in the event itself. Symbols belong to an OPEN
alphabet per channel: a symbol never seen before is legal at any time and gets a new atomic unit. The core
knows nothing else about a channel than its kind:

* ``sense``  what arrives from outside (text first; any other sense later),
* ``act``    what the mind can emit itself; events of others on such a channel are observed behaviour,
* ``feel``   the mind's own body and state (interoception: surprise, relief, load, conflict),
* ``knob``   the mind's own control knobs (internal acts),
* ``source`` the innate channel on which the sources of evidence are units (trust lives on their ties).

Provenance classes (design section 1) say how an event reached the mind; evidence is kept per class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, List, Tuple

SENSED, TOLD, IMITATED, ACTED, INFERRED, IMAGINED = range(6)
PROVENANCE: Tuple[str, ...] = ("sensed", "told", "imitated", "acted", "inferred", "imagined")
N_PROV = len(PROVENANCE)
# Classes that are direct experience: their trust is 1 by definition (what was sensed was sensed; what the mind did,
# it did), so they are never judged themselves.
REFERENCE: Tuple[int, ...] = (SENSED, ACTED)
# The classes whose evidence judges the indirect ones (told, imitated, inferred, imagined): only experience of the
# world. The mind's own acts are certain, but the echo of an act says nothing about whether a told or imagined claim
# about the world holds (decide and dream sorting already set own acts aside as not real), so acted evidence judges
# nothing (audit of 2026-11-15, item 7).
JUDGES: Tuple[int, ...] = (SENSED,)

CHANNEL_KINDS = ("sense", "act", "feel", "knob", "source", "facet", "value", "context")


@dataclass(frozen=True)
class Event:
    """One event handed to the mind in one moment."""

    channel: str
    symbol: Hashable
    prov: int = SENSED
    intensity: float = 1.0


class Channels:
    """Channel registry and the open alphabet of every channel (the adapters' code books).

    The alphabet maps a symbol to a dense index per channel; the unit table stores (channel id, symbol index)
    for atomic units, so the symbol itself (any hashable) never enters a tensor.
    """

    def __init__(self) -> None:
        self.names: List[str] = []
        self.kinds: List[str] = []
        self._id: Dict[str, int] = {}
        self.symbols: List[List[Hashable]] = []
        self._sym: List[Dict[Hashable, int]] = []

    def add(self, name: str, kind: str) -> int:
        if kind not in CHANNEL_KINDS:
            raise ValueError(f"unknown channel kind {kind!r}")
        if name in self._id:
            if self.kinds[self._id[name]] != kind:
                raise ValueError(f"channel {name!r} already registered with another kind")
            return self._id[name]
        self._id[name] = len(self.names)
        self.names.append(name)
        self.kinds.append(kind)
        self.symbols.append([])
        self._sym.append({})
        return self._id[name]

    def id(self, name: str) -> int:
        return self._id[name]

    def has(self, name: str) -> bool:
        return name in self._id

    def kind(self, channel_id: int) -> str:
        return self.kinds[channel_id]

    def symbol_index(self, channel_id: int, symbol: Hashable, create: bool = True) -> int:
        table = self._sym[channel_id]
        index = table.get(symbol, -1)
        if index < 0 and create:
            index = len(self.symbols[channel_id])
            table[symbol] = index
            self.symbols[channel_id].append(symbol)
        return index

    def symbol(self, channel_id: int, index: int) -> Hashable:
        return self.symbols[channel_id][index]

    def truncate(self, channel_id: int, size: int) -> None:
        """Forget the symbols with index >= size (used when the moment that introduced them is undone)."""
        for symbol in self.symbols[channel_id][size:]:
            del self._sym[channel_id][symbol]
        del self.symbols[channel_id][size:]

    def sizes(self) -> List[int]:
        return [len(s) for s in self.symbols]
