"""``Record``: the typed, ordered property bag that operations return and the console pipes.

A record has a type name (``Cevahir.Checkpoint``) and named properties. Property lookup ignores
case, as PowerShell does. A record holds plain data only (``None``, ``bool``, ``int``, ``float``,
``str``, lists, dicts and other records): nothing in it is callable, so the console's expression
language can expose every property without exposing Python objects.
"""


class Record:
    __slots__ = ("type_name", "_props", "_lower")

    def __init__(self, type_name, props=None, **more):
        object.__setattr__(self, "type_name", type_name)
        object.__setattr__(self, "_props", dict(props or {}, **more))
        object.__setattr__(self, "_lower", {name.lower(): name for name in self._props})

    def names(self):
        return list(self._props)

    def has(self, name):
        return isinstance(name, str) and name.lower() in self._lower

    def get(self, name, default=None):
        actual = self._lower.get(name.lower()) if isinstance(name, str) else None
        return default if actual is None else self._props[actual]

    def set(self, name, value):
        actual = self._lower.get(name.lower())
        if actual is None:
            self._lower[name.lower()] = actual = name
        self._props[actual] = value

    def items(self):
        return self._props.items()

    def to_dict(self, typed=False):
        """Plain data, nested records included. ``typed`` adds ``__type`` (the wire form)."""
        data = {"__type": self.type_name} if typed else {}
        data.update((name, plain(value, typed)) for name, value in self._props.items())
        return data

    def __getitem__(self, name):
        if not self.has(name):
            raise KeyError(name)
        return self.get(name)

    def __contains__(self, name):
        return self.has(name)

    def __getattr__(self, name):
        if name.startswith("_") or not self.has(name):
            raise AttributeError(name)
        return self.get(name)

    def __setattr__(self, name, value):
        raise AttributeError("use Record.set(name, value)")

    def __eq__(self, other):
        return isinstance(other, Record) and other.type_name == self.type_name and other._props == self._props

    def __hash__(self):
        return hash((self.type_name, tuple(self._props)))

    def __repr__(self):
        inner = ", ".join(f"{name}={value!r}" for name, value in list(self._props.items())[:4])
        return f"<{self.type_name} {inner}{' …' if len(self._props) > 4 else ''}>"


def plain(value, typed=False):
    """``value`` as JSON-ready data: records become dicts, tuples become lists."""
    if isinstance(value, Record):
        return value.to_dict(typed)
    if isinstance(value, dict):
        return {str(name): plain(item, typed) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item, typed) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)
