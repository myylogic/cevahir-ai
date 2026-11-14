"""The catalogue: every artefact of a home, built by reading its manifests.

Several manifests may describe one key (the same checkpoint registered from two files, or on two
machines); an ``Artifact`` is their union. References resolve here: a full key, an alias (a legacy
tokenizer digest, a CEV-11 weight digest), the key of a payload file, a manifest key, or a short
form ``kind:<hex prefix>`` as the console prints it.

There is no separate index file: the manifests are the truth and are re-read when the catalogue
is rebuilt. That is linear in the number of manifests.
"""
import hashlib
import json
import re

from cevahir_ai.errors import AmbiguousReference, NotFound
from cevahir_ai.keys.grammar import KEY_RE, Key
from cevahir_ai.store.core import MANIFEST_FORMAT

SHORT_RE = re.compile(r"^([a-z][a-z0-9]{1,15}):(?:([a-z0-9]+(?:-[a-z0-9]+)*-v[0-9]+):)?([0-9a-f]{4,64})$")


class Artifact:
    """All manifests of one key."""

    def __init__(self, key):
        self.key = key
        self.manifests = []              # [(manifest id, document)], oldest first

    @property
    def kind(self):
        return self.latest["kind"]

    @property
    def latest(self):
        return self.manifests[-1][1]

    @property
    def created(self):
        return self.manifests[0][1]["created"]

    def _union(self, field, identity):
        seen, merged = set(), []
        for _, document in self.manifests:
            for item in document.get(field, ()):
                token = identity(item)
                if token not in seen:
                    seen.add(token)
                    merged.append(item)
        return merged

    @property
    def aliases(self):
        return self._union("aliases", str)

    @property
    def inputs(self):
        return self._union("inputs", lambda item: (item["role"], item["key"]))

    @property
    def payload(self):
        return self._union("payload", lambda item: (item.get("role"), item["file"]))

    @property
    def files(self):
        return [item["file"] for item in self.payload]

    def short(self):
        return Key.parse(self.key).display()


class Catalog:
    def __init__(self, store):
        self.artifacts = {}
        self.corrupt = []                # object ids whose bytes no longer hash to their name
        self._by_alias, self._by_file, self._by_manifest = {}, {}, {}
        documents = []
        for digest in store.object_ids():
            data = store.object_path(digest).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                self.corrupt.append(digest)
                continue
            document = json.loads(data)
            if document.get("format") == MANIFEST_FORMAT:
                documents.append((document.get("created", ""), digest, document))
        for _, digest, document in sorted(documents, key=lambda item: item[:2]):
            artifact = self.artifacts.setdefault(document["key"], Artifact(document["key"]))
            artifact.manifests.append((digest, document))
            self._by_manifest[digest] = document["key"]
            for alias in document.get("aliases", ()):
                self._by_alias.setdefault(alias, set()).add(document["key"])
            for item in document.get("payload", ()):
                self._by_file.setdefault(item["file"], set()).add(document["key"])

    def of_kind(self, kind=None):
        found = [artifact for artifact in self.artifacts.values() if kind is None or artifact.kind == kind]
        return sorted(found, key=lambda artifact: (artifact.kind, artifact.created, artifact.key))

    def by_file(self, file_key):
        """Artefacts whose payload is the file with this ``file:bytes-v1`` key."""
        return [self.artifacts[key] for key in sorted(self._by_file.get(str(file_key), ()))]

    def canonical(self, key):
        """The artefact key that ``key`` names: itself, or the one artefact it is an alias of."""
        key = str(key)
        if key in self.artifacts:
            return key
        owners = self._by_alias.get(key, ())
        return next(iter(owners)) if len(owners) == 1 else key

    def find(self, reference):
        """Every artefact ``reference`` could mean (see the module docstring)."""
        text = str(reference).strip()
        if KEY_RE.match(text):
            if text in self.artifacts:
                return [self.artifacts[text]]
            key = Key.parse(text)
            if key.kind == "manifest" and key.digest in self._by_manifest:
                return [self.artifacts[self._by_manifest[key.digest]]]
            owners = self._by_alias.get(text) or self._by_file.get(text) or ()
            return [self.artifacts[owner] for owner in sorted(owners)]
        match = SHORT_RE.match(text)
        if not match:
            return []
        kind, scheme, prefix = match.groups()
        head = f"cev:{kind}:{scheme}:" if scheme else f"cev:{kind}:"
        owners = set()
        for key, artifact in self.artifacts.items():
            for name in [key, *artifact.aliases]:
                if name.startswith(head) and name.rsplit(":", 1)[1].startswith(prefix):
                    owners.add(key)
        return [self.artifacts[owner] for owner in sorted(owners)]

    def resolve(self, reference):
        found = self.find(reference)
        if not found:
            raise NotFound(f"no registered artefact matches {str(reference)!r}")
        if len(found) > 1:
            raise AmbiguousReference(f"{str(reference)!r} matches {len(found)} artefacts: "
                                     + ", ".join(artifact.short() for artifact in found[:5]) + "; give more digits")
        return found[0]
