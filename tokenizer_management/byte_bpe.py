# -*- coding: utf-8 -*-
"""Kayıpsız bayt düzeyinde BPE (byte-level BPE) — Türkçe için ve her UTF-8 metin için.

Neden ayrı bir modül?
    Mevcut ``TokenizerCore`` yolu kelime listesi + karakter yedeği gibi çalışır; boşlukları
    birleştirir, ``.,!?`` dışındaki işaretleri ve â/î/û içeren sözcükleri düşürür. Bu modül
    yol haritasındaki TOK-01 (kayıpsız Unicode tokenizasyonu) için bağımsız bir arka uçtur:
    ``decode(encode(x)) == x`` her geçerli Unicode metin için birebir sağlanır. Mevcut yola
    ve kitapta satır düzeyinde bağlı kaynaklara dokunmaz.

Nasıl çalışır?
    1. Ön bölütleme (pre-tokenization) metni harf dizileri, 1-3 haneli sayı grupları,
       noktalama dizileri ve boşluklar olarak parçalar. Tek bir öndeki boşluk sonraki
       parçaya eklenir (GPT-2 geleneği), böylece boşluk ayrı token harcamaz.
       ``Ankara'da`` → ``Ankara`` + ``'da``: Türkçede kesme işareti özel adı ekinden ayırır;
       kesme işareti (' veya ’) ve ardından gelen harfler tek parça olur, böylece kök ile ek
       ayrı tokenlere düşer ve ek allomorfları (``'da``, ``'nın``) tek token olarak öğrenilebilir.
       Parçaların birleşimi her zaman girdinin kendisidir (kayıpsızlık burada başlar).
    2. Her parça UTF-8 baytlarına çevrilir; her bayt görünür bir karaktere eşlenir.
    3. Eğitim, sözcük sıklıkları üzerinden klasik BPE'dir: en sık bitişik çift birleştirilir,
       çift sayıları artımlı güncellenir (yığın + tembel silme). Eşitlikte sözlük sırası
       belirleyicidir; aynı veri ve ayarlarla her çalıştırma aynı birleştirmeleri üretir.
    4. Kodlama, her parçaya birleştirmeleri sıralarına göre uygular; bilinmeyen karakter
       olamaz, çünkü 256 baytın tamamı sözlüktedir (``<UNK>`` yalnızca uyumluluk içindir).

Özel token kimlikleri Cevahir kuralıyla aynıdır: ``<PAD>``=0, ``<UNK>``=1, ``<BOS>``=2,
``<EOS>``=3. Baytlar 4..259, birleştirmeler 260'tan itibaren gelir.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

FORMAT_NAME = "cevahir-byte-bpe"
FORMAT_VERSION = 1
SPECIAL_TOKENS: Tuple[str, ...] = ("<PAD>", "<UNK>", "<BOS>", "<EOS>")
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

# Harf: \w içinden rakam ve alt çizgi hariç her şey (çğıöşüâîû ve tüm Unicode harfleri).
# Kesme işareti + harfler (Türkçe özel ad eki: 'da, ’nın) tek parçadır.
# Sıra önemlidir; her karakter tam olarak bir seçeneğe düşer, böylece hiçbir karakter atlanmaz.
PRETOKENIZE_PATTERN = r" ?[^\W\d_]+|['’][^\W\d_]+| ?\d{1,3}| ?_+| ?[^\s\w]+|\s+(?!\S)|\s+"
_PRETOKENIZE = re.compile(PRETOKENIZE_PATTERN)


def bytes_to_unicode() -> Dict[int, str]:
    """GPT-2 bayt→karakter eşlemesi: 256 baytın her biri yazdırılabilir tek bir karaktere."""
    visible = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))
    chars = visible[:]
    extra = 0
    for b in range(256):
        if b not in visible:
            visible.append(b)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(visible, (chr(c) for c in chars)))


_BYTE_ENCODER = bytes_to_unicode()
_BYTE_DECODER = {c: b for b, c in _BYTE_ENCODER.items()}


def pretokenize(text: str) -> List[str]:
    """Metni kayıpsız parçalara böler; ``"".join(pretokenize(t)) == t`` her zaman doğrudur."""
    return _PRETOKENIZE.findall(text)


def _to_symbols(chunk: str) -> Tuple[str, ...]:
    return tuple(_BYTE_ENCODER[b] for b in chunk.encode("utf-8"))


class ByteLevelBPE:
    """Kayıpsız bayt düzeyinde BPE tokenizer.

    >>> tok = ByteLevelBPE.train(["Ankara'da kar yağıyor.", "Ankara'nın havası"], vocab_size=300)
    >>> tok.decode(tok.encode("Çekoslovakyalılaştıramadıklarımızdan mısınız?"))
    'Çekoslovakyalılaştıramadıklarımızdan mısınız?'
    """

    def __init__(self, merges: Sequence[Tuple[str, str]], cache_size: int = 100_000) -> None:
        self.merges: List[Tuple[str, str]] = [tuple(m) for m in merges]  # type: ignore[misc]
        self.ranks: Dict[Tuple[str, str], int] = {pair: i for i, pair in enumerate(self.merges)}
        if len(self.ranks) != len(self.merges):
            raise ValueError("duplicate merge rules")
        self.id_to_token: List[str] = list(SPECIAL_TOKENS) + [_BYTE_ENCODER[b] for b in range(256)]
        for a, b in self.merges:
            self.id_to_token.append(a + b)
        self.token_to_id: Dict[str, int] = {}
        for i, tok in enumerate(self.id_to_token):
            # Bir birleştirme sonucu bir bayt dizisiyle çakışamaz (birleştirmeler en az iki bayttır),
            # fakat iki farklı birleştirme yolu aynı diziyi üretebilir; ilk kimlik kanoniktir.
            self.token_to_id.setdefault(tok, i)
        self._cache: Dict[str, Tuple[int, ...]] = {}
        self._cache_size = int(cache_size)

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int, min_frequency: int = 2,
              cache_size: int = 100_000) -> "ByteLevelBPE":
        """Sözcük-parça sıklıkları üzerinden BPE birleştirmelerini öğrenir.

        ``vocab_size`` özel tokenler ve 256 bayt dahil hedef sözlük boyutudur. Hiçbir çift
        ``min_frequency`` sıklığına ulaşmadığında eğitim erken durur.
        """
        base = len(SPECIAL_TOKENS) + 256
        if vocab_size < base:
            raise ValueError(f"vocab_size must be >= {base}")
        counts: Counter = Counter()
        for text in texts:
            counts.update(pretokenize(text))
        words: List[List[str]] = []
        freqs: List[int] = []
        for chunk, freq in sorted(counts.items()):
            words.append(list(_to_symbols(chunk)))
            freqs.append(freq)

        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        where: Dict[Tuple[str, str], set] = defaultdict(set)
        for wi, (sym, f) in enumerate(zip(words, freqs)):
            for pair in zip(sym, sym[1:]):
                pair_counts[pair] += f
                where[pair].add(wi)
        heap = [(-c, pair) for pair, c in pair_counts.items()]
        heapq.heapify(heap)

        merges: List[Tuple[str, str]] = []
        target = vocab_size - base
        while len(merges) < target and heap:
            neg, pair = heapq.heappop(heap)
            current = pair_counts.get(pair, 0)
            if current <= 0 or -neg != current:
                if current > 0:  # tembel silme: güncel sayıyla yeniden it
                    heapq.heappush(heap, (-current, pair))
                continue
            if current < min_frequency:
                break
            merges.append(pair)
            a, b = pair
            merged = a + b
            touched: Dict[Tuple[str, str], int] = defaultdict(int)
            for wi in sorted(where.pop(pair, ())):
                sym, f = words[wi], freqs[wi]
                if len(sym) < 2:
                    continue
                for p in zip(sym, sym[1:]):  # eski çiftleri çıkar
                    pair_counts[p] -= f
                    touched[p] += 0
                out: List[str] = []
                i = 0
                while i < len(sym):
                    if i + 1 < len(sym) and sym[i] == a and sym[i + 1] == b:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(sym[i])
                        i += 1
                words[wi] = out
                for p in zip(out, out[1:]):  # yeni çiftleri ekle
                    pair_counts[p] += f
                    where[p].add(wi)
                    touched[p] += 0
            pair_counts.pop(pair, None)
            for p in touched:
                c = pair_counts.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
                else:
                    pair_counts.pop(p, None)
        return cls(merges, cache_size=cache_size)

    # ------------------------------------------------------------------ encoding
    def _bpe(self, chunk: str) -> Tuple[int, ...]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        sym = list(_to_symbols(chunk))
        ranks = self.ranks
        while len(sym) > 1:
            best_rank, best_i = None, -1
            for i in range(len(sym) - 1):
                r = ranks.get((sym[i], sym[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            a, b = self.merges[best_rank]
            out: List[str] = []
            i = 0
            while i < len(sym):
                if i + 1 < len(sym) and sym[i] == a and sym[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(sym[i])
                    i += 1
            sym = out
        ids = tuple(self.token_to_id[s] for s in sym)
        if len(self._cache) >= self._cache_size:
            self._cache.clear()  # sınırlı bellek: sunucuda sınırsız büyümez
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = [BOS_ID] if add_bos else []
        for chunk in pretokenize(text):
            ids.extend(self._bpe(chunk))
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def tokenize(self, text: str) -> List[str]:
        """Okunabilir token parçaları (baytlar UTF-8 olarak çözülmeye çalışılır)."""
        return [self.decode([i]) for i in self.encode(text)]

    def decode(self, ids: Iterable[int], skip_special: bool = True, errors: str = "strict") -> str:
        """Kimlikleri metne çevirir. Tam bir ``encode`` çıktısı için sonuç birebir girdidir.

        Akış hâlinde üretimde çok baytlı bir karakter ortadan bölünebilir; o durumda
        ``errors="replace"`` kullanın.
        """
        out = bytearray()
        n_special = len(SPECIAL_TOKENS)
        for i in ids:
            i = int(i)
            if i < 0 or i >= len(self.id_to_token):
                raise ValueError(f"token id out of range: {i}")
            if i < n_special:
                if skip_special:
                    continue
                out.extend(self.id_to_token[i].encode("utf-8"))
                continue
            out.extend(_BYTE_DECODER[c] for c in self.id_to_token[i])
        return out.decode("utf-8", errors=errors)

    # ------------------------------------------------------------------ metadata & persistence
    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.token_to_id)

    def identity(self) -> str:
        """Biçim, desen, özel tokenler ve birleştirmelerden türetilen kararlı kimlik (sha256)."""
        payload = json.dumps({"format": FORMAT_NAME, "version": FORMAT_VERSION,
                              "pattern": PRETOKENIZE_PATTERN, "special": SPECIAL_TOKENS,
                              "merges": self.merges}, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> None:
        data = {"format": FORMAT_NAME, "version": FORMAT_VERSION, "pattern": PRETOKENIZE_PATTERN,
                "special_tokens": list(SPECIAL_TOKENS), "vocab_size": self.vocab_size,
                "identity": self.identity(), "merges": [list(m) for m in self.merges]}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, cache_size: int = 100_000) -> "ByteLevelBPE":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != FORMAT_NAME or data.get("version") != FORMAT_VERSION:
            raise ValueError(f"not a {FORMAT_NAME} v{FORMAT_VERSION} file: {path}")
        if data.get("pattern") != PRETOKENIZE_PATTERN or tuple(data.get("special_tokens", ())) != SPECIAL_TOKENS:
            raise ValueError("pre-tokenizer pattern or special tokens differ from this implementation")
        tok = cls([tuple(m) for m in data["merges"]], cache_size=cache_size)
        if data.get("identity") and data["identity"] != tok.identity():
            raise ValueError("tokenizer identity mismatch (file was modified)")
        return tok
