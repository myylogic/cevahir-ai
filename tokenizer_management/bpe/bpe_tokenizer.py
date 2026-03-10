# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: bpe_tokenizer.py
Modül: tokenizer_management/bpe
Görev: BPETokenizer interface tanımları - Encoder/Decoder/Trainer için
       Protocol tanımları. SOLID prensiplerine uygun interface segregation
       sağlar. Type checking ve runtime interface kontrolü için kullanılır.

MİMARİ:
- SOLID Prensipleri: Interface Segregation (ayrı interface'ler),
                     Dependency Inversion (Protocol-based interfaces)
- Design Patterns: Protocol Pattern (Python typing.Protocol)
- Endüstri Standartları: Type-safe interfaces, Protocol-based design

KULLANIM:
- Type checking için (mypy, pyright)
- Runtime interface kontrolü için (@runtime_checkable)
- Encoder/Decoder/Trainer interface tanımları

BAĞIMLILIKLAR:
- typing.Protocol: Interface tanımları için
- BPETokenizerError: Özel exception sınıfı

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class BPETokenizerError(Exception):
    """BPETokenizer ile ilgili hataları tanımlamak için özel exception."""
    pass


# --- Arayüzler (SOLID: Interface Segregation) ---------------------------------

@runtime_checkable
class EncoderProtocol(Protocol):
    def set_vocab(self, new_vocab: Dict[str, dict]) -> None: ...
    def set_merges(self, merges: Optional[List[Tuple[str, str]]]) -> None: ...
    def encode(self, tokens_or_text: Union[str, List[str]], mode: str = "inference") -> Union[List[int], Tuple[List[str], List[int]]]: ...
    # İsteğe bağlı ama çoğu encoder’da var:
    def encode_sequence(self, tokens: List[str]) -> List[int]: ...  # pragma: no cover


@runtime_checkable
class DecoderProtocol(Protocol):
    def set_vocab(self, new_vocab: Dict[str, dict]) -> None: ...
    def set_merges(self, merges: Optional[List[Tuple[str, str]]]) -> None: ...
    def decode(
        self,
        token_ids: List[int],
        *,
        remove_specials: bool = True,
        remove_tags: bool = True,
        sep_token: str = "<SEP>",
        collapse_spaces: bool = True,
        lowercase: bool = False
    ) -> str: ...


# --- Uygulama ------------------------------------------------------------------

class BPETokenizer:
    """
    Dış API katmanı: Encoder/Decoder ile metin <-> ID dönüşümlerini gerçekleştirir.
    - Encoder/Decoder, yukarıdaki Protocol’leri karşılamalıdır.
    - Vocab ve merges senkronizasyonu kontrollü ve açık hatalıdır (sessiz drift yok).

    Örnek:
        tok = BPETokenizer(encoder, decoder, vocab)
        ids = tok.encode_text("merhaba dünya", mode="inference")
        text = tok.decode_ids(ids)
    """

    def __init__(
        self,
        encoder: EncoderProtocol,
        decoder: DecoderProtocol,
        vocab: Optional[Dict[str, dict]] = None,
        merges: Optional[List[Tuple[str, str]]] = None
    ) -> None:
        if not isinstance(encoder, EncoderProtocol) or not isinstance(decoder, DecoderProtocol):
            raise BPETokenizerError("Encoder/Decoder beklenen arayüzleri (Protocol) karşılamıyor.")

        self.encoder: EncoderProtocol = encoder
        self.decoder: DecoderProtocol = decoder
        self._vocab: Dict[str, dict] = dict(vocab) if isinstance(vocab, dict) else {}

        # İlk senkronizasyon
        if self._vocab:
            self._sync_vocab()
        if merges is not None:
            self._sync_merges(merges)

        logger.debug("[BPETokenizer] başlatıldı | vocab=%d merges=%s",
                     len(self._vocab), "var" if merges else "yok")

    # ----------------- Kamu API -----------------

    def encode_text(self, text: str, *, mode: str = "inference", return_tokens: bool = False) -> Union[List[int], Tuple[List[str], List[int]]]:
        """
        Metni ID listesine çevirir. Encoder’ın döndürdüğü tipe uyum sağlar.
        Args:
            text: Hammadde metin.
            mode: 'train' veya 'inference'.
            return_tokens: True ise (tokens, ids) döndürmeye çalışır; encoder yalnız ids döndürüyorsa
                           tokens için [] döner.
        """
        if not isinstance(text, str):
            raise BPETokenizerError("encode_text: text str olmalıdır.")
        if not text.strip():
            raise BPETokenizerError("encode_text: boş metin kodlanamaz.")
        if not self._vocab:
            raise BPETokenizerError("encode_text: vocab yüklenmemiş.")

        try:
            out = self.encoder.encode(text, mode=mode)
        except Exception as e:
            raise BPETokenizerError(f"encode_text: encoder hata verdi: {e}") from e

        tokens, ids = self._unpack_encode_output(out)
        logger.debug("[BPETokenizer] encode_text → ids=%s", ids)
        return (tokens, ids) if return_tokens else ids

    def encode_tokens(self, tokens: List[str]) -> List[int]:
        """
        Önceden tokenleştirilmiş bir listeyi ID’lere çevirir (örn. kelime+hece vb.).
        Encoder `encode_sequence` sağlıyorsa onu kullanır, aksi halde `encode(tokens)` çağırır.
        """
        if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
            raise BPETokenizerError("encode_tokens: List[str] beklenir.")
        if not tokens:
            raise BPETokenizerError("encode_tokens: boş liste kabul edilmez.")
        if not self._vocab:
            raise BPETokenizerError("encode_tokens: vocab yüklenmemiş.")

        try:
            if hasattr(self.encoder, "encode_sequence"):
                ids = self.encoder.encode_sequence(tokens)  # type: ignore[attr-defined]
            else:
                out = self.encoder.encode(tokens, mode="inference")
                _, ids = self._unpack_encode_output(out)
        except Exception as e:
            raise BPETokenizerError(f"encode_tokens: encoder hata verdi: {e}") from e

        logger.debug("[BPETokenizer] encode_tokens → ids=%s", ids)
        return ids

    def decode_ids(
        self,
        token_ids: List[int],
        *,
        remove_specials: bool = True,
        remove_tags: bool = True,
        sep_token: str = "<SEP>",
        collapse_spaces: bool = True,
        lowercase: bool = False
    ) -> str:
        """ID listesini metne çevirir (decoder’ın gelişmiş bayraklarıyla)."""
        if not isinstance(token_ids, list) or not all(isinstance(i, int) for i in token_ids):
            raise BPETokenizerError("decode_ids: List[int] beklenir.")
        if not token_ids:
            return ""
        if not self._vocab:
            raise BPETokenizerError("decode_ids: vocab yüklenmemiş.")

        try:
            text = self.decoder.decode(
                token_ids,
                remove_specials=remove_specials,
                remove_tags=remove_tags,
                sep_token=sep_token,
                collapse_spaces=collapse_spaces,
                lowercase=lowercase,
            )
        except Exception as e:
            raise BPETokenizerError(f"decode_ids: decoder hata verdi: {e}") from e

        logger.debug("[BPETokenizer] decode_ids → %s", text if len(text) < 200 else text[:200] + "…")
        return text

    # Kısayollar (eski API uyumu)
    def encode(self, text: str, mode: str = "inference") -> List[int]:
        return self.encode_text(text, mode=mode, return_tokens=False)  # type: ignore[return-value]

    def decode(self, token_ids: List[int]) -> str:
        return self.decode_ids(token_ids)

    def get_token_ids(self, text: str) -> List[int]:
        return self.encode_text(text, mode="inference", return_tokens=False)  # type: ignore[return-value]

    def get_text(self, token_ids: List[int]) -> str:
        return self.decode_ids(token_ids)

    # ----------------- Senkronizasyon -----------------

    def update_vocab(self, new_vocab: Dict[str, dict]) -> None:
        """Yeni vocab’ı yükle ve encoder/decoder ile senkronize et."""
        if not isinstance(new_vocab, dict) or not new_vocab:
            raise BPETokenizerError("update_vocab: geçersiz veya boş vocab.")
        self._vocab = dict(new_vocab)
        self._sync_vocab()
        logger.debug("[BPETokenizer] vocab güncellendi | size=%d", len(self._vocab))

    def set_merges(self, merges: Optional[List[Tuple[str, str]]]) -> None:
        """Dışarıdan merges set et ve encoder/decoder’a aktar."""
        self._sync_merges(merges if merges is not None else [])

    def get_vocab(self) -> Dict[str, dict]:
        return dict(self._vocab)

    # ----------------- İç yardımcılar -----------------

    def _sync_vocab(self) -> None:
        try:
            self.encoder.set_vocab(self._vocab)
            self.decoder.set_vocab(self._vocab)
        except Exception as e:
            raise BPETokenizerError(f"_sync_vocab: senkronizasyon hatası: {e}") from e
        logger.debug("[BPETokenizer] encoder/decoder ile vocab senkronize edildi.")

    def _sync_merges(self, merges: List[Tuple[str, str]]) -> None:
        try:
            self.encoder.set_merges(merges)
            self.decoder.set_merges(merges)
        except Exception as e:
            raise BPETokenizerError(f"_sync_merges: senkronizasyon hatası: {e}") from e
        logger.debug("[BPETokenizer] encoder/decoder ile merges senkronize edildi | count=%d", len(merges))

    @staticmethod
    def _unpack_encode_output(
        result: Union[List[int], Tuple[List[str], List[int]]]
    ) -> Tuple[List[str], List[int]]:
        """
        Encoder’ın olası iki dönüş tipini normalize eder:
        - [ids] → ([], ids)
        - (tokens, ids) → (tokens, ids)
        """
        if isinstance(result, tuple):
            if len(result) != 2 or not isinstance(result[0], list) or not isinstance(result[1], list):
                raise BPETokenizerError("Encoder dönüşü (tokens, ids) beklenirken uyumsuz tip alındı.")
            tokens, ids = result
            if not all(isinstance(t, str) for t in tokens) or not all(isinstance(i, int) for i in ids):
                raise BPETokenizerError("Encoder (tokens, ids) tipleri geçersiz.")
            return tokens, ids

        # Salt ID listesi
        ids_only = result
        if not isinstance(ids_only, list) or not all(isinstance(i, int) for i in ids_only):
            raise BPETokenizerError("Encoder dönüşü List[int] ya da (List[str], List[int]) olmalıdır.")
        return [], ids_only
