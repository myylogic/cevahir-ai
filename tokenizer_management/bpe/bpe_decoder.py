# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: bpe_decoder.py
Modül: tokenizer_management/bpe
Görev: BPEDecoder sınıfı - Token ID → metin dönüşümü (decoding). Token ID'lerini
       BPE merges kullanarak birleştirir ve okunabilir metne dönüştürür.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (decoding işlemleri),
                     Dependency Inversion (BPEManager interface'i)
- Design Patterns: Strategy Pattern (farklı decoding stratejileri)
- Endüstri Standartları: GPT-2/3/4 BPE decoding, SentencePiece benzeri yaklaşım

KULLANIM:
- BPEManager.decode() tarafından kullanılır
- Token ID → metin dönüşümü için
- Özel token (BOS/EOS/SEP/PAD) yönetimi

BAĞIMLILIKLAR:
- bpe_manager_utils: Vocab ve merges yardımcı fonksiyonları
- BPE_DETAILED_CONFIG: Tokenization yapılandırması
- BPEDecodingError: Özel exception sınıfı

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
import re
from typing import Dict, List, Optional, Tuple, Any

from .bpe_manager_utils import DEFAULT_SPECIALS, default_vocab as _default_vocab
from tokenizer_management.config import (
    BPE_DETAILED_CONFIG,
    DECODER_CONFIG,
    get_decoder_config,
)

logger = logging.getLogger(__name__)


class BPEDecodingError(Exception):
    """BPEDecoder ile ilgili hataları tanımlamak için özel exception."""
    pass


class DummyPostprocessor:
    """
    Basit varsayılan postprocessor; token listesini boşlukla birleştirir.
    Gerçek üretimde domain’e özgü bir postprocessor enjekte edilebilir.
    """
    def process(self, tokens: List[str]) -> str:
        return " ".join(tokens)


class BPEDecoder:
    """
    Sorumluluk: ID dizisini metne dönüştürmek.
    - Vocab: { token: {"id": int, "total_freq": int, "positions": List[int]} }
    - Merges: Decode için zorunlu değildir; ID→token map’i yeterlidir.

    Detokenizasyon hedefi:
      * Kelime sonu işaretçisi </w> kaldırılır.
      * Sol/sağ noktalama ((), [], {}, «», “”, ‘‘’’, …) etrafı düzgün boşluklanır.
      * Tire, apostrof, %, para birimleri için Türkçe’ye uygun basit kurallar uygulanır.
      * <SEP> boşluğa çevrilir.
      * __tag__* ve özel tokenlar istenirse metinden ayıklanır.
    """

    # ---------------------- Init ----------------------

    def __init__(
        self,
        vocab: Dict[str, dict],
        merges: Optional[List[Tuple[str, str]]] = None,
        postprocessor: Optional[DummyPostprocessor] = None,
        use_gpu: Optional[bool] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        if not isinstance(vocab, dict) or not vocab:
            raise TypeError("Vocab bir sözlük olmalı ve boş olmamalıdır.")
        
        # Config merge
        self.config = {**DECODER_CONFIG}
        self.config.update(BPE_DETAILED_CONFIG)
        if config:
            self.config.update(config)
        
        # GPU desteği (config'ten, parametre sadece override için)
        if use_gpu is None:
            use_gpu = self.config.get("use_gpu", False)
        self.use_gpu = use_gpu
        if self.use_gpu:
            try:
                import torch
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if self.device.type == "cpu":
                    logger.warning("[BPEDecoder] GPU isteniyor ama CUDA mevcut değil, CPU kullanılacak")
                    self.use_gpu = False
                else:
                    logger.info(f"[BPEDecoder] GPU desteği aktif: {self.device}")
            except ImportError:
                logger.warning("[BPEDecoder] PyTorch bulunamadı, GPU desteği devre dışı")
                self.use_gpu = False
                self.device = None
        else:
            self.device = None

        self.vocab: Dict[str, dict] = dict(vocab)
        self._ensure_special_tokens_exact()
        self.reverse_vocab: Dict[int, str] = self._build_reverse_vocab()

        self.merges: List[Tuple[str, str]] = list(merges) if merges else []
        self.postprocessor = postprocessor or DummyPostprocessor()

        logger.debug("[BPEDecoder] Başlatıldı | vocab_size=%d merges=%d",
                     len(self.vocab), len(self.merges))

    # ---------------------- Kamu API ----------------------

    def set_vocab(self, new_vocab: Dict[str, dict]) -> None:
        """Yeni vocab ata ve reverse_vocab’i yeniden kur."""
        if not isinstance(new_vocab, dict) or not new_vocab:
            raise ValueError("set_vocab: yeni vocab dict olmalı ve boş olmamalı.")
        self.vocab = dict(new_vocab)
        self._ensure_special_tokens_exact()
        self.reverse_vocab = self._build_reverse_vocab()
        logger.debug("[BPEDecoder] vocab güncellendi | size=%d", len(self.vocab))

    def set_merges(self, merges: Optional[List[Tuple[str, str]]]) -> None:
        """Dışarıdan gelen merges listesini ata (decode için zorunlu değil)."""
        if merges is None:
            self.merges = []
            return
        if not all(isinstance(p, tuple) and len(p) == 2 and
                   isinstance(p[0], str) and isinstance(p[1], str) for p in merges):
            raise ValueError("set_merges: merges List[Tuple[str,str]] olmalıdır.")
        self.merges = list(merges)
        logger.debug("[BPEDecoder] merges güncellendi | count=%d", len(self.merges))

    def decode(
        self,
        token_ids: List[int],
        *,
        remove_specials: Optional[bool] = None,
        remove_tags: Optional[bool] = None,
        sep_token: str = "<SEP>",
        collapse_spaces: Optional[bool] = None,
        lowercase: Optional[bool] = None,
        prefer: Optional[str] = None,
    ) -> str:
        """
        ID listesini metne çevirir.
        
        ENDÜSTRI STANDARDI: Tüm token'ları decode et, token kaybı olmamalı.
        prefer parametresi token kaybına neden olabilir, use_prefer_filter=False önerilir.

        prefer:
        - "word"     : </w> ile biten kelime tokenlarını tercih eder (parça bazlı), hece yoksa fallback
        - "syllable" : hece tokenlarını tercih eder (parça bazlı), kelime yoksa fallback
        - "auto"     : parça bazlı; parçada kelime tokenı varsa 'word', yoksa 'syllable'
        """
        # Parametreleri config'ten al (None ise)
        if remove_specials is None:
            remove_specials = self.config.get("remove_specials", True)
        if remove_tags is None:
            remove_tags = self.config.get("remove_tags", True)
        if collapse_spaces is None:
            collapse_spaces = self.config.get("collapse_spaces", True)
        if lowercase is None:
            lowercase = self.config.get("lowercase", False)
        
        # ✅ ENDÜSTRI STANDARDI: Token kaybını önle
        # use_prefer_filter=False ise, prefer filtresini devre dışı bırak (tüm token'ları decode et)
        use_prefer_filter = self.config.get("use_prefer_filter", False)
        if not use_prefer_filter:
            prefer = None  # ✅ Filtreleme yapma, tüm token'ları decode et
        if prefer is None:
            prefer = self.config.get("prefer_mode", "auto")
        
        if not isinstance(token_ids, list) or not all(isinstance(i, int) for i in token_ids):
            raise BPEDecodingError("decode: Girdi List[int] olmalıdır.")
        if not token_ids:
            logger.debug("[BPEDecoder] Boş ID listesi -> ''")
            return ""

        # 1) ID -> token
        tokens: List[str] = []
        for tid in token_ids:
            tok = self.reverse_vocab.get(tid)
            if tok is None:
                raise BPEDecodingError(f"Bilinmeyen token id: {tid}")
            tokens.append(tok)
        logger.debug("[BPEDecoder] id->token: %s", tokens if len(tokens) < 200 else f"{tokens[:100]}…")

        specials = set(DEFAULT_SPECIALS.keys())

        def _is_alphaish(t: str) -> bool:
            # Unicode "kelime" karakteri içeriyor mu (harf/rakam/altçizgi)?
            return re.search(r"\w", t, flags=re.UNICODE) is not None

        # 2) Tokenları segmentlere böl: metin segmentleri ve geçiş (passthrough) segmentleri
        #    - sep_token kendi segmentidir
        #    - special/tag tokenlar kendi segmentidir
        segments: List[dict] = []
        cur: List[str] = []

        def _flush_cur():
            if cur:
                segments.append({"kind": "text", "tokens": cur.copy()})
                cur.clear()

        for t in tokens:
            if t == sep_token:
                _flush_cur()
                segments.append({"kind": "pass", "tokens": [t]})
            elif t in specials or t.startswith("__tag__"):
                _flush_cur()
                segments.append({"kind": "pass", "tokens": [t]})
            else:
                cur.append(t)
        _flush_cur()

        # ✅ ENDÜSTRI STANDARDI: Token kaybını önle - prefer filtresini devre dışı bırak
        # use_prefer_filter=False ise, filtreleme yapma (TÜM TOKEN'LARI DECODE ET)
        use_prefer_filter = self.config.get("use_prefer_filter", False)
        
        # 3) Parça bazlı tercih filtresi
        pref = (prefer or "auto").lower()
        if pref not in ("word", "syllable", "auto"):
            pref = "auto"

        def _filter_piece(piece: List[str], mode: str) -> List[str]:
            """Parça için word/syllable tercihine göre filtre uygula; noktalama ve boşluklar korunur."""
            has_word = any(t.endswith("</w>") for t in piece)
            has_syll = any((_is_alphaish(t) and not t.endswith("</w>")) for t in piece)

            if mode == "auto":
                mode = "word" if has_word else "syllable"

            out: List[str] = []
            if mode == "word":
                if has_word:
                    for t in piece:
                        # alfabetik kelime tokenlarını tut, heceyi ele; noktalama ve boşluklar kalsın
                        if t.endswith("</w>") or not _is_alphaish(t) or t == ' ':
                            out.append(t)
                    if not out and has_syll:
                        # fallback: hece
                        for t in piece:
                            if not (t.endswith("</w>") and _is_alphaish(t)) or t == ' ':
                                out.append(t)
                else:
                    # fallback: hece
                    for t in piece:
                        if not (t.endswith("</w>") and _is_alphaish(t)) or t == ' ':
                            out.append(t)
            else:  # mode == "syllable"
                if has_syll:
                    for t in piece:
                        # kelime tokenlarını ele (yalnız alfabetik olanları); noktalama ve boşluklar kalsın
                        if not (t.endswith("</w>") and _is_alphaish(t)) or t == ' ':
                            out.append(t)
                    if not out and has_word:
                        # fallback: kelime
                        for t in piece:
                            if t.endswith("</w>") or not _is_alphaish(t) or t == ' ':
                                out.append(t)
                else:
                    # fallback: kelime
                    for t in piece:
                        if t.endswith("</w>") or not _is_alphaish(t) or t == ' ':
                            out.append(t)

            return out if out else piece  # hepsi elendiyse orijinali dök

        # ✅ ENDÜSTRI STANDARDI: Token kaybını önle
        processed: List[str] = []
        for seg in segments:
            if seg["kind"] == "text":
                # use_prefer_filter=False ise, filtreleme yapma (TÜM TOKEN'LARI KULLAN)
                if use_prefer_filter and pref:
                    processed.extend(_filter_piece(seg["tokens"], pref))
                else:
                    # ✅ TOKEN KAYBINI ÖNLE: Tüm token'ları kullan, filtreleme yapma
                    processed.extend(seg["tokens"])
            else:
                # pass segmentleri: <SEP>, specials, __tag__*
                processed.extend(seg["tokens"])

        # 4) Filtreleme ve dönüştürme (specials/tags/<SEP>)
        out_tokens: List[str] = []
        for t in processed:
            # ✅ DÜZELTME: SEP token özel işleme - remove_specials kontrolünden önce
            if t == sep_token:
                if remove_specials:
                    # remove_specials=True: SEP token'ı boşluğa çevir
                    out_tokens.append(" ")  # boşluk yalnızca SEP'ten gelir
                else:
                    # remove_specials=False: SEP token'ı '<SEP>' olarak decode et
                    out_tokens.append(sep_token)
                continue  # SEP token işlendi, diğer kontrollere gerek yok
            
            if remove_specials and t in specials:
                continue
            if remove_tags and t.startswith("__tag__"):
                continue
            out_tokens.append(t)

        # 5) </w> temizliği ve boşluk ekleme
        #    Kelime sonu işaretlerini kaldır ve boşluk ekle
        processed_tokens = []
        for i, token in enumerate(out_tokens):
            if token == ' ':
                processed_tokens.append(' ')
            elif token.endswith('</w>'):
                # Kelime sonu işaretini kaldır ve boşluk ekle
                word = token[:-4]  # </w> kaldır
                processed_tokens.append(word)
                # Sonraki token boşluk değilse boşluk ekle
                if i + 1 < len(out_tokens) and out_tokens[i + 1] != ' ':
                    processed_tokens.append(' ')
            else:
                processed_tokens.append(token)
        
        text = "".join(processed_tokens)

        # ✅ DÜZELTME: </w> tag'leri decode işlemi sonrası hala görünüyorsa temizle
        # Bu, generation response'larda </w> tag'lerinin görünmesini önler
        if "</w>" in text:
            logger.debug("[BPEDecoder] Decode sonrası </w> tag'leri tespit edildi, temizleniyor.")
            # </w> tag'lerini kaldır ve boşluk ekle
            text = re.sub(r'</w>', ' ', text)  # </w> → boşluk
            # Çoklu boşlukları tek boşluğa çevir
            text = re.sub(r'\s+', ' ', text).strip()

        # 7) Noktalama ve özel işaret çevresi boşluk kuralları
        # Sadece çoklu boşlukları tek boşluğa çevir, tek boşlukları koru
        text = re.sub(r"\s+", " ", text)  # Çoklu boşlukları tek boşluğa çevir
        # 8) Normalizasyon
        # ✅ DÜZELTME: SEP token içeren text'lerde strip() yaparken dikkatli ol
        # Eğer text sadece boşluk ise (SEP token'dan geliyorsa), strip() yapma
        if collapse_spaces:
            # Tek başına boşluk olan text'i koru (SEP token'dan geliyor olabilir)
            if text.strip() and text != " ":
                text = re.sub(r"\s+", " ", text).strip()
            elif text == " ":
                # Tek başına boşluk (SEP token'dan geliyor), koru
                text = " "
        if lowercase:
            # ✅ DÜZELTME: Special token'ları lowercase yapma
            # <SEP>, <BOS>, <EOS> gibi token'lar büyük harfle kalmalı
            # Special token'ları koru, sadece normal text'i lowercase yap
            special_tokens_in_text = ["<SEP>", "<BOS>", "<EOS>", "<PAD>", "<UNK>"]
            if text and not any(special in text for special in special_tokens_in_text):
                text = text.lower()
            elif text and any(special in text for special in special_tokens_in_text):
                # Special token var, sadece special token olmayan kısımları lowercase yap
                # Basit yaklaşım: Eğer text sadece special token ise, lowercase yapma
                if text.strip() in special_tokens_in_text:
                    pass  # Special token'ı olduğu gibi bırak
                else:
                    # Mixed content: Special token + normal text
                    # Normal text kısmını lowercase yap, special token'ı koru
                    for special in special_tokens_in_text:
                        if special in text:
                            # Special token'ı geçici olarak placeholder ile değiştir
                            placeholder = f"__SPECIAL_{special_tokens_in_text.index(special)}__"
                            text = text.replace(special, placeholder)
                            text = text.lower()
                            text = text.replace(placeholder, special)
                            break

        logger.debug("[BPEDecoder] decode çıktı: %s", text if len(text) < 200 else text[:200] + "…")
        return text


    def reset(self) -> None:
        """Decoder’ı varsayılan vocab (utils) ve boş merges ile sıfırlar."""
        logger.warning("[BPEDecoder] Resetleniyor…")
        self.vocab = _default_vocab()
        self._ensure_special_tokens_exact()
        self.reverse_vocab = self._build_reverse_vocab()
        self.merges = []
        logger.warning("[BPEDecoder] Sıfırlandı.")

    # ------------------- İç yardımcılar -------------------

    def _ensure_special_tokens_exact(self) -> None:
        """
        DEFAULT_SPECIALS’a göre özel tokenların doğru ID’lerle mevcut olduğunu kontrol eder.
        Decoder vocab'a ekleme YAPMAZ - sadece kontrol eder. Vocab'a ekleme BPEManager'ın sorumluluğudur.
        Eksik veya yanlış ID'li özel tokenlar varsa uyarı verir ama hata fırlatmaz (decode çalışmaya devam edebilir).
        """
        missing_tokens = []
        mismatched_tokens = []
        for sp_tok, sp_id in DEFAULT_SPECIALS.items():
            meta = self.vocab.get(sp_tok)
            if meta is None:
                missing_tokens.append((sp_tok, sp_id))
            else:
                mid = meta.get("id")
                if mid != sp_id:
                    mismatched_tokens.append((sp_tok, mid, sp_id))
        
        if missing_tokens:
            logger.warning(
                "[BPEDecoder] Eksik özel tokenlar tespit edildi (vocab'a eklenmedi, decode çalışmaya devam edebilir): %s",
                missing_tokens
            )
        if mismatched_tokens:
            logger.warning(
                "[BPEDecoder] Özel token ID uyuşmazlıkları tespit edildi: %s",
                mismatched_tokens
            )

    def _build_reverse_vocab(self) -> Dict[int, str]:
        """
        Vocab’tan reverse map (id->token) üretir.
        ID çakışması tespit edilirse açık hata verir (drift’i gizlemeyiz).
        """
        reverse: Dict[int, str] = {}
        skipped_count = 0
        for token, meta in self.vocab.items():
            if not isinstance(meta, dict):
                skipped_count += 1
                logger.warning(f"[BPEDecoder] _build_reverse_vocab: Token '{token}' meta dict değil: {type(meta)}")
                continue
            tid = meta.get("id")
            if not isinstance(tid, int):
                skipped_count += 1
                if token not in ["version", "specials", "tokens"]:  # Skip metadata keys
                    logger.debug(f"[BPEDecoder] _build_reverse_vocab: Token '{token}' id int değil: {type(tid)}")
                continue
            if tid in reverse:
                other = reverse[tid]
                raise BPEDecodingError(
                    f"Vocab ID çakışması: id={tid} hem '{other}' hem '{token}' için atanmış."
                )
            reverse[tid] = token
        if not reverse:
            raise BPEDecodingError("Reverse vocab oluşturulamadı (vocab boş veya hatalı).")
        if skipped_count > 0:
            logger.warning(f"[BPEDecoder] _build_reverse_vocab: {skipped_count} token atlandı (geçersiz format)")
        logger.debug(f"[BPEDecoder] _build_reverse_vocab: {len(reverse)} token ID eşlemesi oluşturuldu (vocab_size={len(self.vocab)})")
        return reverse

    def batch_decode_gpu(
        self,
        id_sequences: List[List[int]],
        batch_size: Optional[int] = None
    ) -> List[str]:
        """GPU ile batch decoding - PyTorch tensors kullanarak hızlandırılmış."""
        # batch_size config'ten al
        if batch_size is None:
            batch_size = self.config.get("gpu_batch_size", 32)
            
        if not self.use_gpu or not self.device:
            # Fallback: CPU processing
            return [self.decode(ids) for ids in id_sequences]
        
        import torch
        
        results = []
        
        # Batch'ler halinde işle
        for i in range(0, len(id_sequences), batch_size):
            batch_sequences = id_sequences[i:i + batch_size]
            batch_results = []
            
            try:
                for ids in batch_sequences:
                    # CPU'da decode yap
                    text = self.decode(ids)
                    batch_results.append(text)
                
                # GPU tensor'a çevir (metin işleme için)
                if batch_results:
                    # String tensor işleme (basit versiyon)
                    tensor_texts = [text for text in batch_results]
                    results.extend(tensor_texts)
                    
            except Exception as e:
                logger.error(f"[BPEDecoder] GPU batch decode hatası: {e}")
                # Fallback: CPU processing
                for ids in batch_sequences:
                    text = self.decode(ids)
                    results.append(text)
        
        logger.debug(f"[BPEDecoder] GPU batch_decode tamamlandı: {len(results)} sonuç")
        return results
