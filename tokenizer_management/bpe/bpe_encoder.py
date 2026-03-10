# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: bpe_encoder.py
Modül: tokenizer_management/bpe
Görev: BPEEncoder sınıfı - Metin → token ID dönüşümü (encoding). BPE merges
       kullanarak metni token'lara ayırır ve vocab'dan token ID'lerini döndürür.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (encoding işlemleri),
                     Dependency Inversion (BPEManager interface'i)
- Design Patterns: Strategy Pattern (farklı encoding stratejileri)
- Endüstri Standartları: GPT-2/3/4 BPE encoding, SentencePiece benzeri yaklaşım

KULLANIM:
- BPEManager.encode() tarafından kullanılır
- Metin → token ID dönüşümü için
- OOV (Out-of-Vocabulary) token yönetimi

BAĞIMLILIKLAR:
- bpe_manager_utils: Vocab ve merges yardımcı fonksiyonları
- BPE_DETAILED_CONFIG: Tokenization yapılandırması

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
import torch
from typing import Dict, List, Optional, Tuple, Iterable, Any

from .bpe_manager_utils import DEFAULT_SPECIALS, get_valid_ids
from tokenizer_management.config import (
    BPE_DETAILED_CONFIG,
    ENCODER_CONFIG,
    get_encoder_config,
)

logger = logging.getLogger(__name__)

# --- Heuristik parça ayırıcı -------------------------------------------------
# Kelime ve tekil noktalama/sembol: "asistan.,net,güvenli" -> ["asistan", ".", ",", "net", ",", "güvenli"]
_SPLIT_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_WORD_RE  = re.compile(r"^\w+$", flags=re.UNICODE)


class BPEEncodingError(Exception):
    """BPEEncoder ile ilgili hataları tanımlamak için özel exception."""
    pass


class BPEEncoder:
    """
    Tek sorumluluk: Verilen vocab + merges ile token(lar)ı ID’lere dönüştürmek.

    - Vocab sözleşmesi (normalize edilmiş):
      { token: { "id": int, "total_freq": int, "positions": List[int] } }

    - Merges sözleşmesi:
      List[Tuple[str, str]]  # (left, right) -> merge sırası (rank)
      Rank tabanı deterministik birleştirme için kullanılır (daha küçük rank daha önce gelir).

    Önemli notlar:
    - Bu encoder standard BPE davranışını uygular: bir girdi "token"ı, birden fazla alt-token ID’sine
      dönüşebilir (flatten). Bu yüzden encode_single içte çoklu ID üretebilir; geriye dönük uyumluluk
      için ilk ID’yi döndürür ve uyarı loglar. Üretimde encode_sequence kullanın.
    """

    def __init__(
        self,
        vocab: Dict[str, dict],
        merges: Optional[List[Tuple[str, str]]] = None,
        use_gpu: Optional[bool] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        if not isinstance(vocab, dict) or not vocab:
            raise TypeError("Vocab bir sözlük olmalı ve boş olmamalıdır.")

        # Config merge
        self.config = {**ENCODER_CONFIG}
        self.config.update(BPE_DETAILED_CONFIG)
        if config:
            self.config.update(config)

        self.vocab: Dict[str, dict] = dict(vocab)  # kopya
        # GPU support (config'ten, parametre sadece override için)
        if use_gpu is None:
            use_gpu = self.config.get("use_gpu", False)
        self.use_gpu = use_gpu
        
        # GPU device setup
        if self.use_gpu:
            try:
                import torch
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if self.device.type == "cpu":
                    logger.warning("[BPEEncoder] GPU isteniyor ama CUDA mevcut değil, CPU kullanılacak")
                    self.use_gpu = False
                else:
                    logger.info(f"[BPEEncoder] GPU desteği aktif: {self.device}")
            except ImportError:
                logger.warning("[BPEEncoder] PyTorch bulunamadı, GPU desteği devre dışı")
                self.use_gpu = False
                self.device = None
        else:
            self.device = None
        
        self._ensure_special_tokens()
        self._update_reverse_vocab()

        # Merges'i tuple listesine çevir
        self.merges: List[Tuple[str, str]] = []
        if merges:
            for merge in merges:
                if isinstance(merge, (list, tuple)) and len(merge) == 2:
                    self.merges.append((merge[0], merge[1]))
                elif isinstance(merge, str):
                    parts = merge.split()
                    if len(parts) == 2:
                        self.merges.append((parts[0], parts[1]))
        
        self._merge_ranks: Dict[Tuple[str, str], int] = self._build_merge_ranks(self.merges)

        # Hızlı erişim
        self._unk_id: Optional[int] = self._maybe_id("<UNK>")
        self._unk_char_id: Optional[int] = self._maybe_id("<UNK_CHAR>")

        logger.debug("[BPEEncoder] Başlatıldı | merges=%d", len(self.merges))

    # ----------------------- Kamu API -----------------------

    def set_vocab(self, new_vocab: Dict[str, dict]) -> None:
        """
        Vocab güncellemesi. Özel tokenların (DEFAULT_SPECIALS) ID’leriyle mevcut olduğundan emin olur.
        Çakışma/uygunsuzluk varsa açık hata fırlatır (sessiz drift yok).
        """
        if not isinstance(new_vocab, dict) or not new_vocab:
            raise TypeError("new_vocab bir sözlük olmalı ve boş olmamalıdır.")

        self.vocab = dict(new_vocab)
        self._ensure_special_tokens()
        self._update_reverse_vocab()
        self._unk_id = self._maybe_id("<UNK>")
        self._unk_char_id = self._maybe_id("<UNK_CHAR>")
        logger.debug("[BPEEncoder] vocab güncellendi | size=%d", len(self.vocab))

    def set_merges(self, merges: Optional[List[Tuple[str, str]]]) -> None:
        """Öğrenilmiş merges listesini günceller ve rank tablosunu yeniden kurar."""
        self.merges = list(merges) if merges else []
        self._merge_ranks = self._build_merge_ranks(self.merges)
        logger.debug("[BPEEncoder] merges güncellendi | count=%d", len(self.merges))

    def encode_single(self, token: str) -> int:
        """
        Tek bir token’ı ID’ye çevirir (geriye dönük uyumluluk).
        Not: Standard BPE, tek "token"dan çoklu alt-token ID üretebilir. Bu durumda ilk ID döner.
        Üretimde encode_sequence kullanılması tavsiye edilir.
        """
        ids = self._encode_token_to_ids(token)
        if not ids:
            raise BPEEncodingError(f"encode_single: token '{token}' boş çıktı üretti.")
        if len(ids) > 1:
            logger.warning("[BPEEncoder] encode_single: '%s' -> çoklu parça %s; ilk id=%d döndü",
                           token, ids, ids[0])
        return ids[0]

    def encode(self, tokens: List[str], mode: str = "train") -> List[int]:
        """
        Token listesini ID listesine dönüştürür.
        Not: mode parametresi şimdilik yalnızca API uyumu içindir; encoder read-only çalışır.
        """
        if mode not in ("train", "inference"):
            raise ValueError("mode 'train' veya 'inference' olmalıdır.")
        return self.encode_sequence(tokens)

    def encode_sequence(self, tokens: List[str]) -> List[int]:
        """Liste halindeki tokenları (her biri bir "kelime/altkelime" olabilir) ID’lere çevirir ve düzleştirir."""
        if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
            raise TypeError("encode_sequence: Girdi List[str] olmalıdır.")
        if not tokens:
            raise ValueError("encode_sequence: boş liste kabul edilmez.")

        flat_ids: List[int] = []
        for t in tokens:
            piece_ids = self._encode_token_to_ids(t)
            if not piece_ids:
                raise BPEEncodingError(f"encode_sequence: token '{t}' için boş çıktı.")
            flat_ids.extend(piece_ids)

        # Üyelik-tabanlı doğrulama (len(vocab) varsayımı yok)
        valid_ids = get_valid_ids(self.vocab)
        for tid in flat_ids:
            if tid not in valid_ids:
                raise BPEEncodingError(f"Geçersiz token id: {tid} (vocab ile uyumsuz)")

        logger.debug("[BPEEncoder] encode_sequence -> %s", flat_ids)
        return flat_ids

    def reset(self) -> None:
        """
        Encoder’ı merges’i temizlenmiş ve yalnızca özel tokenları koruyan minimal bir duruma getirir.
        (Eğitim dışı senaryolarda acil kurtarma için.)
        """
        logger.warning("[BPEEncoder] Resetleniyor...")
        preserved = {tok: self.vocab[tok] for tok in DEFAULT_SPECIALS if tok in self.vocab}
        self.vocab = preserved
        self._update_reverse_vocab()
        self.merges = []
        self._merge_ranks = {}
        self._unk_id = self._maybe_id("<UNK>")
        self._unk_char_id = self._maybe_id("<UNK_CHAR>")
        logger.warning("[BPEEncoder] Sıfırlandı. specials=%d", len(self.vocab))

    # ----------------------- İç yardımcılar -----------------------

    def _maybe_id(self, tok: str) -> Optional[int]:
        meta = self.vocab.get(tok)
        if isinstance(meta, dict) and "id" in meta:
            try:
                return int(meta["id"])
            except Exception:
                return None
        return None

    def _is_special(self, tok: str) -> bool:
        return tok in DEFAULT_SPECIALS

    def _ensure_special_tokens(self) -> None:
        """
        DEFAULT_SPECIALS’a göre özel tokenların sözlükte ve doğru ID ile bulunduğunu garanti eder.
        ID çakışması varsa açık hata fırlatır (sessizce yeni ID vermeyiz).
        """
        # 1) Mevcut ID kümesi
        existing_ids = {meta.get("id") for meta in self.vocab.values() if isinstance(meta, dict) and "id" in meta}

        # 2) Her özel tokenı doğrula/ekle
        for sp_tok, sp_id in DEFAULT_SPECIALS.items():
            meta = self.vocab.get(sp_tok)
            if meta is None:
                # Token yoksa ekle; ID çakışması kontrolü
                if sp_id in existing_ids:
                    raise BPEEncodingError(
                        f"Özel token ID çakışması: {sp_tok} için {sp_id} zaten kullanılıyor."
                    )
                self.vocab[sp_tok] = {"id": sp_id, "total_freq": 0, "positions": []}
                existing_ids.add(sp_id)
                logger.info("[BPEEncoder] Eksik özel token eklendi: %s -> id=%d", sp_tok, sp_id)
            else:
                # Var ama id yanlışsa hata
                mid = meta.get("id")
                if mid != sp_id:
                    raise BPEEncodingError(
                        f"Özel token ID uyuşmazlığı: {sp_tok} id={mid}, beklenen={sp_id}. "
                        "Vocab/konfig sürtüşmesi var."
                    )

    def _update_reverse_vocab(self) -> None:
        if not self.vocab:
            raise BPEEncodingError("Vocab boş, reverse vocab oluşturulamadı.")
        self.reverse_vocab: Dict[int, str] = {
            int(meta["id"]): tok for tok, meta in self.vocab.items() if isinstance(meta, dict) and "id" in meta
        }
        if not self.reverse_vocab:
            raise BPEEncodingError("Reverse vocab boş; vocab meta hatalı görünüyor.")
        logger.debug("[BPEEncoder] reverse_vocab yenilendi | size=%d", len(self.reverse_vocab))

    @staticmethod
    def _build_merge_ranks(merges: List[Tuple[str, str]]) -> Dict[Tuple[str, str], int]:
        """
        Merges listesinden deterministik bir rank tablosu kurar.
        Daha küçük rank, daha öncelikli birleştirme demektir.
        """
        ranks: Dict[Tuple[str, str], int] = {}
        for i, pair in enumerate(merges):
            if not (isinstance(pair, tuple) and len(pair) == 2 and all(isinstance(x, str) for x in pair)):
                raise BPEEncodingError(f"Geçersiz merge çifti: {pair!r}")
            ranks[(pair[0], pair[1])] = i
        return ranks

    # ----------------------- BPE çekirdeği -----------------------

    def _encode_token_to_ids(self, token: str) -> List[int]:
        """
        Tek bir "token" stringini bir veya daha fazla alt-token ID’sine çevirir.
        Strateji:
        1) Doğrudan vocab isabeti → tek ID.
        2) Özel token → tek ID.
        3) BPE birleştirme (karakter + '</w>' ile) → alt-semboller → id listesi.
        4) Hâlâ başarısızsa heuristik böl → parça başına 1–3 adımlar.
        5) En son çare: karakter-düzeyi gerileme (</w> aktarımı), <UNK_CHAR>→<UNK>.
        """
        if not isinstance(token, str):
            raise TypeError("encode: token str olmalıdır.")
        if token == "":
            return [self._require_unk("boş token")]

        # GPU ile işleme
        # NOT: GPU encoding sadece tek karakterli token'lar için çalışır
        # Çok karakterli token'lar (whole words, özel tokenlar) için CPU fallback kullan
        if self.use_gpu and self.device:
            # Önce vocab'da doğrudan kontrol et (GPU'ya göndermeden önce)
            meta = self.vocab.get(token)
            if isinstance(meta, dict) and "id" in meta:
                return [int(meta["id"])]
            if self._is_special(token):
                mid = self._maybe_id(token)
                if mid is not None:
                    return [mid]
            
            # Çok karakterli token'lar için CPU fallback (GPU encoding desteklemiyor)
            # Sadece tek karakterli token'lar için GPU kullan
            if len(token) == 1:
                try:
                    import torch
                    with torch.amp.autocast('cuda'):
                        # Tokenı karakter tensoruna çevir
                        chars = torch.tensor([ord(c) for c in token], device=self.device, dtype=torch.long)
                        ids = self._bpe_ids_for_token_gpu(chars)
                        if ids.numel() > 0:
                            return ids.cpu().tolist()
                except Exception as e:
                    logger.debug(f"[BPEEncoder] GPU encoding başarısız, CPU fallback: {e}")
            # Çok karakterli token'lar için direkt CPU fallback'e geç

        # CPU fallback
        # 0) Doğrudan/special
        meta = self.vocab.get(token)
        if isinstance(meta, dict) and "id" in meta:
            return [int(meta["id"])]
        if self._is_special(token):
            mid = self._maybe_id(token)
            if mid is not None:
                return [mid]

        # 1) BPE dene
        ids = self._bpe_ids_for_token(token)
        if ids:
            return ids

        # 2) Heuristik split dene (örn: "asistan.,net,güvenli</w>")
        pieces = self._split_preserving_eow(token)
        if len(pieces) > 1:
            out: List[int] = []
            for p in pieces:
                sub_ids = self._bpe_ids_for_token(p)
                if sub_ids:
                    out.extend(sub_ids)
                else:
                    out.extend(self._char_fallback_ids(p))
            if out:
                logger.debug("[BPEEncoder] Heuristik split ile çözüldü: '%s' -> %s", token, out)
                return out

        # 3) Son çare: karakter düzeyi
        cf = self._char_fallback_ids(token)
        logger.debug("[BPEEncoder] Karakter fallback: '%s' -> %s", token, cf)
        return cf

    def _bpe_ids_for_token(self, token: str) -> List[int]:
        """
        Standard BPE: başlangıçta karakter listesi + '</w>' (eğer yoksa),
        komşu çiftlerden rank’ı en küçük olanları sırayla birleştir,
        sonunda elde edilen sembolleri vocab’ten ID’ye çevir.
        Her sembolün vocab’te olmaması durumunda parçayı daha küçük alt parçalara
        (karakter düzeyi) geriletir.
        """
        symbols = self._token_to_symbols(token)
        if not self._merge_ranks or len(symbols) <= 1:
            # merges yoksa doğrudan maplemeyi deneriz
            return self._map_symbols_to_ids(symbols)

        # GPU ile işleme
        if self.use_gpu and self.device:
            import torch
            try:
                with torch.amp.autocast('cuda'):
                    # Çok karakterli token'ları (özel tokenlar) tespit et
                    # Eğer symbols içinde çok karakterli token varsa, CPU fallback kullan
                    has_multi_char = any(len(s) > 1 for s in symbols)
                    if has_multi_char:
                        # GPU encoding çok karakterli token'ları desteklemiyor, CPU fallback
                        raise ValueError("Multi-character tokens detected, using CPU fallback")
                    
                    # Symbols'ü GPU tensor'ına çevir (string'leri ord ile kodla)
                    symbols_tensor = torch.tensor([ord(c) for c in ''.join(symbols)], device=self.device, dtype=torch.long)
                    ids = self._bpe_ids_for_token_gpu(symbols_tensor)
                    return ids.cpu().tolist() if ids.numel() > 0 else []
            except (ValueError, TypeError) as e:
                # GPU encoding başarısız oldu, CPU fallback kullan
                logger.debug(f"[BPEEncoder] GPU encoding başarısız, CPU fallback: {e}")
                # CPU fallback'e devam et (aşağıdaki kod)

        # CPU fallback
        # En iyi komşu çifti bul
        def best_pair(seq: List[str]) -> Optional[Tuple[int, Tuple[str, str], int]]:
            best_idx = -1
            best_pair_: Optional[Tuple[str, str]] = None
            best_rank = None
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                rank = self._merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair_ = pair
                    best_idx = i
            if best_pair_ is None:
                return None
            return best_idx, best_pair_, int(best_rank)  # type: ignore[arg-type]

        seq = list(symbols)
        while True:
            found = best_pair(seq)
            if found is None:
                break
            i, (a, b), _rank = found
            merged = a + b
            seq[i:i+2] = [merged]

        return self._map_symbols_to_ids(seq)

    def _bpe_ids_for_token_gpu(self, tokens_tensor: torch.Tensor) -> torch.Tensor:
        """
        GPU ile BPE birleştirme işlemini gerçekleştirir.
        Tensor girişi (karakter ASCII kodları) alır ve ID'ler tensor'ını döndürür.
        """
        if not self.use_gpu or not self.device:
            raise ValueError("GPU desteği devre dışı veya cihaz ayarlanmamış.")

        with torch.amp.autocast('cuda'):
            try:
                symbols = self._token_to_symbols_gpu(tokens_tensor)
                if len(symbols) <= 1 or not self._merge_ranks:
                    return self._map_symbols_to_ids_gpu(symbols)

                seq = symbols.clone()
                while True:
                    best_pairs = self._find_best_pairs_gpu(seq)
                    if best_pairs is None or best_pairs.size(0) == 0:
                        break
                    # Tensor bazlı birleştirme
                    indices = best_pairs[:, 0].long()
                    merged = torch.cat((seq[indices], seq[indices + 1]), dim=1)
                    seq = torch.cat((seq[:indices[0]], merged, seq[indices[0] + 2:]))
                    if seq.size(0) < 2:
                        break

                return self._map_symbols_to_ids_gpu(seq)
            except Exception as e:
                logger.error(f"[BPEEncoder] GPU BPE hatası: {e}")
                return torch.tensor([], device=self.device)

    def _token_to_symbols_gpu(self, tokens_tensor: torch.Tensor) -> torch.Tensor:
        """
        Tokenı semboller tensor'una çevirir (GPU uyumlu).
        '</w>' ekleme mantığını tensor bazında uygular.
        NOT: Bu metod sadece tek karakterli token'lar için çalışır.
        """
        if not self.use_gpu or not self.device:
            raise ValueError("GPU desteği devre dışı.")
        symbols = tokens_tensor.clone()
        # '</w>' karakter bazlı değil, bu yüzden burada eklenmemeli
        # Bu metod sadece karakter bazlı token'lar için kullanılmalı
        return symbols

    def _find_best_pairs_gpu(self, seq: torch.Tensor) -> Optional[torch.Tensor]:
        """
        GPU ile en iyi komşu çiftleri bulur (paralel arama).
        """
        if seq.size(0) < 2:
            return None

        with torch.amp.autocast('cuda'):
            indices = torch.arange(seq.size(0) - 1, device=self.device)
            pairs = torch.stack((seq[indices], seq[indices + 1]), dim=1)
            pair_strs = [(chr(p[0].item()), chr(p[1].item())) for p in pairs]
            ranks = torch.tensor([self._merge_ranks.get(p, float('inf')) for p in pair_strs], device=self.device)
            if torch.all(ranks == float('inf')):
                return None
            min_idx = torch.argmin(ranks)
            return torch.stack((pairs[min_idx], torch.tensor([min_idx], device=self.device)))

    def _map_symbols_to_ids_gpu(self, symbols: torch.Tensor) -> torch.Tensor:
        """
        Sembolleri GPU ile ID'lere çevirir.
        """
        if not self.use_gpu or not self.device:
            raise ValueError("GPU desteği devre dışı.")

        with torch.amp.autocast('cuda'):
            ids = torch.zeros(symbols.size(0), dtype=torch.long, device=self.device)
            for i, sym in enumerate(symbols):
                char = chr(sym.item())
                meta = self.vocab.get(char)
                if isinstance(meta, dict) and "id" in meta:
                    ids[i] = int(meta["id"])
                else:
                    ids[i] = self._unk_char_id if self._unk_char_id is not None else self._require_unk(f"char:{char}")
            return ids

    def _map_symbols_to_ids(self, symbols: Iterable[str]) -> List[int]:
        """
        Her sembolü vocab’te arar, yoksa karakter-düzeyi gerileme uygular.
        Sembol örnekleri: "gü", "ven", "li</w>", ".", ",", "</w>"
        """
        out: List[int] = []
        for sym in symbols:
            meta = self.vocab.get(sym)
            if isinstance(meta, dict) and "id" in meta:
                out.append(int(meta["id"]))
                continue
            # Sembol doğrudan yoksa, daha küçük birimlere parçala (char fallback, </w> korunur)
            out.extend(self._char_fallback_ids(sym))
        return out

    def _char_fallback_ids(self, piece: str) -> List[int]:
        """
        Bir parça vocab'te yoksa karakterlere geriler (endüstri standardı: GPT-2/GPT-3/GPT-4).
        
        ENDÜSTRİ STANDARDI:
        - Karakterler her zaman vocab'te olmalı (BPE training sırasında eklenmeli)
        - Word boundary (`</w>`) bilgisi korunur ama ayrı token olarak eklenir
        - Vocab'da sadece tek karakterler var, `char + '</w>'` formları YOK (gereksiz vocab büyümesi)
        
        Örnek:
        - "kelime</w>" vocab'te yoksa -> ['k', 'e', 'l', 'i', 'm', 'e', '</w>'] (7 token)
        - "kelime" vocab'te yoksa -> ['k', 'e', 'l', 'i', 'm', 'e'] (6 token, word boundary yok)
        """
        if piece == "":
            return [self._require_unk("empty-piece")]

        ids: List[int] = []
        
        # Word boundary kontrolü
        has_word_boundary = piece.endswith("</w>")
        
        if has_word_boundary:
            base = piece[:-4]  # "</w>" kısmını çıkar
            if base == "":
                # Yalnızca '</w>' token'ı
                meta = self.vocab.get("</w>")
                if isinstance(meta, dict) and "id" in meta:
                    return [int(meta["id"])]
                return [self._require_unk("bare-</w>")]
            
            # Base kelimeyi karakterlere parçala
            chars = list(base)
            # Tüm karakterleri encode et (vocab'da sadece tek karakterler var)
            for c in chars:
                ids.append(self._lookup_char(c))
            
            # Word boundary bilgisini koru: ayrı bir '</w>' token'ı ekle
            # Bu sayede decode sırasında word boundary geri kazanılabilir
            meta = self.vocab.get("</w>")
            if isinstance(meta, dict) and "id" in meta:
                ids.append(int(meta["id"]))
            # Eğer '</w>' vocab'te yoksa, bu bir bug (BPE training sırasında eklenmeli)
            # Ama yine de devam et (UNK ekleme, sadece word boundary kaybolur)
        else:
            # Word boundary yok, sadece karakterleri encode et
            for c in piece:
                ids.append(self._lookup_char(c))
        
        return ids

    def _lookup_char(self, ch: str) -> int:
        """
        Karakteri vocab'te arar (endüstri standardı - GPT, BERT gibi).
        
        ENDÜSTRİ STANDARDI:
        - Base alphabet karakterleri vocab başlangıcında otomatik eklenir
        - Karakterler SADECE tek karakter formatında ("a", "a</w>" değil)
        - Fallback: Eğer "char" yoksa "char</w>" formatını da kontrol et
        
        Args:
            ch: Aranacak karakter (örn: "a", "i", "e")
            
        Returns:
            int: Karakter ID'si veya UNK ID'si
        """
        # 1. Önce sadece karakteri ara (base alphabet - endüstri standardı)
        meta = self.vocab.get(ch)
        if isinstance(meta, dict) and "id" in meta:
            return int(meta["id"])
        
        # 2. Fallback: char + </w> formatını kontrol et
        # (Vocab'ta yanlışlıkla char</w> formatında eklenmişse - backward compatibility)
        ch_with_suffix = ch + "</w>"
        meta = self.vocab.get(ch_with_suffix)
        if isinstance(meta, dict) and "id" in meta:
            logger.debug(f"[BPEEncoder] Fallback: Karakter '{ch}' bulunamadı, '{ch_with_suffix}' formatı kullanıldı (ID: {meta['id']})")
            return int(meta["id"])
        
        # 3. UNK döndür (karakter vocab'te yok)
        if self._unk_char_id is not None:
            logger.warning("[BPEEncoder] Karakter vocab'te yok: '%s' -> <UNK_CHAR>(id=%d). Base alphabet otomatik eklenmeli!", ch, self._unk_char_id)
            return self._unk_char_id
        logger.warning("[BPEEncoder] Karakter vocab'te yok ve UNK_CHAR da yok: '%s' -> <UNK>. Base alphabet otomatik eklenmeli!", ch)
        return self._require_unk(f"char:{ch}")

    def _require_unk(self, ctx: str) -> int:
        if self._unk_id is None:
            raise BPEEncodingError(f"<UNK> yok; {ctx} çözümlenemedi.")
        logger.debug("[BPEEncoder] Fallback -> <UNK>(id=%d) | ctx=%s", self._unk_id, ctx)
        return self._unk_id

    def batch_encode_sequence_gpu(
        self,
        token_sequences: List[List[str]],
        batch_size: Optional[int] = None
    ) -> List[List[int]]:
        """GPU ile batch sequence encoding - PyTorch tensors kullanarak hızlandırılmış."""
        # batch_size config'ten al
        if batch_size is None:
            batch_size = self.config.get("gpu_batch_size", 32)
            
        if not self.use_gpu or not self.device:
            return [self.encode_sequence(tokens) for tokens in token_sequences]
        
        import torch
        
        results = []
        for i in range(0, len(token_sequences), batch_size):
            batch_sequences = token_sequences[i:i + batch_size]
            batch_tensors = [torch.tensor([ord(c) for t in seq for c in t], device=self.device) for seq in batch_sequences]
            
            try:
                batch_results = [self._encode_token_to_ids_gpu(t).cpu().tolist() for t in batch_tensors]
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"[BPEEncoder] GPU batch processing hatası: {e}")
                results.extend([self.encode_sequence(seq) for seq in batch_sequences])
        
        logger.debug(f"[BPEEncoder] GPU batch_encode_sequence tamamlandı: {len(results)} sonuç")
        return results

    # ----------------------- Sembolizasyon & Split -----------------------

    def _token_to_symbols(self, token: str) -> List[str]:
        """
        Girdi tokenını başlangıç sembollerine dönüştürür:
        - Eğer token zaten '</w>' ile bitiyorsa, onu korur.
        - Aksi halde sonuna '</w>' sembolünü ayrı bir parça olarak ekler.
        - Özel tokenlar olduğu gibi döner.
        """
        if self._is_special(token):
            return [token]

        if token.endswith("</w>"):
            base = token[:-4]
            if base == "":
                return ["</w>"]
            symbols = list(base) + ["</w>"]
        else:
            symbols = list(token) + ["</w>"]
        return symbols

    def _split_preserving_eow(self, token: str) -> List[str]:
        """
        Heuristik parçalama: word characters (\\w+) ve tekil noktalama/semboller.
        Token '</w>' ile bitiyorsa bu işareti son kelime parçasına aktarır.
        """
        had_eow = token.endswith("</w>")
        core = token[:-4] if had_eow else token
        parts = _SPLIT_RE.findall(core)
        if not parts:
            return [token]

        if had_eow:
            # '</w>' son "kelime" parçasına (varsa) eklenir; yoksa en sona eklenir
            for i in range(len(parts) - 1, -1, -1):
                if _WORD_RE.match(parts[i]):
                    parts[i] = parts[i] + "</w>"
                    break
            else:
                parts[-1] = parts[-1] + "</w>"
        return parts
