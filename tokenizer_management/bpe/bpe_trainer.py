# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: bpe_trainer.py
Modül: tokenizer_management/bpe
Görev: BPETrainer sınıfı - BPE (Byte Pair Encoding) algoritması ile vocab ve
       merges dosyalarını oluşturur. Veri setinden en sık kullanılan token
       çiftlerini bulur ve birleştirir.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (BPE training işlemleri),
                     Dependency Inversion (BPEManager interface'i)
- Design Patterns: Strategy Pattern (farklı training stratejileri)
- Endüstri Standartları: GPT-2/3/4 BPE training algoritması,
                         SentencePiece benzeri yaklaşım

KULLANIM:
- BPEManager.train() tarafından kullanılır
- Yeni vocab/merges oluşturma için
- Mevcut vocab'ı genişletme için

BAĞIMLILIKLAR:
- torch: GPU/CPU training desteği
- numpy: Hesaplamalı işlemler
- bpe_manager_utils: Vocab yardımcı fonksiyonları
- TRAINER_CONFIG: Training yapılandırması
- BPETrainingError: Özel exception sınıfı

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations

import copy
import logging
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np

from .bpe_manager_utils import (
    DEFAULT_SPECIALS,
    default_vocab as _default_vocab_utils,
    next_id as _next_id,
)
from tokenizer_management.config import (
    BPE_DETAILED_CONFIG,
    TRAINER_CONFIG,
    get_trainer_config,
)

logger = logging.getLogger(__name__)


class BPETrainingError(Exception):
    """BPETrainer ile ilgili hataları tanımlamak için özel exception."""
    pass


class BPETrainer:
    """
    Sorumluluk: BPE merges listesini ve (gerekirse) vocab genişlemesini, deterministik ve
    tekrar üretilebilir şekilde gerçekleştirmek.

    Sözleşmeler:
    - vocab: normalize edilmiş sözlük
      { token: { "id": int, "total_freq": int, "positions": List[int] } }
    - merges: List[Tuple[str, str]]  # (left, right), sıra deterministik rank'ı ifade eder
    - GPU support: Optional GPU acceleration for training
    """

    def __init__(self, vocab: Dict[str, dict], use_gpu: Optional[bool] = None, config: Optional[Dict[str, Any]] = None) -> None:
        if not isinstance(vocab, dict):
            raise TypeError("Vocab bir sözlük olmalıdır.")
        self.vocab: Dict[str, dict] = copy.deepcopy(vocab) if vocab else _default_vocab_utils()
        self._ensure_special_tokens_exact()  # Sabit ID'ler ile
        self._merges: List[Tuple[str, str]] = []
        self._merge_ranks: Dict[Tuple[str, str], int] = {}
        
        # ============================================================================
        # CONFIG MERGE: TRAINER_CONFIG + override mekanizması
        # ============================================================================
        self.config = {**TRAINER_CONFIG}  # Default: Trainer config
        self.config.update(BPE_DETAILED_CONFIG)  # Merge: Detaylı BPE config
        if config:
            self.config.update(config)  # Override: Kullanıcı config'i
        
        # Config validation
        min_vocab_size = self.config.get("min_vocab_size", 30000)
        max_vocab_size = self.config.get("max_vocab_size", 60000)
        if min_vocab_size >= max_vocab_size:
            raise BPETrainingError(f"Config hatası: min_vocab_size ({min_vocab_size}) >= max_vocab_size ({max_vocab_size}) olmamalı!")
        if max_vocab_size <= 0:
            raise BPETrainingError(f"Config hatası: max_vocab_size ({max_vocab_size}) > 0 olmalı!")
        if min_vocab_size < 0:
            raise BPETrainingError(f"Config hatası: min_vocab_size ({min_vocab_size}) >= 0 olmalı!")
        
        # GPU support (config'ten, parametre sadece override için)
        if use_gpu is None:
            use_gpu = self.config.get("use_gpu", False)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        
        # Training parametreleri (config'ten - artık hardcoded değil!)
        self.max_iter = self.config.get("max_iter", 50000)  # KRİTİK: 50000! (1000 değil)
        self.min_frequency = self.config.get("min_frequency", 2)
        self.chunk_size = self.config.get("chunk_size", 2000)
        self.log_interval = self.config.get("log_interval", 30.0)
        self.batch_size = self.config.get("batch_size", 30000)
        
        if self.use_gpu:
            logger.info(f"[BPETrainer] GPU training enabled: {self.device}")
        else:
            logger.info("[BPETrainer] CPU training (GPU not available or disabled)")
        
        logger.info(f"[BPETrainer] Başlatıldı | vocab_size={len(self.vocab)}, max_iter={self.max_iter}")  # max_iter logla!

    # -------------------- Kamu API --------------------

    def set_merges(self, merges: Optional[List[Tuple[str, str]]]) -> None:
        """
        Dışarıdan öğrenilmiş merges’leri yükler (ör. diskten).
        """
        incoming = list(merges) if merges else []
        for pair in incoming:
            if not (isinstance(pair, tuple) and len(pair) == 2 and all(isinstance(x, str) for x in pair)):
                raise BPETrainingError(f"Geçersiz merge çifti: {pair!r}")
        self._merges = incoming
        self._merge_ranks = self._build_merge_ranks(self._merges)
        logger.debug("[BPETrainer] merges yüklendi | count=%d", len(self._merges))

    def train(
        self,
        tokenized_corpus: List[List[str]],
        *,
        target_merges: Optional[int] = None,
        max_iter: Optional[int] = None,
        min_frequency: Optional[int] = None,
        append_eos: Optional[bool] = None,
        protect_specials: bool = True,
    ) -> None:
        """
        Deterministik BPE eğitimi.
        - Mevcut merges (set_merges ile verilen) baştan uygulanır.
        - Her iterasyonda en sık görülen komşu çift seçilir; eşitlikte leksikografik küçük olan tercih edilir.
        - Yeni birleştirilmiş token vocab'ta yoksa `next_id` ile eklenir.

        Args:
            tokenized_corpus: List[List[str]] biçiminde önceden tokenleştirilmiş satırlar.
            target_merges: İstenirse toplam merge sayısını sınırlar; None ise yalnızca istatistik bittiğinde durur.
            max_iter: Emniyet tavanı (sonsuz döngüleri önler).
            min_frequency: Bir çiftin merge adayı sayılması için asgari frekans.
            append_eos: Her satır sonuna "<EOS>" ekle (deterministik sınır belirtkesi).
            protect_specials: Özel tokenlar komşu çiftlerde asla merge edilmez.
        """
        import multiprocessing
        import concurrent.futures
        import time
        import psutil
        
        if not tokenized_corpus or not all(isinstance(seq, list) and all(isinstance(t, str) for t in seq)
                                           for seq in tokenized_corpus):
            raise ValueError("Corpus, List[List[str]] ve öğeleri str olmalıdır.")

        # Parametreleri config'ten al (None ise)
        if max_iter is None:
            max_iter = self.config.get("max_iter", 50000)
        if min_frequency is None:
            min_frequency = self.config.get("min_frequency", 2)
        if append_eos is None:
            append_eos = True

        # Başlangıç zamanı ve bellek izleme
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        logger.info(f"[BPETrainer] Eğitim başlıyor - Corpus: {len(tokenized_corpus)} cümle")
        logger.info(f"[BPETrainer] Config: max_iter={max_iter}, min_frequency={min_frequency}")
        logger.info(f"[BPETrainer] Başlangıç bellek kullanımı: {initial_memory:.2f} MB")

        # 1) Korpusu tek sequence'te birleştir (sırayı koru)
        logger.info("[BPETrainer] Corpus birleştiriliyor...")
        sequence: List[str] = []
        if append_eos:
            for seq in tokenized_corpus:
                sequence.extend(seq)
                sequence.append("<EOS>")
        else:
            for seq in tokenized_corpus:
                sequence.extend(seq)
        
        total_tokens = len(sequence)
        logger.info(f"[BPETrainer] Corpus birleştirildi - Toplam token: {total_tokens:,}")

        # 2) Varsa mevcut merges'i sırayla uygula (deterministik)
        if self._merges:
            logger.info(f"[BPETrainer] Mevcut {len(self._merges)} merge uygulanıyor...")
            sequence = self._apply_merges_to_sequence(sequence, self._merge_ranks)
            logger.info(f"[BPETrainer] Mevcut merges uygulandı - Yeni sequence uzunluğu: {len(sequence):,}")

        merges_done = len(self._merges)
        target = target_merges if (isinstance(target_merges, int) and target_merges > 0) else None
        
        logger.info(f"[BPETrainer] Hedef merge sayısı: {target if target else 'Sınırsız'}")
        logger.info(f"[BPETrainer] Maksimum iterasyon: {max_iter}")
        logger.info(f"[BPETrainer] Minimum frekans: {min_frequency}")

        # 3) GPU/CPU Batch merge processing
        last_log_time = time.time()
        log_interval = self.config.get("log_interval", 30.0)  # Config'ten log interval
        
        # GPU batch processing kullan
        if self.use_gpu:
            logger.info("[BPETrainer] GPU batch processing ile merge training başlıyor...")
            self._train_gpu_batch(
                sequence, target, max_iter, min_frequency, 
                merges_done, protect_specials, last_log_time, log_interval
            )
        else:
            logger.info("[BPETrainer] CPU sequential processing ile merge training başlıyor...")
            self._train_cpu_sequential(
                sequence, target, max_iter, min_frequency, 
                merges_done, protect_specials, last_log_time, log_interval
            )
        
        # Final rapor
        final_time = time.time()
        total_elapsed = final_time - start_time
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Vocab size kontrolü (config'ten limitler)
        final_vocab_size = len(self.vocab)
        min_vocab_size = self.config.get("min_vocab_size", 30000)
        max_vocab_size = self.config.get("max_vocab_size", 60000)
        
        logger.info(f"[BPETrainer] Eğitim tamamlandı!")
        logger.info(f"[BPETrainer] Toplam süre: {total_elapsed:.2f} saniye")
        logger.info(f"[BPETrainer] Toplam merge: {len(self._merges)}")
        logger.info(f"[BPETrainer] Final vocab size: {final_vocab_size}")
        logger.info(f"[BPETrainer] Final bellek: {final_memory:.2f} MB (+{memory_increase:.2f} MB)")
        if total_elapsed > 0:
            logger.info(f"[BPETrainer] Ortalama merge hızı: {len(self._merges)/total_elapsed:.2f} merge/saniye")
        else:
            logger.info(f"[BPETrainer] Ortalama merge hızı: N/A (çok hızlı, <0.01 saniye)")
        
        # Final GPU memory cleanup
        if self.use_gpu:
            torch.cuda.empty_cache()
        
        # Vocab size limit kontrolü ve düzeltme
        if final_vocab_size < min_vocab_size:
            logger.warning(f"[BPETrainer] UYARI: Vocab size çok küçük! {final_vocab_size} < {min_vocab_size} (min)")
        elif final_vocab_size > max_vocab_size:
            logger.error(f"[BPETrainer] HATA: Vocab size limit aşıldı! {final_vocab_size} > {max_vocab_size} (max)")
            logger.error(f"[BPETrainer] Final vocab size max_vocab_size'ı aşmamalı! Bu bir bug!")
            
            # Fazla token'ları kaldır (en düşük frekanslı olanları)
            excess = final_vocab_size - max_vocab_size
            logger.warning(f"[BPETrainer] {excess} fazla token kaldırılıyor...")
            
            # En düşük frekanslı token'ları bul ve kaldır
            from tokenizer_management.bpe.bpe_manager_utils import DEFAULT_SPECIALS
            special_tokens = set(DEFAULT_SPECIALS.keys())
            
            # Special token'ları hariç tut, sadece normal token'ları sırala
            normal_tokens = [(token, data) for token, data in self.vocab.items() if token not in special_tokens]
            sorted_tokens = sorted(normal_tokens, key=lambda x: x[1].get("total_freq", 0))
            
            # En düşük frekanslı token'ları kaldır
            removed_count = 0
            for token, _ in sorted_tokens[:excess]:
                if token in self.vocab:
                    del self.vocab[token]
                    removed_count += 1
            
            # Merge listesinden de kaldır (bu token'ları içeren merge'ler)
            # NOT: Bu karmaşık olabilir, şimdilik sadece vocab'tan kaldırıyoruz
            
            final_vocab_size = len(self.vocab)
            logger.warning(f"[BPETrainer] {removed_count} token kaldırıldı. Final vocab: {final_vocab_size}")
            
            if final_vocab_size > max_vocab_size:
                logger.error(f"[BPETrainer] HATA: Vocab size hala limit aşıyor! {final_vocab_size} > {max_vocab_size}")
            else:
                logger.info(f"[BPETrainer] Vocab size düzeltildi: {final_vocab_size} <= {max_vocab_size}")
        else:
            logger.info(f"[BPETrainer] Vocab size normal aralıkta: {min_vocab_size} <= {final_vocab_size} <= {max_vocab_size}")

    def get_merges(self) -> List[Tuple[str, str]]:
        return list(self._merges)

    def get_vocab(self) -> Dict[str, dict]:
        if not self.vocab:
            raise BPETrainingError("Vocab boş; önce train veya reset çağırın.")
        return copy.deepcopy(self.vocab)

    def update_vocab(self, new_tokens: List[str]) -> None:
        """
        Eğitimden önce dışarıdan gelen tokenları vocab’a eklemek için.
        """
        added = 0
        for tok in new_tokens:
            if tok not in self.vocab:
                nid = _next_id(self.vocab)
                self.vocab[tok] = {"id": nid, "total_freq": 0, "positions": []}
                added += 1
        if added:
            logger.info("[BPETrainer] vocab’a %d yeni token eklendi | new_size=%d", added, len(self.vocab))

    def reset(self) -> None:
        """Trainer’ı sıfırlar: vocab default’a, merges boş listeye döner."""
        self.vocab = _default_vocab_utils()
        self._ensure_special_tokens_exact()
        self._merges.clear()
        self._merge_ranks.clear()
        logger.info("[BPETrainer] sıfırlandı.")

    # -------------------- İç yardımcılar --------------------

    def _ensure_special_tokens_exact(self) -> None:
        """
        DEFAULT_SPECIALS’a göre özel tokenların doğru ID’lerle mevcut olduğunu garanti eder.
        Uyuşmazlıkta sessiz düzeltme yapmaz; açık hata fırlatır.
        """
        for sp_tok, sp_id in DEFAULT_SPECIALS.items():
            meta = self.vocab.get(sp_tok)
            if meta is None:
                # Eksikse ekleyelim; ID çakışması kontrolünü utils tarafında değil burada yapıyoruz
                # çünkü DEFAULT_SPECIALS sabittir ve üretim için referanstır.
                self.vocab[sp_tok] = {"id": sp_id, "total_freq": 0, "positions": []}
                logger.info("[BPETrainer] eksik özel token eklendi: %s -> id=%d", sp_tok, sp_id)
            else:
                mid = meta.get("id")
                if mid != sp_id:
                    raise BPETrainingError(
                        f"Özel token ID uyuşmazlığı: {sp_tok} id={mid}, beklenen={sp_id}. "
                        "Konfigürasyon/vocab sürtüşmesi var."
                    )

    @staticmethod
    def _build_merge_ranks(merges: List[Tuple[str, str]]) -> Dict[Tuple[str, str], int]:
        ranks: Dict[Tuple[str, str], int] = {}
        for i, pair in enumerate(merges):
            ranks[(pair[0], pair[1])] = i
        return ranks

    def _apply_merges_to_sequence(
        self,
        sequence: List[str],
        ranks: Dict[Tuple[str, str], int],
    ) -> List[str]:
        """
        Verilen sequence üzerinde, mevcut merges rank’larına göre ardışık birleştirmeleri uygular.
        Deterministik: her adımda en düşük rank’lı komşu çift birleştirilir.
        """
        if not ranks:
            return sequence[:]

        seq = list(sequence)

        def best_pair_index(seq_: List[str]) -> Optional[int]:
            best_i = None
            best_rank = None
            for i in range(len(seq_) - 1):
                pair = (seq_[i], seq_[i + 1])
                r = ranks.get(pair)
                if r is None:
                    continue
                if best_rank is None or r < best_rank:
                    best_rank = r
                    best_i = i
            return best_i

        while True:
            idx = best_pair_index(seq)
            if idx is None:
                break
            merged = seq[idx] + seq[idx + 1]
            seq[idx : idx + 2] = [merged]

        return seq

    def _get_pair_stats(
        self,
        sequence: List[str],
        min_frequency: int,
        *,
        protect_specials: bool,
    ) -> Dict[Tuple[str, str], int]:
        """
        OPTIMIZED: Komşu çift frekanslarını sayar. 
        Endüstri standardı: Sparse counting, memory efficient
        İsteğe bağlı olarak special tokenların dahil olduğu çiftleri korur (saymaz).
        """
        if len(sequence) < 2:
            return {}
            
        # Pre-compute specials set for O(1) lookup
        specials = frozenset(DEFAULT_SPECIALS.keys()) if protect_specials else frozenset()
        
        # Use Counter for better performance than defaultdict
        from collections import Counter
        counts = Counter()
        
        # Optimized iteration with tuple unpacking
        for i in range(len(sequence) - 1):
            a, b = sequence[i], sequence[i + 1]
            if a not in specials and b not in specials:
                counts[(a, b)] += 1

        # Early filtering for memory efficiency
        if min_frequency > 1:
            return {pair: freq for pair, freq in counts.items() if freq >= min_frequency}
        
        return dict(counts)

    @staticmethod
    def _select_best_pair(stats: Dict[Tuple[str, str], int]) -> Tuple[str, str]:
        """
        Deterministik seçim: (frekans DESC, pair lexicographic ASC).
        """
        # max ile custom key
        return max(stats.items(), key=lambda kv: (kv[1], tuple(kv[0])))[0]

    @staticmethod
    def _merge_pair_linear(sequence: List[str], pair: Tuple[str, str]) -> List[str]:
        """
        OPTIMIZED: Komşu eşleşmeleri sırayla tarayıp verilen çifti birleştirir.
        Endüstri standardı: In-place merge, memory efficient
        Örn. [a, b, a, b] + pair=(a,b) -> [ab, ab]
        """
        if not sequence or len(sequence) < 2:
            return sequence
            
        # Pre-allocate output with known size (memory efficient)
        out = []
        out_append = out.append  # Method reference for speed
        i = 0
        n = len(sequence)
        
        # Optimized loop with early termination
        while i < n - 1:
            if sequence[i] == pair[0] and sequence[i + 1] == pair[1]:
                out_append(sequence[i] + sequence[i + 1])
                i += 2
            else:
                out_append(sequence[i])
                i += 1
        
        # Handle last element if exists
        if i < n:
            out_append(sequence[i])
            
        return out

    def _apply_merges_batch_optimized(self, sequence: List[str], merges: List[Tuple[str, str]], batch_size: Optional[int] = None) -> List[str]:
        """Memory-efficient batch merge application"""
        if not merges or not sequence:
            return sequence
        
        # Config'ten batch size al
        if batch_size is None:
            batch_size = self.config.get("batch_merge_size", 1000)
        
        # Merges'leri batch'lere böl
        for i in range(0, len(merges), batch_size):
            batch_merges = merges[i:i + batch_size]
            
            # Her batch için merge uygula
            for pair in batch_merges:
                sequence = self._merge_pair_linear(sequence, pair)
            
            # Memory cleanup (config'ten interval)
            gc_interval = self.config.get("gc_interval", 1000)
            if i % gc_interval == 0:
                import gc
                gc.collect()
                # GPU memory cleanup
                if self.use_gpu:
                    torch.cuda.empty_cache()
        
        return sequence

    def _get_pair_stats_optimized(self, sequence: List[str], min_frequency: int, protect_specials: bool = True) -> Dict[Tuple[str, str], int]:
        """Memory-efficient pair statistics computation"""
        from collections import defaultdict
        
        # Memory-efficient approach
        pair_counts = defaultdict(int)
        
        # Process in chunks to reduce memory usage (config'ten)
        chunk_size = self.config.get("pair_stats_chunk_size", 10000)
        for i in range(0, len(sequence) - 1, chunk_size):
            chunk = sequence[i:i + chunk_size + 1]
            
            for j in range(len(chunk) - 1):
                left, right = chunk[j], chunk[j + 1]
                
                # Skip special tokens if protection enabled
                if protect_specials and (left.startswith('<') or right.startswith('<')):
                    continue
                
                pair_counts[(left, right)] += 1
        
        # Filter by minimum frequency
        return {pair: count for pair, count in pair_counts.items() if count >= min_frequency}

    def _get_pair_stats_parallel(self, sequence: List[str], min_frequency: int, protect_specials: bool = True, num_workers: Optional[int] = None) -> Dict[Tuple[str, str], int]:
        """Parallel pair statistics computation - 25K+ merges için"""
        from collections import defaultdict
        import multiprocessing as mp
        
        # num_workers config'ten al
        if num_workers is None:
            num_workers = self.config.get("parallel_workers", 4)
        
        # Sequence'u chunk'lara böl (config'ten min chunk size)
        min_chunk = self.config.get("min_chunk_size", 1000)
        chunk_size = max(min_chunk, len(sequence) // num_workers)
        chunks = [sequence[i:i + chunk_size] for i in range(0, len(sequence), chunk_size)]
        
        def process_chunk(chunk):
            """Chunk'ı işle"""
            pair_counts = defaultdict(int)
            for j in range(len(chunk) - 1):
                left, right = chunk[j], chunk[j + 1]
                
                # Skip special tokens if protection enabled
                if protect_specials and (left.startswith('<') or right.startswith('<')):
                    continue
                
                pair_counts[(left, right)] += 1
            return dict(pair_counts)
        
        # Parallel processing
        with mp.Pool(processes=num_workers) as pool:
            chunk_results = pool.map(process_chunk, chunks)
        
        # Sonuçları birleştir
        final_counts = defaultdict(int)
        for chunk_result in chunk_results:
            for pair, count in chunk_result.items():
                final_counts[pair] += count
        
        # Filter by minimum frequency
        return {pair: count for pair, count in final_counts.items() if count >= min_frequency}
    

    def _get_pair_stats_gpu(self, sequence: List[str], min_frequency: int, protect_specials: bool = True) -> Dict[Tuple[str, str], int]:
        """
        GPU-accelerated pair counting using PyTorch operations.
        """
        if not self.use_gpu:
            logger.warning("[BPETrainer] GPU not available, falling back to CPU")
            return self._get_pair_stats_optimized(sequence, min_frequency, protect_specials=protect_specials)

        logger.info(f"[BPETrainer] GPU pair computation: {len(sequence):,} tokens")
        
        # Tokenları ID'lere çevir
        token_to_id = {token: data["id"] for token, data in self.vocab.items()}
        id_to_token = {data["id"]: token for token, data in self.vocab.items()}
        sequence_ids = torch.tensor([token_to_id.get(token, 0) for token in sequence], device=self.device, dtype=torch.long)
        
        # Özel tokenları koruma
        specials = frozenset(DEFAULT_SPECIALS.keys()) if protect_specials else frozenset()
        
        with torch.amp.autocast('cuda'):
            # Shift tensor to create pairs
            shifted = sequence_ids[1:]  # Sonraki token'lar
            current = sequence_ids[:-1]  # Mevcut token'lar
            pair_tensor = torch.stack((current, shifted), dim=1)  # (n-1, 2) shape
            
            # Unique pairs and counts
            unique_pairs, counts = torch.unique(pair_tensor, dim=0, return_counts=True)
            
            # ID'leri token çiftlerine geri çevir ve filtrele
            pair_counts = {}
            for i in range(len(unique_pairs)):
                token1 = id_to_token.get(int(unique_pairs[i][0].item()), "<UNK>")
                token2 = id_to_token.get(int(unique_pairs[i][1].item()), "<UNK>")
                if protect_specials and (token1 in specials or token2 in specials):
                    continue
                count = counts[i].item()
                if count >= min_frequency:
                    pair_counts[(token1, token2)] = count
        
        logger.info(f"[BPETrainer] GPU found {len(pair_counts)} pairs")
        return pair_counts

    
    def _train_gpu_batch(
        self, 
        sequence: List[str], 
        target: Optional[int], 
        max_iter: int, 
        min_frequency: int,
        merges_done: int,
        protect_specials: bool,
        last_log_time: float,
        log_interval: float
    ) -> None:
        """
        GPU batch processing ile merge training (dinamik batch size ile optimize)
        """
        from concurrent.futures import ThreadPoolExecutor
        import time
        import psutil
        import torch

        logger.info("[BPETrainer] GPU/CPU batch processing ile merge training başlıyor... (Dinamik optimize)")
        
        # Pre-compute all possible pairs with GPU/CPU acceleration
        all_stats = self._get_pair_stats_gpu(sequence, min_frequency, protect_specials=protect_specials)
        
        if not all_stats:
            logger.info(f"[BPETrainer] Hiç merge bulunamadı (min_freq={min_frequency})")
            return
        
        # Sort pairs by frequency for batch processing
        sorted_pairs = sorted(all_stats.items(), key=lambda x: (-x[1], x[0]))
        
        # Initialize pair_index
        pair_index = 0
        
        # Dynamic batch size initialization
        if self.use_gpu:
            device_props = torch.cuda.get_device_properties(self.device)
            total_memory = device_props.total_memory / 1024**2  # MB
            allocated_memory = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
            available_memory = total_memory - allocated_memory
            
            # A100 (40/80 GB), T4 (16 GB), GTX 1050 Ti (4.29 GB) için tahmini ayar
            if total_memory >= 40000:  # A100
                base_batch_size = 5000
                memory_per_merge = 5  # MB/merge (A100 için optimize)
            elif total_memory >= 15000:  # T4
                base_batch_size = 2000
                memory_per_merge = 8
            else:  # Düşük GPU (ör. GTX 1050 Ti)
                base_batch_size = 500
                memory_per_merge = 10
                
            max_batch_size = max(base_batch_size, int(available_memory / memory_per_merge))
            batch_size = min(max_batch_size, 10000, max_iter - merges_done, len(sorted_pairs) - pair_index)
            logger.info(f"[BPETrainer] GPU: {device_props.name}, Total Memory: {total_memory:.1f} MB, "
                        f"Available: {available_memory:.1f} MB, Batch size: {batch_size}")
        else:
            # CPU için çekirdek sayısına ve belleğe göre tahmini
            import multiprocessing as mp
            num_cores = mp.cpu_count()
            memory = psutil.virtual_memory().available / 1024**2  # MB
            base_batch_size = max(50, int(num_cores * memory / 10000))  # Basit bir tahmin
            batch_size = min(base_batch_size, 500, max_iter - merges_done, len(sorted_pairs) - pair_index)
            logger.info(f"[BPETrainer] CPU: {num_cores} cores, Available Memory: {memory:.1f} MB, Batch size: {batch_size}")

        # Progress tracking
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Vocab size kontrolü için config'ten al
        max_vocab_size = self.config.get("max_vocab_size", 60000)
        # NOT: Final vocab size max_vocab_size'ı aşmamalı!
        # Tolerans kullanmıyoruz, direkt max_vocab_size kontrolü yapıyoruz
        
        while merges_done < max_iter and pair_index < len(sorted_pairs):
            if target is not None and merges_done >= target:
                logger.info(f"[BPETrainer] Hedef merge sayısına ulaşıldı: {merges_done}/{target}")
                break

            # Vocab size kontrolü (max_vocab_size limit'i, tolerans YOK)
            # NOT: Final vocab size max_vocab_size'ı aşmamalı!
            if len(self.vocab) >= max_vocab_size:
                logger.warning(f"[BPETrainer] Vocab size limit aşıldı: {len(self.vocab)} >= {max_vocab_size}")
                logger.info(f"[BPETrainer] Merge training durduruluyor. Toplam merge: {merges_done}")
                break

            # Progress logging
            current_time = time.time()
            if current_time - last_log_time >= log_interval:
                elapsed = current_time - start_time
                current_memory = process.memory_info().rss / 1024 / 1024
                progress_pct = (merges_done / target * 100) if target else (merges_done / max_iter * 100)
                if self.use_gpu:
                    gpu_memory = torch.cuda.memory_allocated() / 1024**2
                    logger.info(f"[BPETrainer] GPU Progress: {merges_done:,} merges | "
                                f"Vocab: {len(self.vocab):,} | "
                                f"Sequence: {len(sequence):,} | "
                                f"GPU Memory: {gpu_memory:.1f} MB | "
                                f"Bellek: {current_memory:.1f}MB (+{current_memory-initial_memory:.1f}MB) | "
                                f"Süre: {elapsed:.1f}s | "
                                f"İlerleme: {progress_pct:.1f}%")
                else:
                    logger.info(f"[BPETrainer] CPU Progress: {merges_done:,} merges | "
                                f"Vocab: {len(self.vocab):,} | "
                                f"Sequence: {len(sequence):,} | "
                                f"Bellek: {current_memory:.1f}MB (+{current_memory-initial_memory:.1f}MB) | "
                                f"Süre: {elapsed:.1f}s | "
                                f"İlerleme: {progress_pct:.1f}%")
                last_log_time = current_time

            # Batch processing
            batch_merges = []
            for _ in range(batch_size):
                if pair_index >= len(sorted_pairs):
                    break
                
                # Vocab size kontrolü - token eklemeden ÖNCE kontrol et
                # NOT: Final vocab size max_vocab_size'ı aşmamalı!
                if len(self.vocab) >= max_vocab_size:
                    logger.debug(f"[BPETrainer] Vocab size limit aşıldı, batch processing durduruluyor: {len(self.vocab)} >= {max_vocab_size}")
                    break
                    
                best_pair, freq = sorted_pairs[pair_index]
                new_token = best_pair[0] + best_pair[1]
                if self._is_valid_merge(sequence, best_pair, protect_specials):
                    # Yeni token'ı vocab'a ekle (vocab size limit kontrolü ile)
                    if new_token not in self.vocab:
                        # Token eklemeden ÖNCE tekrar kontrol et (güvenlik için)
                        if len(self.vocab) >= max_vocab_size:
                            logger.debug(f"[BPETrainer] Vocab size limit aşıldı, token eklenemiyor: {new_token}")
                            break  # Batch processing'i durdur
                        new_id = _next_id(self.vocab)
                        self.vocab[new_token] = {"id": new_id, "total_freq": freq, "positions": []}
                    batch_merges.append((best_pair, new_token, freq))
                pair_index += 1
            
            # Apply batch merges (vocab size kontrolü ile)
            if batch_merges:
                # Batch uygulanmadan önce vocab size kontrolü (max_vocab_size limit'i)
                if len(self.vocab) >= max_vocab_size:
                    logger.warning(f"[BPETrainer] Vocab size limit aşıldı: {len(self.vocab)} >= {max_vocab_size}")
                    logger.info(f"[BPETrainer] Merge training durduruluyor. Toplam merge: {merges_done}")
                    break
                
                # Batch içindeki merge'leri filtrele (vocab size limit'i aşmayacak şekilde)
                filtered_batch_merges = []
                for pair, new_token, freq in batch_merges:
                    # Token zaten vocab'te olabilir (önceden eklenmiş)
                    # Ama yine de kontrol et (güvenlik için)
                    if len(self.vocab) >= max_vocab_size:
                        logger.debug(f"[BPETrainer] Vocab size limit aşıldı, merge atlanıyor: {new_token}")
                        break  # Bu batch'teki kalan merge'leri atla
                    filtered_batch_merges.append((pair, new_token, freq))
                
                if filtered_batch_merges:
                    sequence = self._apply_batch_merges_gpu(sequence, filtered_batch_merges)
                    merges_done += len(filtered_batch_merges)
                    # Merge'leri _merges listesine ekle
                    for pair, _, _ in filtered_batch_merges:
                        self._merges.append(pair)
                        self._merge_ranks[pair] = len(self._merges) - 1
                    
                    # GPU memory cleanup (her batch'ten sonra)
                    if self.use_gpu:
                        torch.cuda.empty_cache()
                else:
                    # Tüm merge'ler filtrelendi, training durdurulmalı
                    logger.warning(f"[BPETrainer] Tüm merge'ler vocab size limit nedeniyle filtrelendi. Training durduruluyor.")
                    break
            else:
                break
    
    def _train_cpu_sequential(
        self, 
        sequence: List[str], 
        target: Optional[int], 
        max_iter: int, 
        min_frequency: int,
        merges_done: int,
        protect_specials: bool,
        last_log_time: float,
        log_interval: float
    ) -> None:
        """
        CPU sequential processing ile merge training (eski yöntem)
        """
        import time
        import psutil
        
        logger.info("[BPETrainer] CPU sequential processing ile merge training başlıyor...")
        
        # Pre-compute all possible pairs
        all_stats = self._get_pair_stats_optimized(sequence, min_frequency, protect_specials=protect_specials)
        
        if not all_stats:
            logger.info(f"[BPETrainer] Hiç merge bulunamadı (min_freq={min_frequency})")
            return
        
        # Sort pairs by frequency
        sorted_pairs = sorted(all_stats.items(), key=lambda x: (-x[1], x[0]))
        
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Vocab size kontrolü için config'ten al
        max_vocab_size = self.config.get("max_vocab_size", 60000)
        # NOT: Final vocab size max_vocab_size'ı aşmamalı!
        # Tolerans kullanmıyoruz, direkt max_vocab_size kontrolü yapıyoruz
        
        for pair_index, (best_pair, freq) in enumerate(sorted_pairs):
            if merges_done >= max_iter:
                break
            if target is not None and merges_done >= target:
                logger.info(f"[BPETrainer] Hedef merge sayısına ulaşıldı: {merges_done}/{target}")
                break
            
            # Vocab size kontrolü (max_vocab_size limit'i, tolerans YOK)
            # NOT: Final vocab size max_vocab_size'ı aşmamalı!
            if len(self.vocab) >= max_vocab_size:
                logger.warning(f"[BPETrainer] Vocab size limit aşıldı: {len(self.vocab)} >= {max_vocab_size}")
                logger.info(f"[BPETrainer] Merge training durduruluyor. Toplam merge: {merges_done}")
                break

            # Progress logging
            current_time = time.time()
            if current_time - last_log_time >= log_interval:
                elapsed = current_time - start_time
                current_memory = process.memory_info().rss / 1024 / 1024
                progress_pct = (merges_done / target * 100) if target else (merges_done / max_iter * 100)
                
                logger.info(f"[BPETrainer] CPU Progress: {merges_done:,} merges | "
                           f"Vocab: {len(self.vocab):,} | "
                           f"Sequence: {len(sequence):,} | "
                           f"Bellek: {current_memory:.1f}MB (+{current_memory-initial_memory:.1f}MB) | "
                           f"Süre: {elapsed:.1f}s | "
                           f"İlerleme: {progress_pct:.1f}%")
                last_log_time = current_time

            # Apply single merge
            if self._is_valid_merge(sequence, best_pair, protect_specials):
                new_sequence = self._apply_single_merge(sequence, best_pair)
                if new_sequence is None:
                    # Vocab limit aşıldı, training durdurulmalı
                    logger.warning(f"[BPETrainer] Vocab size limit aşıldı, merge yapılamadı. Training durduruluyor.")
                    break
                sequence = new_sequence
                merges_done += 1
    
    def _is_valid_merge(self, sequence: List[str], pair: Tuple[str, str], protect_specials: bool) -> bool:
        """
        Merge'in hala geçerli olup olmadığını kontrol et
        """
        if protect_specials:
            if pair[0] in DEFAULT_SPECIALS or pair[1] in DEFAULT_SPECIALS:
                return False
        
        # Sequence'de bu pair'in var olup olmadığını kontrol et
        for i in range(len(sequence) - 1):
            if sequence[i] == pair[0] and sequence[i + 1] == pair[1]:
                return True
        return False
    
    def _apply_single_merge(self, sequence: List[str], pair: Tuple[str, str]) -> Optional[List[str]]:
        """
        Tek merge'i uygula. Vocab limit aşıldıysa None döner (merge yapılmaz).
        """
        new_token = pair[0] + pair[1]
        
        # Vocab size kontrolü
        max_vocab_size = self.config.get("max_vocab_size", 60000)
        
        # Yeni token'ı vocab'a ekle (vocab size limit kontrolü ile)
        # NOT: Final vocab size max_vocab_size'ı aşmamalı!
        if new_token not in self.vocab:
            # Token eklemeden ÖNCE kontrol et
            if len(self.vocab) >= max_vocab_size:
                # Vocab size limit aşıldı, merge yapılmamalı
                # Çünkü merge yapılırsa vocab'te olmayan token oluşur (tutarsızlık)
                logger.debug(f"[BPETrainer] Vocab size limit aşıldı, merge yapılmıyor: {new_token}")
                return None  # Merge yapılmadı, training durdurulmalı
            new_id = _next_id(self.vocab)
            self.vocab[new_token] = {"id": new_id, "total_freq": 0, "positions": []}
        
        # Merge'i merges listesine ekle
        self._merges.append(pair)
        self._merge_ranks[pair] = len(self._merges) - 1
        
        # Sequence'de pair'leri yeni token ile değiştir
        new_sequence = []
        i = 0
        while i < len(sequence):
            if i < len(sequence) - 1 and sequence[i] == pair[0] and sequence[i + 1] == pair[1]:
                new_sequence.append(new_token)
                i += 2
            else:
                new_sequence.append(sequence[i])
                i += 1
        
        return new_sequence
    
    def _apply_batch_merges_gpu(self, sequence: List[str], batch_merges: List[Tuple[Tuple[str, str], str, int]]) -> List[str]:
        if not self.use_gpu or not batch_merges:
            for pair, new_token, _ in batch_merges:
                sequence = self._apply_single_merge(sequence, pair)
            return sequence

        logger.info(f"[BPETrainer] Applying {len(batch_merges)} merges with GPU")
        
        # Token'ları ID'lere çevir
        token_to_id = {token: data["id"] for token, data in self.vocab.items()}
        id_to_token = {data["id"]: token for token, data in self.vocab.items()}
        sequence_ids = torch.tensor([token_to_id.get(token, 0) for token in sequence], device=self.device, dtype=torch.long)
        
        # Merge pair'lerini ve new token'ları ID'lere çevir
        merge_pairs = torch.tensor([(token_to_id[pair[0]], token_to_id[pair[1]]) for pair, _, _ in batch_merges], device=self.device, dtype=torch.long)
        new_tokens = torch.tensor([token_to_id[new_token] for _, new_token, _ in batch_merges], device=self.device, dtype=torch.long)

        # GPU'da merge uygula (AMP ile optimize)
        with torch.amp.autocast('cuda'):
            new_sequence_ids = sequence_ids.clone()
            for i in range(len(merge_pairs)):
                id1, id2 = merge_pairs[i]
                new_id = new_tokens[i]
                mask = (new_sequence_ids[:-1] == id1) & (new_sequence_ids[1:] == id2)
                indices = torch.where(mask)[0]
                if indices.numel() > 0:
                    new_sequence_ids[indices] = new_id
                    # -1 ile işaretleme ve filtreleme tensor operasyonlarıyla
                    shift_mask = torch.zeros_like(new_sequence_ids, dtype=torch.bool, device=self.device)
                    shift_mask[indices + 1] = True
                    new_sequence_ids = new_sequence_ids[~shift_mask]

        # ID'leri token'lara geri çevir
        return [id_to_token.get(int(id.item()), "<UNK>") for id in new_sequence_ids.cpu()]
