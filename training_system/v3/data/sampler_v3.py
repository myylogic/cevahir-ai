# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: sampler_v3.py
Modül: training_system/v3/data
Görev: BucketBatchSampler V3 - Sequence uzunluğuna göre gruplama.
       Benzer uzunluktaki sequence'ları aynı batch'e koyar → padding waste azalır.
       Endüstri standardı: fairseq, HuggingFace Trainer, OpenNMT.

AMAÇ:
       Static padding (tüm sequence'lar max_seq_length'e pad) → GPU'nun %70-90'ı
       PAD token'ı hesaplar. Bucket batching ile padding waste %20-40'a düşer.

ALGORİTMA:
       1. Sequence'ları uzunluğa göre sırala
       2. num_buckets adet bucket'a böl
       3. Her epoch başında bucket'lar arası shuffle (farklı batch kombinasyonları)
       4. Her bucket içinde batch'ler oluştur

Referans: Schwartz et al. 2020 "Right Tool for the Job" (bucket batching efficiency)

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.

================================================================================
"""

import random
import logging
import math
from typing import List, Iterator, Optional

from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class BucketBatchSampler(Sampler[List[int]]):
    """
    BucketBatchSampler V3.

    Sequence'ları uzunluğa göre gruplayarak padding waste'i minimize eder.

    Args:
        lengths: Her örnek için gerçek sequence uzunluğu (pad hariç)
        batch_size: Batch başına örnek sayısı
        num_buckets: Bucket sayısı (daha fazla = daha iyi gruplama, daha az çeşitlilik)
        shuffle_buckets: Epoch başında bucket'ları karıştır (randomness için)
        shuffle_within_bucket: Bucket içinde karıştır
        drop_last: Son incomplete batch'i at
        seed: Random seed (tekrarlanabilirlik)

    Kullanım:
        sampler = BucketBatchSampler(dataset.lengths, batch_size=16, num_buckets=32)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=dynamic_pad_collate)
    """

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        num_buckets: int = 32,
        shuffle_buckets: bool = True,
        shuffle_within_bucket: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.lengths = lengths
        self.batch_size = batch_size
        self.num_buckets = num_buckets
        self.shuffle_buckets = shuffle_buckets
        self.shuffle_within_bucket = shuffle_within_bucket
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0

        # Bucket'ları oluştur
        self._buckets = self._build_buckets()

        # İstatistik
        total_batches = sum(
            len(bucket) // batch_size + (0 if len(bucket) % batch_size == 0 or drop_last else 1)
            for bucket in self._buckets
        )
        avg_bucket_size = len(lengths) / max(num_buckets, 1)
        logger.info(
            f"[BucketSampler] {len(lengths):,} örnek → {num_buckets} bucket "
            f"(avg {avg_bucket_size:.0f} örnek/bucket) → ~{total_batches} batch"
        )

    def _build_buckets(self) -> List[List[int]]:
        """Uzunluğa göre sıralanmış indeksleri bucket'lara böl."""
        # Uzunluğa göre sırala
        sorted_indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])

        # num_buckets'a böl
        bucket_size = max(1, math.ceil(len(sorted_indices) / self.num_buckets))
        buckets = []
        for i in range(0, len(sorted_indices), bucket_size):
            bucket = sorted_indices[i : i + bucket_size]
            if bucket:
                buckets.append(bucket)

        return buckets

    def set_epoch(self, epoch: int) -> None:
        """Epoch bazlı seed için epoch sayısını güncelle."""
        self._epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self._epoch)

        # Bucket kopyaları oluştur (in-place shuffle yapmamak için)
        buckets = [list(b) for b in self._buckets]

        # Bucket içi shuffle
        if self.shuffle_within_bucket:
            for bucket in buckets:
                rng.shuffle(bucket)

        # Her bucket'tan batch'ler topla
        all_batches: List[List[int]] = []
        for bucket in buckets:
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        # Batch'leri karıştır (bucket sıralaması belli olmasın)
        if self.shuffle_buckets:
            rng.shuffle(all_batches)

        yield from all_batches

    def __len__(self) -> int:
        total = 0
        for bucket in self._buckets:
            n = len(bucket)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += math.ceil(n / self.batch_size)
        return total
