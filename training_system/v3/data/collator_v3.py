# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: collator_v3.py
Modül: training_system/v3/data
Görev: DynamicPaddingCollator V3 - Batch içi maksimum uzunluğa dinamik padding.
       Her batch sadece kendi içindeki en uzun sequence'a kadar pad edilir.
       Global max_seq_length'e pad edilmez → GPU memory kullanımı optimize edilir.

AKADEMİK REFERANS:
       Ott et al. 2019, "Scaling Neural Machine Translation" (fairseq)
       — Dynamic padding ile training throughput %2-3x artış

V2 FARK:
       V2: custom_collate → torch.stack() (statik padding, global max_seq_length)
       V3: DynamicPaddingCollator → batch'teki max uzunluğa dinamik padding

GPU ETKİSİ:
       Kısa sequence'lar için bellek / compute tasarrufu:
       - Batch {len=10, len=12, len=15} → max_pad=15 (global 512 yerine!)
       - GPU 97% daha az PAD token hesaplar bu batch için

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.

================================================================================
"""

import torch
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DynamicPaddingCollator:
    """
    Dynamic Padding Collator V3.

    Her batch'teki maksimum sequence uzunluğuna pad eder.
    BucketBatchSampler ile kullanıldığında padding waste minimumdur.

    Args:
        pad_id: PAD token ID
        max_seq_length: Absolut üst limit (güvenlik için)
        non_blocking: GPU transfer non-blocking modu (async prefetch için)
    """

    def __init__(
        self,
        pad_id: int = 0,
        max_seq_length: Optional[int] = None,
        non_blocking: bool = True,
    ):
        self.pad_id = pad_id
        self.max_seq_length = max_seq_length
        self.non_blocking = non_blocking

    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch içi dinamik padding.

        Args:
            batch: List of (inp_tensor, tgt_tensor) pairs (farklı uzunlukta olabilir)

        Returns:
            (inputs, targets) — (batch_size, max_len_in_batch) shape
        """
        if not batch:
            raise ValueError("[CollatorV3] Boş batch!")

        # Batch içindeki maksimum uzunluk
        max_len = max(item[0].size(-1) for item in batch)

        # Absolut üst limit
        if self.max_seq_length is not None:
            max_len = min(max_len, self.max_seq_length)

        batch_size = len(batch)
        inputs = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        targets = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)

        for i, (inp, tgt) in enumerate(batch):
            inp_len = min(inp.size(-1), max_len)
            tgt_len = min(tgt.size(-1), max_len)
            inputs[i, :inp_len] = inp[:inp_len]
            targets[i, :tgt_len] = tgt[:tgt_len]

        return inputs, targets


def create_static_collate(pad_id: int = 0):
    """
    Statik collate fonksiyonu (V2 geriye uyumluluk).
    Tüm tensörler aynı boyuttaysa (cache'den) kullanılır.
    """
    def collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        return inputs, targets
    return collate
