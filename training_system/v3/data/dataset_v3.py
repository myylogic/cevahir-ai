# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: dataset_v3.py
Modül: training_system/v3/data
Görev: Dataset V3 - Sequence uzunluklarına hızlı erişim sağlayan dataset.
       BucketBatchSampler ile entegre çalışır.
       V2'deki SimpleDataset'in gelişmiş versiyonu.

KRİTİK GELİŞTİRMELER (V2 → V3):
- Sequence uzunluğu indeksi (bucket batching için)
- PAD maskesi hesaplama (gereksiz pad token'larını dışlar)
- Lazy tensor dönüşümü (büyük datasets için bellek dostu)
- İstatistik raporlama (sequence uzunluk dağılımı)

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.

================================================================================
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CevahirDataset(Dataset):
    """
    Cevahir eğitim dataset'i V3.

    Özellikler:
    - Sequence uzunluk indeksi (BucketBatchSampler için)
    - PAD token maskeleme
    - Lazy veya pre-computed tensor modu
    - İstatistik raporlama
    """

    def __init__(
        self,
        data: List[Tuple],
        pad_id: int = 0,
        precompute_lengths: bool = True,
    ):
        """
        Args:
            data: List of (inp_tensor, tgt_tensor) or (inp_list, tgt_list) pairs
            pad_id: PAD token ID (uzunluk hesaplama için)
            precompute_lengths: True ise başlangıçta tüm uzunlukları hesapla
        """
        self.data = data
        self.pad_id = pad_id
        self._lengths: Optional[List[int]] = None

        if precompute_lengths:
            self._lengths = self._compute_lengths()
            logger.info(
                f"[Dataset V3] {len(data):,} örnek yüklendi. "
                f"Uzunluk aralığı: [{min(self._lengths)}–{max(self._lengths)}] "
                f"(ortalama={sum(self._lengths)/len(self._lengths):.1f})"
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        # source_id varsa kaldır
        if len(item) == 3:
            inp, tgt, _ = item
        else:
            inp, tgt = item

        # Zaten tensor ise doğrudan kullan
        if isinstance(inp, torch.Tensor):
            inp_t = inp
            tgt_t = tgt
        else:
            inp_t = torch.tensor(inp, dtype=torch.long)
            tgt_t = torch.tensor(tgt, dtype=torch.long)

        return inp_t, tgt_t

    def _compute_lengths(self) -> List[int]:
        """
        Her sequence'ın gerçek uzunluğunu hesapla (PAD token'lar hariç).
        BucketBatchSampler bu bilgiyi kullanır.
        """
        lengths = []
        for item in self.data:
            if len(item) >= 2:
                inp = item[0]
                if isinstance(inp, torch.Tensor):
                    inp_list = inp.tolist()
                else:
                    inp_list = list(inp)
                # Son PAD olmayan pozisyonu bul
                real_len = len(inp_list)
                for i in range(len(inp_list) - 1, -1, -1):
                    if inp_list[i] != self.pad_id:
                        real_len = i + 1
                        break
                lengths.append(real_len)
            else:
                lengths.append(0)
        return lengths

    @property
    def lengths(self) -> List[int]:
        """Sequence uzunlukları (BucketBatchSampler için)."""
        if self._lengths is None:
            self._lengths = self._compute_lengths()
        return self._lengths

    def get_length_stats(self) -> Dict[str, Any]:
        """Uzunluk dağılımı istatistikleri."""
        lengths = self.lengths
        if not lengths:
            return {}

        sorted_lens = sorted(lengths)
        n = len(sorted_lens)
        return {
            "count": n,
            "min": sorted_lens[0],
            "max": sorted_lens[-1],
            "mean": sum(sorted_lens) / n,
            "median": sorted_lens[n // 2],
            "p25": sorted_lens[n // 4],
            "p75": sorted_lens[3 * n // 4],
            "p90": sorted_lens[int(0.9 * n)],
            "p99": sorted_lens[int(0.99 * n)],
        }
