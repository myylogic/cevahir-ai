# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: dataloader_v3.py
Modül: training_system/v3/data
Görev: DataLoader V3 Factory - Gelişmiş GPU optimizasyonu ile DataLoader oluşturma.
       BucketBatchSampler + DynamicPaddingCollator + async GPU transfer entegrasyonu.

GPU OPTİMİZASYONLARI:
       1. pin_memory=True  → Sabitlenmiş RAM → PCIe bus üzerinden hızlı DMA transfer
       2. non_blocking=True → Async GPU transfer (hesaplama ile örtüşür)
       3. prefetch_factor=2 → DataLoader N batch önceden hazırlar
       4. persistent_workers=True → Worker process'leri epoch'lar arası canlı kalır
       5. BucketBatchSampler → Padding waste minimize
       6. DynamicPaddingCollator → Batch başına dinamik pad uzunluğu

REFERANSLAR:
       - PyTorch DataLoader best practices (pytorch.org/docs)
       - Ott et al. 2019, fairseq dynamic batching
       - Schwartz et al. 2020, "Right Tool for the Job"

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.

================================================================================
"""

import os
import logging
from typing import List, Tuple, Optional, Dict, Any

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from .dataset_v3 import CevahirDataset
from .sampler_v3 import BucketBatchSampler
from .collator_v3 import DynamicPaddingCollator, create_static_collate

logger = logging.getLogger(__name__)


def create_dataloader_v3(
    data: List[Tuple],
    batch_size: int,
    pad_id: int = 0,
    device: str = "cuda",
    shuffle: bool = True,
    # Bucket batching
    use_bucket_batching: bool = True,
    num_buckets: int = 32,
    # Dynamic padding
    use_dynamic_padding: bool = True,
    max_seq_length: Optional[int] = None,
    # Workers
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    # Diğer
    drop_last: bool = False,
    seed: int = 42,
    epoch: int = 0,
) -> DataLoader:
    """
    Gelişmiş GPU optimizasyonu ile DataLoader oluştur.

    Args:
        data: Eğitim verisi listesi (inp, tgt) veya (inp, tgt, source_id)
        batch_size: Batch boyutu
        pad_id: PAD token ID
        device: Hedef cihaz ("cuda" veya "cpu")
        shuffle: Veriyi karıştır (train için True, val için False)
        use_bucket_batching: BucketBatchSampler kullan (padding waste azaltır)
        num_buckets: Bucket sayısı (daha fazla = daha az padding, daha az randomness)
        use_dynamic_padding: DynamicPaddingCollator kullan (batch içi dinamik pad)
        max_seq_length: Absolut üst limit (güvenlik için)
        num_workers: DataLoader worker sayısı (0 = ana process)
        pin_memory: Sabitlenmiş RAM (None ise cuda için otomatik True)
        prefetch_factor: Worker başına prefetch batch sayısı
        persistent_workers: Worker'ları epoch'lar arası canlı tut
        drop_last: Son incomplete batch'i at
        seed: Random seed
        epoch: Epoch numarası (BucketBatchSampler için)

    Returns:
        DataLoader instance
    """
    # pin_memory: CUDA için otomatik True
    if pin_memory is None:
        pin_memory = (device == "cuda" and torch.cuda.is_available())

    # Worker sayısı: Windows'ta num_workers>0 sorunlu olabilir
    if num_workers > 0 and os.name == "nt":  # Windows
        logger.warning(
            f"[DataLoaderV3] Windows'ta num_workers={num_workers} — "
            f"sorun yaşarsanız num_workers=0 deneyin"
        )

    # Dataset
    dataset = CevahirDataset(
        data=data,
        pad_id=pad_id,
        precompute_lengths=use_bucket_batching,
    )

    # Collate fn
    if use_dynamic_padding:
        collate_fn = DynamicPaddingCollator(
            pad_id=pad_id,
            max_seq_length=max_seq_length,
        )
        logger.info(f"[DataLoaderV3] Dynamic padding collator aktif (pad_id={pad_id})")
    else:
        collate_fn = create_static_collate(pad_id=pad_id)
        logger.info(f"[DataLoaderV3] Statik collate kullanılıyor")

    # Sampler / BatchSampler
    if use_bucket_batching and shuffle:
        # BucketBatchSampler: batch_sampler olarak geçirilir
        batch_sampler = BucketBatchSampler(
            lengths=dataset.lengths,
            batch_size=batch_size,
            num_buckets=num_buckets,
            shuffle_buckets=True,
            shuffle_within_bucket=True,
            drop_last=drop_last,
            seed=seed,
        )
        batch_sampler.set_epoch(epoch)

        loader = DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers and num_workers > 0),
            prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        )
        logger.info(
            f"[DataLoaderV3] BucketBatchSampler aktif: "
            f"{len(batch_sampler)} batch, {num_buckets} bucket, "
            f"batch_size={batch_size}"
        )

    else:
        # Standart shuffle veya sequential sampler
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers and num_workers > 0),
            prefetch_factor=(prefetch_factor if num_workers > 0 else None),
            drop_last=drop_last,
        )
        logger.info(
            f"[DataLoaderV3] Standart DataLoader: "
            f"batch_size={batch_size}, shuffle={shuffle}"
        )

    # İstatistik logu
    _log_dataloader_stats(dataset, loader, device, pin_memory, num_workers)

    return loader


def create_dataloaders_v3(
    train_data: List[Tuple],
    val_data: List[Tuple],
    batch_size: int,
    pad_id: int = 0,
    device: str = "cuda",
    # Bucket batching
    use_bucket_batching: bool = True,
    num_buckets: int = 32,
    # Dynamic padding
    use_dynamic_padding: bool = True,
    max_seq_length: Optional[int] = None,
    # Workers
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    # Diğer
    drop_last_train: bool = False,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Train ve validation DataLoader'larını oluştur.

    Train: BucketBatchSampler + shuffle
    Val: Sequential (shuffle yok, bucket opsiyonel)

    Returns:
        (train_loader, val_loader)
    """
    logger.info(f"[DataLoaderV3] Train DataLoader oluşturuluyor ({len(train_data):,} örnek)...")
    train_loader = create_dataloader_v3(
        data=train_data,
        batch_size=batch_size,
        pad_id=pad_id,
        device=device,
        shuffle=True,
        use_bucket_batching=use_bucket_batching,
        num_buckets=num_buckets,
        use_dynamic_padding=use_dynamic_padding,
        max_seq_length=max_seq_length,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        drop_last=drop_last_train,
        seed=seed,
    )

    logger.info(f"[DataLoaderV3] Val DataLoader oluşturuluyor ({len(val_data):,} örnek)...")
    val_loader = create_dataloader_v3(
        data=val_data,
        batch_size=batch_size,
        pad_id=pad_id,
        device=device,
        shuffle=False,
        use_bucket_batching=False,   # Val: sıralı, bucket yok
        use_dynamic_padding=use_dynamic_padding,
        max_seq_length=max_seq_length,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        drop_last=False,
        seed=seed,
    )

    logger.info(
        f"[DataLoaderV3] DataLoaders hazır: "
        f"train={len(train_loader)} batch, val={len(val_loader)} batch"
    )

    return train_loader, val_loader


def _log_dataloader_stats(
    dataset: CevahirDataset,
    loader: DataLoader,
    device: str,
    pin_memory: bool,
    num_workers: int,
) -> None:
    """DataLoader konfigürasyon istatistiklerini logla."""
    stats = dataset.get_length_stats()
    if stats:
        logger.info(
            f"[DataLoaderV3] Sequence uzunluk istatistikleri:\n"
            f"  min={stats['min']}, max={stats['max']}, "
            f"mean={stats['mean']:.1f}, median={stats['median']}\n"
            f"  p25={stats['p25']}, p75={stats['p75']}, "
            f"p90={stats['p90']}, p99={stats['p99']}"
        )

    logger.info(
        f"[DataLoaderV3] Konfigürasyon:\n"
        f"  device={device}, pin_memory={pin_memory}, "
        f"num_workers={num_workers}\n"
        f"  toplam_batch={len(loader)}"
    )

    if pin_memory and device == "cuda":
        logger.info(
            f"[DataLoaderV3] GPU optimizasyonu: pin_memory=True → "
            f"async DMA transfer aktif"
        )
