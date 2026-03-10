# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: data_loader_wrapper.py
Modül: training_system/v2/utils
Görev: DataLoader Wrapper - Minimal DataLoader Wrapper. Cache'den gelen veriyi
       minimal DataLoader'lara çevirme. V2 TrainingManager için gerekli
       (DataLoader interface bekliyor). SimpleDataset ve DataLoader oluşturma.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (DataLoader wrapper)
- Design Patterns: Wrapper Pattern (DataLoader interface sağlama)
- Endüstri Standartları: PyTorch DataLoader interface

KULLANIM:
- Cache'den gelen veriyi DataLoader'a çevirmek için
- V2 TrainingManager için DataLoader oluşturmak için
- Dataset wrapper oluşturmak için

BAĞIMLILIKLAR:
- torch.utils.data: DataLoader ve Dataset sınıfları

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Optional, Any


class SimpleDataset(Dataset):
    """Basit dataset wrapper"""
    
    def __init__(self, data: List[Tuple[torch.Tensor, torch.Tensor]]):
        """
        Args:
            data: List of (input_tensor, target_tensor) pairs
        """
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]


def custom_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function - CPU'da stack yap.
    GPU transfer pin_memory ile otomatik yapılacak.
    
    Args:
        batch: List of (input_tensor, target_tensor) pairs
        
    Returns:
        Tuple of (inputs_batch, targets_batch)
    """
    inputs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return inputs, targets


def create_dataloaders(
    train_data: List[Tuple[torch.Tensor, torch.Tensor]],
    val_data: List[Tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    device: str = "cuda",
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Cache'den gelen veriyi minimal DataLoader'lara çevir.
    
    Args:
        train_data: Train dataset (list of tuples)
        val_data: Validation dataset (list of tuples)
        batch_size: Batch size
        device: Device ("cuda" or "cpu")
        num_workers: Number of workers (default: 0 - single process)
        pin_memory: Pin memory for faster GPU transfer (default: True if CUDA)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Pin memory sadece CUDA için
    pin_memory = pin_memory and device == "cuda"
    
    # Persistent workers
    persistent_workers = num_workers > 0
    
    # Train DataLoader
    train_dataset = SimpleDataset(train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    # Validation DataLoader
    val_dataset = SimpleDataset(val_data)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    return train_loader, val_loader

