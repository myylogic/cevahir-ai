# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: base_tokenizer_manager.py
Modül: tokenizer_management
Görev: BaseTokenizerManager - Tokenizer manager'lar için abstract base class.
       Ortak tokenizer interface'i tanımlar (encode, decode, train, get_vocab).

MİMARİ:
- SOLID Prensipleri: Dependency Inversion (abstract interface),
                     Liskov Substitution (alt sınıflar uyumlu)
- Design Patterns: Template Method Pattern (abstract method'lar),
                  Strategy Pattern (farklı tokenizer implementasyonları)
- Endüstri Standartları: Interface segregation, Abstract base class pattern

KULLANIM:
- BPEManager ve diğer tokenizer manager'lar bu sınıftan türetilir
- Ortak API garantisi sağlar (encode, decode, train, get_vocab)
- TokenizerCore tarafından kullanılır

BAĞIMLILIKLAR:
- abc: Abstract base class desteği için

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

import abc
from typing import Any, Dict, List, Tuple

class BaseTokenizerManager(abc.ABC):
    """
    Ortak bir tokenizer‐manager arayüzü.
    Alt sınıflar aşağıdaki public API’ları sağlamalı:
      - encode
      - decode
      - train
      - get_vocab
      - get_merges
      - set_vocab
    update_reverse_vocab ise artık bir hook; default no-op.
    """

    @abc.abstractmethod
    def encode(self, text: str, mode: str) -> Any:
        """
        Metni tokenize edip, (tokens, ids) veya ids döndürür.
        :param text: Girdi metni
        :param mode: 'train' veya 'inference'
        """
        pass

    @abc.abstractmethod
    def decode(self, token_ids: List[int], method: str = "bpe") -> str:
        """
        Token ID listesini string’e dönüştürür.
        :param token_ids: ID listesi
        :param method: 'raw' veya 'bpe'
        """
        pass

    @abc.abstractmethod
    def train(self, corpus: List[Any], *args, **kwargs) -> None:
        """
        Eğitim için corpus alır, vocab ve merges öğrenimini yapar.
        """
        pass

    @abc.abstractmethod
    def get_vocab(self) -> Dict[str, Any]:
        """
        Şu anki vocab sözlüğünü döner.
        """
        pass

    @abc.abstractmethod
    def get_merges(self) -> List[Tuple[str, str]]:
        """
        Şu anki merges çiftlerini döner.
        """
        pass

    @abc.abstractmethod
    def set_vocab(self, new_vocab: Dict[str, Any]) -> None:
        """
        Harici bir vocab sözlüğünü yükler ve manager içindeki
        encoder/decoder/trainer ile senkronize eder.
        """
        pass

    def update_reverse_vocab(self) -> None:
        """
        (Optional) Reverse‐vocab mapping’in güncellenmesi için hook.
        Alt sınıf isterse override edebilir; aksi takdirde no‐op.
        """
        return
