"""
Cevahir V3 - Batch Veri İşlemcisi
===================================
Bu modül, eğitim sırasında farklı biçimlerde gelen batch verilerini
standart bir (inputs, targets) çiftine dönüştürür.

Desteklenen Batch Biçimleri:
    1. Tensor        : [batch, seq_len+1] → dil modeli kayması ile bölünür
    2. Tuple/List    : (inputs, targets) veya [inputs, targets]
    3. Dict          : input_ids/labels, inputs/targets, x/y anahtarları
    4. List[Tensor]  : Tensor listesi (ilk ikisi kullanılır)

Dil Modeli Kayması (Language Model Shift):
    Tek tensor verildiğinde: inputs = tensor[:, :-1], targets = tensor[:, 1:]
    Bu, bir sonraki token tahmin paradigmasını uygular (autoregressive LM).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

__all__ = ["BatchProcessor"]

# Desteklenen sözlük anahtar çiftleri (öncelik sırasıyla)
_DICT_KEY_PAIRS: List[Tuple[str, str]] = [
    ("input_ids", "labels"),    # HuggingFace transformer tarzı
    ("inputs", "targets"),      # Genel isim
    ("x", "y"),                 # Kısa alias
    ("src", "tgt"),             # Seq2seq tarzı
    ("encoder_input", "decoder_target"),  # Kodlayıcı-çözücü tarzı
]


class BatchProcessor:
    """
    Cevahir V3 Batch Veri İşlemcisi
    ================================
    Eğitim döngüsünden gelen çeşitli batch formatlarını tek tip
    (inputs: Tensor, targets: Tensor) çıktısına dönüştürür.

    Özellikler:
        - Otomatik biçim algılama (Tensor, Tuple, Dict, List)
        - Dil modeli kaydırma (next-token prediction için)
        - Cihaz aktarımı (CPU/GPU)
        - Dtype doğrulama (uzun tamsayı beklenir)
        - Şekil doğrulama (batch ve seq boyutları uyumlu mu?)
        - Ayrıntılı hata mesajları (Türkçe)

    Kullanım:
        processor = BatchProcessor(device='cuda', pad_token_id=0)

        # Tensor batch (dil modeli kayması otomatik uygulanır)
        inputs, targets = processor.process(batch_tensor)

        # Sözlük batch
        inputs, targets = processor.process({'input_ids': x, 'labels': y})
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cpu",
        pad_token_id: int = 0,
        dtype: torch.dtype = torch.long,
        validate_shapes: bool = True,
        min_seq_len: int = 2,
        max_seq_len: Optional[int] = None,
    ):
        """
        Args:
            device         : Tensor'ların taşınacağı cihaz ('cpu', 'cuda', 'mps')
            pad_token_id   : Dolgu token ID'si (şekil doğrulamada kullanılır)
            dtype          : Beklenen tensor veri tipi (varsayılan: torch.long)
            validate_shapes: Giriş/hedef şekillerini doğrula
            min_seq_len    : Minimum kabul edilebilir sekans uzunluğu
            max_seq_len    : Maximum sekans uzunluğu (None = sınırsız)
        """
        self.device = torch.device(device)
        self.pad_token_id = pad_token_id
        self.dtype = dtype
        self.validate_shapes = validate_shapes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len

        logger.debug(
            f"BatchProcessor başlatıldı | device={device}, "
            f"dtype={dtype}, min_seq={min_seq_len}"
        )

    # ------------------------------------------------------------------
    # Ana İşleme Arayüzü
    # ------------------------------------------------------------------

    def process(
        self,
        batch: Union[torch.Tensor, Tuple, Dict, List],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch verisini (inputs, targets) çiftine dönüştürür.

        Args:
            batch: Aşağıdaki formatlardan herhangi biri:
                - torch.Tensor : [batch, seq_len+1] → kayma ile bölünür
                - tuple/list   : (inputs, targets) veya [inputs, targets]
                - dict         : input_ids/labels, inputs/targets vb. anahtarlar
                - List[Tensor] : İlk iki tensör kullanılır

        Returns:
            (inputs, targets): Her ikisi de [batch, seq_len] boyutlu LongTensor

        Raises:
            TypeError : Tanınmayan batch formatı
            ValueError: Geçersiz şekil veya dtype
        """
        # Biçimi algıla ve uygun işleyiciye yönlendir
        if isinstance(batch, torch.Tensor):
            inputs, targets = self._process_tensor(batch)

        elif isinstance(batch, dict):
            inputs, targets = self._process_dict(batch)

        elif isinstance(batch, (tuple, list)):
            inputs, targets = self._process_sequence(batch)

        else:
            raise TypeError(
                f"Tanınmayan batch tipi: {type(batch).__name__}. "
                f"Desteklenen tipler: Tensor, dict, tuple, list"
            )

        # Cihaza taşı — non_blocking=True: pinned_memory DataLoader ile async CUDA transfer
        # (CPU→GPU aktarımı, model forward başlamadan önce arka planda tamamlanır)
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        # Dtype dönüşümü (gerekirse)
        inputs, targets = self._ensure_dtype(inputs, targets)

        # Şekil doğrulama
        if self.validate_shapes:
            self._validate_shapes(inputs, targets)

        return inputs, targets

    # ------------------------------------------------------------------
    # Biçim-Özel İşleyiciler
    # ------------------------------------------------------------------

    def _process_tensor(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tek tensor batch'i dil modeli kaydırma ile işler.

        Dil modeli paradigması (next-token prediction / autoregressive):
            inputs  = tensor[:, :-1]  → girdi tokenları (son token hariç)
            targets = tensor[:, 1:]   → hedef tokenlar (ilk token hariç)

        Örnek (seq_len=5):
            tensor  : [BOS, T1, T2, T3, EOS]
            inputs  : [BOS, T1, T2, T3]      → modele verilir
            targets : [T1,  T2, T3, EOS]     → tahmin edilecek

        Args:
            tensor: [batch_size, seq_len+1] veya [seq_len+1] (batch=1 olarak ele alınır)

        Returns:
            (inputs, targets): Her biri [batch, seq_len]

        Raises:
            ValueError: Sekans uzunluğu kaydırma için çok kısa (< 2)
        """
        # Tek örneklik tensörü batch haline getir
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        if tensor.dim() != 2:
            raise ValueError(
                f"Tensor batch 1D veya 2D olmalı, alınan: {tensor.dim()}D "
                f"(şekil: {tensor.shape})"
            )

        seq_len = tensor.shape[1]
        if seq_len < 2:
            raise ValueError(
                f"Dil modeli kaydırması için sekans uzunluğu en az 2 olmalı. "
                f"Alınan: {seq_len}. "
                f"İpucu: Veri yükleyici token sayısı çok az olabilir."
            )

        # Otoregressif kaydırma: inputs son tokeni almaz, targets ilk tokeni almaz
        # [PERF FIX] .contiguous() CPU'da çağrılıp ardından .to(device) yapılıyordu:
        # bu, CPU'da ekstra bir kopya + GPU'ya transfer = toplam 2 tahsis demektir.
        # CUDA transferi (.to('cuda')) zaten contiguous tensor döndürür.
        # CPU eğitimi için downstream (F.linear vb.) non-contiguous slice'ı tolere eder.
        # Gerçekten gerekirse .to(device) sonrası contiguous() çağrılabilir.
        inputs = tensor[:, :-1]
        targets = tensor[:, 1:]

        return inputs, targets

    def _process_dict(
        self,
        batch: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sözlük batch'inden inputs ve targets çıkarır.

        Anahtar çiftlerini öncelik sırasıyla dener:
            1. ('input_ids', 'labels')          - HuggingFace stili
            2. ('inputs', 'targets')             - Genel stili
            3. ('x', 'y')                        - Kısa alias
            4. ('src', 'tgt')                    - Seq2seq stili
            5. ('encoder_input', 'decoder_target') - Kodlayıcı-çözücü stili

        Yalnızca input anahtarı varsa (labels yoksa) → tensor kaydırma uygulanır.

        Args:
            batch: Token ID tensörlerini içeren sözlük

        Returns:
            (inputs, targets)

        Raises:
            KeyError: Hiçbir bilinen anahtar çifti bulunamadı
        """
        batch_keys = set(batch.keys())

        # Bilinen anahtar çiftlerini sırayla dene
        for input_key, target_key in _DICT_KEY_PAIRS:
            if input_key in batch_keys:
                input_tensor = batch[input_key]

                if target_key in batch_keys:
                    # Her iki anahtar da mevcut: doğrudan al
                    target_tensor = batch[target_key]
                    logger.debug(
                        f"Dict batch: '{input_key}' → inputs, '{target_key}' → targets"
                    )
                    return self._ensure_tensor(input_tensor), self._ensure_tensor(target_tensor)
                else:
                    # Yalnızca giriş anahtarı var: dil modeli kaydırması uygula
                    logger.debug(
                        f"Dict batch: yalnızca '{input_key}' bulundu, kaydırma uygulanıyor"
                    )
                    return self._process_tensor(self._ensure_tensor(input_tensor))

        # Hiçbir bilinen anahtar bulunamadı
        available = sorted(batch_keys)
        known_keys = [k for pair in _DICT_KEY_PAIRS for k in pair]
        raise KeyError(
            f"Sözlük batch'inde tanınan giriş anahtarı bulunamadı.\n"
            f"Mevcut anahtarlar  : {available}\n"
            f"Beklenen anahtarlar: {known_keys}\n"
            f"İpucu: Veri setinizin anahtar adlarını kontrol edin."
        )

    def _process_sequence(
        self,
        batch: Union[Tuple, List],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tuple veya liste batch'ini işler.

        Desteklenen formatlar:
            - (inputs_tensor, targets_tensor)         → doğrudan kullan
            - [inputs_tensor, targets_tensor]         → doğrudan kullan
            - [tensor1, tensor2, ...]                 → ilk ikisini kullan
            - Tensor listesi → her biri tek örnek, stack edilir

        Args:
            batch: Tensor içeren tuple veya liste

        Returns:
            (inputs, targets)

        Raises:
            ValueError: Liste boş veya tek elemanlı
            TypeError : Elemanlar Tensor değil
        """
        if len(batch) == 0:
            raise ValueError("Boş batch alındı. Veri yükleyiciyi kontrol edin.")

        if len(batch) == 1:
            # Tek elemanlı liste/tuple → tensor kaydırma uygula
            single = batch[0]
            if isinstance(single, torch.Tensor):
                logger.debug("Tek elemanlı sequence batch: kaydırma uygulanıyor")
                return self._process_tensor(single)
            raise TypeError(
                f"Tek elemanlı batch Tensor bekleniyor, alınan: {type(single).__name__}"
            )

        # İki veya daha fazla elemanlı: ilk ikisini inputs/targets olarak kullan
        first = batch[0]
        second = batch[1]

        if not isinstance(first, torch.Tensor):
            raise TypeError(
                f"Sequence batch'inin ilk elemanı Tensor olmalı, "
                f"alınan: {type(first).__name__}"
            )
        if not isinstance(second, torch.Tensor):
            raise TypeError(
                f"Sequence batch'inin ikinci elemanı Tensor olmalı, "
                f"alınan: {type(second).__name__}"
            )

        if len(batch) > 2:
            logger.debug(
                f"Sequence batch {len(batch)} elemanlı, yalnızca ilk 2 kullanılıyor"
            )

        return first, second

    # ------------------------------------------------------------------
    # Doğrulama ve Dönüşüm Yardımcıları
    # ------------------------------------------------------------------

    def _ensure_tensor(self, x) -> torch.Tensor:
        """
        Girdiyi Tensor'a dönüştürür. Zaten Tensor ise olduğu gibi döner.

        Args:
            x: Tensor, liste veya sayı

        Returns:
            torch.Tensor
        """
        if isinstance(x, torch.Tensor):
            return x
        try:
            return torch.tensor(x)
        except Exception as e:
            raise TypeError(
                f"Tensor'a dönüştürülemedi: {type(x).__name__}. Hata: {e}"
            )

    def _ensure_dtype(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tensor dtype'larını beklenen dtype'a dönüştürür.

        Dil modelleri token ID'leri için torch.long (int64) gerektirir.
        Float veya int32 tensor'lar otomatik dönüştürülür.

        Args:
            inputs : Giriş token ID tensörü
            targets: Hedef token ID tensörü

        Returns:
            (inputs, targets) — her ikisi de self.dtype türünde
        """
        if inputs.dtype != self.dtype:
            logger.debug(
                f"inputs dtype dönüştürülüyor: {inputs.dtype} → {self.dtype}"
            )
            inputs = inputs.to(self.dtype)

        if targets.dtype != self.dtype:
            logger.debug(
                f"targets dtype dönüştürülüyor: {targets.dtype} → {self.dtype}"
            )
            targets = targets.to(self.dtype)

        return inputs, targets

    def _validate_shapes(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Giriş ve hedef tensor şekillerini doğrular.

        Kontroller:
            1. Her ikisi de en az 2D olmalı
            2. Batch boyutları eşleşmeli
            3. Sekans uzunlukları eşleşmeli
            4. Sekans uzunluğu min_seq_len'den az olmamalı
            5. Sekans uzunluğu max_seq_len'i aşmamalı (ayarlıysa)

        Args:
            inputs : [batch, seq_len] giriş tensörü
            targets: [batch, seq_len] hedef tensörü

        Raises:
            ValueError: Herhangi bir kontrol başarısız olursa
        """
        # Boyut kontrolü
        if inputs.dim() < 2:
            raise ValueError(
                f"inputs en az 2D olmalı (batch, seq), alınan: {inputs.dim()}D"
            )
        if targets.dim() < 2:
            raise ValueError(
                f"targets en az 2D olmalı (batch, seq), alınan: {targets.dim()}D"
            )

        # Batch boyutu uyumu
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError(
                f"inputs ve targets batch boyutları eşleşmiyor: "
                f"inputs={inputs.shape[0]}, targets={targets.shape[0]}"
            )

        # Sekans uzunluğu uyumu
        if inputs.shape[1] != targets.shape[1]:
            raise ValueError(
                f"inputs ve targets sekans uzunlukları eşleşmiyor: "
                f"inputs={inputs.shape[1]}, targets={targets.shape[1]}. "
                f"İpucu: Kaydırma işlemi her ikisini de eşit uzunlukta bırakmalı."
            )

        seq_len = inputs.shape[1]

        # Minimum uzunluk kontrolü
        if seq_len < self.min_seq_len:
            raise ValueError(
                f"Sekans uzunluğu ({seq_len}) minimum değerin ({self.min_seq_len}) altında. "
                f"Tokenizasyon veya veri kırpma ayarlarını kontrol edin."
            )

        # Maksimum uzunluk kontrolü
        if self.max_seq_len is not None and seq_len > self.max_seq_len:
            raise ValueError(
                f"Sekans uzunluğu ({seq_len}) maksimum değeri ({self.max_seq_len}) aşıyor. "
                f"max_seq_len ayarını artırın veya veri kırpmayı etkinleştirin."
            )

    # ------------------------------------------------------------------
    # Yardımcı Metotlar
    # ------------------------------------------------------------------

    def get_batch_stats(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict:
        """
        İşlenmiş batch hakkında istatistiksel bilgi döndürür.
        Debug ve izleme amaçlıdır.

        Args:
            inputs : İşlenmiş giriş tensörü [batch, seq_len]
            targets: İşlenmiş hedef tensörü [batch, seq_len]

        Returns:
            Batch istatistiklerini içeren sözlük:
                - batch_size   : Batch büyüklüğü
                - seq_len      : Sekans uzunluğu
                - total_tokens : Toplam token sayısı
                - pad_ratio    : Dolgu token oranı
                - device       : Cihaz adı
                - dtype        : Veri tipi
        """
        with torch.no_grad():
            batch_size, seq_len = inputs.shape[:2]
            total_tokens = batch_size * seq_len

            # Dolgu oranı (hedef üzerinden hesapla)
            pad_count = (targets == self.pad_token_id).sum().item()
            pad_ratio = pad_count / total_tokens if total_tokens > 0 else 0.0

        return {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "total_tokens": total_tokens,
            "pad_ratio": pad_ratio,
            "device": str(inputs.device),
            "dtype": str(inputs.dtype),
        }

    def to_device(self, device: Union[str, torch.device]) -> "BatchProcessor":
        """
        İşlemcinin hedef cihazını değiştirir.

        Args:
            device: Yeni hedef cihaz

        Returns:
            self (zincirleme kullanım için)
        """
        self.device = torch.device(device)
        logger.info(f"BatchProcessor cihazı güncellendi: {self.device}")
        return self

    def __repr__(self) -> str:
        return (
            f"BatchProcessor("
            f"device={self.device}, "
            f"dtype={self.dtype}, "
            f"pad_token_id={self.pad_token_id}, "
            f"validate_shapes={self.validate_shapes}, "
            f"min_seq_len={self.min_seq_len}"
            f")"
        )
