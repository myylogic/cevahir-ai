# -*- coding: utf-8 -*-
"""
Advanced Token Metrics - Per-token accuracy ve special token tracking
EOS, BOS, PAD gibi special token'ların öğrenilip öğrenilmediğini track eder.
"""

import torch
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger("AdvancedTokenMetrics")


class AdvancedTokenMetrics:
    """Per-token accuracy ve special token tracking"""
    
    def __init__(self, vocab_size: int, special_tokens_dict: Dict[int, str]):
        """
        Args:
            vocab_size: Vocabulary size
            special_tokens_dict: {token_id: token_name}, e.g. {0: 'PAD', 1: 'BOS', 2: 'EOS', 3: 'UNK'}
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens_dict
        logger.info(f"[INIT] AdvancedTokenMetrics: vocab_size={vocab_size}, special_tokens={special_tokens_dict}")
    
    def compute_per_token_accuracy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[float, Dict[str, Tuple[float, int]], float, float]:
        """
        Compute detailed per-token metrics.
        
        Args:
            logits: (B*T, vocab_size) - model output
            targets: (B*T,) - ground truth token IDs
            
        Returns:
            (overall_acc, special_accs, top5_acc, entropy)
            where special_accs = {token_name: (accuracy, count)}
        """
        # Sayısal kararlılık: AMP/float16 ile softmax önce float32 (entropy doğru hesaplansın)
        logits_float = logits.float() if logits.dtype != torch.float32 else logits
        probs = torch.softmax(logits_float, dim=-1)
        predictions = probs.argmax(dim=-1)
        
        # Get PAD token ID (should be in special_tokens)
        pad_id = None
        for token_id, token_name in self.special_tokens.items():
            if token_name == "PAD":
                pad_id = token_id
                break
        
        # 1. Overall accuracy (excluding PAD tokens - standard NLP practice)
        if pad_id is not None:
            mask = (targets != pad_id)
            if mask.any():
                overall_acc = (predictions[mask] == targets[mask]).float().mean().item()
            else:
                overall_acc = 0.0
        else:
            # Fallback if PAD not defined (use all tokens)
            overall_acc = (predictions == targets).float().mean().item()
        
        # 2. Per-special-token accuracy
        special_accs = {}
        for token_id, token_name in self.special_tokens.items():
            mask = (targets == token_id)
            if mask.any():
                acc = (predictions[mask] == targets[mask]).float().mean().item()
                count = mask.sum().item()
                special_accs[token_name] = (acc, count)
            else:
                special_accs[token_name] = (0.0, 0)
        
        # 3. Top-5 accuracy (also excluding PAD tokens)
        if logits.shape[-1] >= 5:
            top5_preds = probs.topk(5, dim=-1)[1]
            top5_matches = (top5_preds == targets.unsqueeze(-1)).any(dim=-1).float()
            if pad_id is not None:
                mask = (targets != pad_id)
                if mask.any():
                    top5_acc = top5_matches[mask].mean().item()
                else:
                    top5_acc = 0.0
            else:
                top5_acc = top5_matches.mean().item()
        else:
            top5_acc = overall_acc  # Fallback if vocab < 5
        
        # 4. Entropy (measure of probability collapse)
        # KRİTİK: Sadece PAD-dışı pozisyonlarda hesapla; PAD'de model eğitilmediği için
        # dağılım uniform kalır ve ortalama entropy ~ln(V) çıkar (sabit 11.0 gibi).
        # Sadece içerik pozisyonlarında entropy = modelin gerçek "güven" göstergesi.
        ent_per_pos = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)  # (B*T,) veya (B,T)
        if pad_id is not None:
            non_pad = (targets != pad_id)
            if non_pad.any():
                entropy = ent_per_pos[non_pad].mean().item()
            else:
                entropy = ent_per_pos.mean().item()
        else:
            entropy = ent_per_pos.mean().item()
        
        return overall_acc, special_accs, top5_acc, entropy
    
    def compute_special_token_probabilities(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute average probability assigned to each special token.
        
        Args:
            logits: (B*T, vocab_size)
            targets: (B*T,) - for reference
            
        Returns:
            {token_name: avg_probability}
        """
        probs = torch.softmax(logits, dim=-1)
        token_probs = {}
        
        for token_id, token_name in self.special_tokens.items():
            # Average probability when this token appears in targets
            mask = (targets == token_id)
            if mask.any():
                # Probability of predicting this token when it's the target
                pred_prob = probs[mask, token_id].mean().item()
            else:
                pred_prob = probs[:, token_id].mean().item()  # Overall average
            
            token_probs[token_name] = pred_prob
        
        return token_probs
    
    def compute_loss_per_token(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: torch.nn.Module
    ) -> Dict[str, float]:
        """
        Compute loss contribution per token type.
        
        Args:
            logits: (B*T, vocab_size)
            targets: (B*T,)
            loss_fn: CrossEntropyLoss or similar
            
        Returns:
            {token_name: loss_value}
        """
        token_losses = {}
        
        for token_id, token_name in self.special_tokens.items():
            mask = (targets == token_id)
            if mask.any():
                token_loss = loss_fn(logits[mask], targets[mask])
                token_losses[token_name] = token_loss.item()
            else:
                token_losses[token_name] = 0.0
        
        return token_losses
    
    def format_token_metrics(
        self,
        overall_acc: float,
        special_accs: Dict[str, Tuple[float, int]],
        top5_acc: float,
        entropy: float,
        grad_norm: Optional[float] = None
    ) -> str:
        """Format metrics as readable string."""
        msg = f"\n{'='*70}\n[TOKEN METRICS]\n"
        msg += f"├─ Overall Accuracy: {overall_acc:.2%}\n"
        msg += f"├─ Special Token Accuracies:\n"
        
        for token_name, (acc, count) in special_accs.items():
            if count > 0:
                if token_name == "PAD":
                    status = "[MASKED]" if acc <= 0.1 else "[OK]"  # PAD loss'ta ignore_index
                elif acc > 0.5:
                    status = "[OK]"
                elif acc > 0.1:
                    status = "[LOW]"
                else:
                    status = "[CRITICAL]"
                msg += f"│  ├─ {token_name:6s}: {acc:6.2%} ({count:4d} samples) {status}\n"
            else:
                msg += f"│  ├─ {token_name:6s}: Not in targets\n"
        
        msg += f"├─ Top-5 Accuracy: {top5_acc:.2%}\n"
        max_ent = torch.log(torch.tensor(float(self.vocab_size))).item()
        msg += f"├─ Entropy (content only): {entropy:.4f} (max: {max_ent:.4f})\n"
        
        if grad_norm is not None:
            msg += f"└─ Gradient Norm: {grad_norm:.6f}\n"
        else:
            msg += f"└─ [No gradient info]\n"
        
        msg += f"{'='*70}"
        return msg
    
    def check_critical_issues(
        self,
        overall_acc: float,
        special_accs: Dict[str, Tuple[float, int]],
        entropy: float
    ) -> list:
        """
        Check for critical training issues.
        
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Check EOS accuracy
        if 'EOS' in special_accs:
            eos_acc, eos_count = special_accs['EOS']
            if eos_count > 0 and eos_acc < 0.01:
                warnings.append(
                    f"[CRITICAL] EOS Token Not Learning! Accuracy: {eos_acc:.4f} "
                    f"→ Check eos_token_weight and label_smoothing"
                )
        
        # Check BOS accuracy
        if 'BOS' in special_accs:
            bos_acc, bos_count = special_accs['BOS']
            if bos_count > 0 and bos_acc < 0.05:
                warnings.append(
                    f"[WARNING] BOS Token Low Accuracy: {bos_acc:.4f}"
                )
        
        # Check mode collapse / low entropy (çok düşük entropy = dağılım çok sivri)
        # Eğitimde doğru token verildiği için accuracy yüksek olabilir; üretimde kendi çıktısı
        # kullanıldığında tekrarlayan veya konu dışı cevaplar görülebilir (exposure bias).
        max_entropy = torch.log(torch.tensor(float(self.vocab_size))).item()
        if entropy < max_entropy * 0.1:  # Max'in %10'undan az (örn. vocab 60k → eşik ~1.1)
            if overall_acc >= 0.6:
                warnings.append(
                    f"[INFO] Low entropy (Entropy={entropy:.4f}, max={max_entropy:.4f}). "
                    "Model çok güvenli tahmin yapıyor. Epoch sonu üretim kötüyse: veri çeşitliliği, "
                    "temperature/top_p veya exposure bias (öğretmen zorlaması vs serbest üretim) düşünülebilir."
                )
            else:
                warnings.append(
                    f"[WARNING] Mode Collapse Detected: Entropy={entropy:.4f} "
                    f"(max={max_entropy:.4f}). Accuracy düşük — model tekrarlayan/benzer çıktılara kilitlenmiş olabilir."
                )
        
        # Check if no learning is happening
        if overall_acc < 0.001 and entropy < max_entropy * 0.2:
            warnings.append(
                "[CRITICAL] Model Not Learning! Overall accuracy < 0.1% and low entropy"
            )
        
        return warnings
