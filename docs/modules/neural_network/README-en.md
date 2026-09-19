# Neural network core

**Updated:** 20 September 2026 · [Türkçe](README.md)

`CevahirNeuralNetwork` is the configurable Transformer core that converts token sequences into next-token scores. Language-model use applies causal attention; tokenization, the training loop, text generation and cognitive orchestration are separate surrounding layers. The core combines attention, positional encoding, dense or expert-based feed-forward computation, memory and training options within one model construction path.

The main implementation is [neural_network.py][core], with components in [neural_network_module][layers]. See the [architecture contract][architecture] for the complete system and the [development roadmap][roadmap] for the next work cycle.

## What do V4, V7 and V8 mean?

Remaining V4/V6 headings in source do not limit the model to those capabilities. The labels below identify development stages in source comments; they do not describe five independent models or one release number shared by the entire project.

| Label in source | Implementation in the current code |
| --- | --- |
| V4 | RoPE integration with attention; RMSNorm, SwiGLU, weight tying, KV cache, activation checkpointing and optional MoE. |
| V5 | `num_kv_heads` for GQA/MQA, sliding-window attention and YaRN/linear RoPE scaling. |
| V6 | PyTorch SDPA routing, QK-Norm, parallel residual, output/attention logit soft-caps and residual projection initialization changes. |
| V7 | Depth-dependent stochastic depth; merged FFN `gate_up_proj`; propagation of KV eviction and attention sink settings through the layer. |
| V8 component fixes | Sink/window boundary validation in KV cache and changes to MoE auxiliary-loss accumulation and consumption. |

This map follows the [main model][core], [Transformer layer][transformer], [FFN][ffn] and [KVCache][cache]. For example, KVCache carries V5 comments while exposing its options through the parent layer is marked V7.

The configuration fields `config_version=1` and `architecture_version="cevahir-capabilities-1"` belong to the [configuration schema][config]. They are separate from these historical labels. Weight-file compatibility depends on the actual construction settings, tensor shapes and tokenizer identity, rather than a V number in a filename.

## Execution flow

```text
Token IDs [B, T]
  → LanguageEmbedding
  → positional encoding + dropout
  → TransformerEncoderLayer × N
      → normalization + MultiHeadAttention
      → residual + FFN or MixtureOfExperts
  → output normalization
  → vocabulary projection + optional logit soft-cap
  → logits [B, T, vocab_size]
```

RoPE is applied to Q/K tensors inside attention. Sinusoidal and learned positional encoding alternatives are also available. Although the class is named `TransformerEncoderLayer`, its causal configuration serves a decoder-style language-model flow. `parallel_residual=True` together with `pre_norm=True` selects a path where attention and FFN branches consume the same normalized input; the post-norm path remains sequential.

| Component | Role and relevant options |
| --- | --- |
| [Main model][core] | Constructs the layer stack, ties output and embedding weights when applicable, and propagates cache and diagnostic requests. |
| [TransformerEncoderLayer][transformer] | Pre/post norm, residual connections, stochastic depth and activation checkpointing; dense FFN or MoE branch. |
| [MultiHeadAttention][attention] | MHA/GQA/MQA, causal/padding masks, RoPE, QK-Norm, sliding windows and backend selection. |
| [FeedForwardNetwork][ffn] | Merged gate/up projection for gated paths such as SwiGLU/GeGLU; alternative activations. |
| [MixtureOfExperts][moe] | Routes tokens to selected experts and produces an auxiliary loss for the training objective to consume. |
| [KVCache][cache] | Retains K/V state during autoregressive generation; enforces capacity and eviction policy. |

## Configuration and boundaries

New model construction uses [ModelManager][manager] and the [shared configuration schema][config]. Dimensions, head relationships and feature settings are normalized and validated there. Legacy direct-constructor defaults should not be assumed identical to schema defaults.

- `num_heads` must be divisible by `num_kv_heads`; head dimension is derived from `embed_dim / num_heads`.
- `seq_proj_dim` remains a compatibility field for older files; it does not construct an independent sequence-projection layer. A different value can affect the weight-tying choice.
- YaRN scaling and larger cache capacity are configurable. A trained model's language quality at the new sequence length requires separate evaluation.
- Selecting SDPA does not establish that a particular GPU kernel ran. Backend choice depends on settings, tensors, PyTorch and hardware.
- `attn_logit_cap > 0` or requesting attention weights selects manual attention computation, changing memory and runtime costs.
- Activation checkpointing recomputes intermediate activations during training. KV caching and model checkpoints saved to disk have different purposes.
- Quantization is not applied automatically in the constructor; `apply_quantization()` is a separate lifecycle step. Available paths and dependencies are defined in the [quantization manager][quantization].

## Output and observability

An ordinary core call returns `(logits, attention_weights)`; `use_cache=True` adds layer cache outputs as a third value. `return_attention_weights=False` is the default, so ordinary execution does not return an attention matrix. When requested, the returned matrix belongs to the final layer.

`collect_diagnostics=True` collects tensor statistics for the latest call. An ordinary call retains only small shape/type metadata; enabling TensorBoard can also collect statistics at its configured interval. These controls allow inspection without making it a mandatory cost on every training or generation step.

Custom MoE training loops must consume `get_and_reset_moe_loss()` and add the auxiliary loss to their training objective. Standalone generation flows using caching must also manage state between requests; the core itself is not a session manager.

## Small CPU example

Run from the project root in an environment with PyTorch:

```python
import torch
from model_management.config_schema import tiny_model_config
from model_management.model_manager import ModelManager

torch.set_num_threads(1)
manager = ModelManager(tiny_model_config())
manager.initialize(build_optimizer=False, build_criterion=False, build_scheduler=False)
logits, attention = manager.forward(torch.tensor([[1, 5, 8, 2]]), inference=True)
print(logits.shape)  # torch.Size([1, 4, 128])
```

This example demonstrates the connections of a newly initialized small model. Weights from the project's previous real training and the [training outputs in the main README][training] are distinct project assets that must be preserved.

## Preserving trained weights

Model saving and loading use the [shared checkpoint contract][checkpoint]. Before loading, the recorded construction settings, architecture choices that can change computation without changing tensor shapes, and state shapes are checked. A file carrying tokenizer identity requires that identity to be verified; equal vocabulary size alone is insufficient.

Older files may lack architecture or tokenizer metadata. Preserve original weights and the tokenizer used during training, and investigate compatibility with an explicit configuration. Renaming an older file with a new version label does not migrate it. New architecture experiments should produce separate outputs from the preserved training files.

The next priorities are model/cache ownership, explicit migration rules for old checkpoints and consistent semantics across execution paths. Detailed scope is maintained in the [development roadmap][roadmap].

[core]: ../../../src/neural_network.py
[layers]: ../../../src/neural_network_module
[transformer]: ../../../src/neural_network_module/ortak_katman_module/transformer_encoder_layer.py
[attention]: ../../../src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py
[ffn]: ../../../src/neural_network_module/ortak_katman_module/feed_forward_network.py
[moe]: ../../../src/neural_network_module/ortak_katman_module/mixture_of_experts.py
[cache]: ../../../src/neural_network_module/ortak_katman_module/kv_cache.py
[quantization]: ../../../src/neural_network_module/ortak_katman_module/quantization_manager.py
[config]: ../../../model_management/config_schema.py
[manager]: ../../../model_management/model_manager.py
[checkpoint]: ../../../model_management/checkpoint_contract.py
[architecture]: ../../architecture/CEVAHIR_ARCHITECTURE_SPEC.md
[roadmap]: ../../architecture/NEXT_DEVELOPMENT_ROADMAP.md
[training]: ../../../README.md
