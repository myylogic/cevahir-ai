# Training Management — optimization loop and training state

[Türkçe](README.md) · [English](README-en.md) · [Architecture](../../architecture/CEVAHIR_ARCHITECTURE_SPEC.md)

**Matched against code:** September 20, 2026. **Active backend:** `training_management.v2`.

This module manages how the model learns: forward passes, loss, backpropagation,
optimizer updates, validation, learning rates, and persisted training state.
Data preparation and service assembly belong to [training_system](../training_system/README-en.md).

## Versions and execution path

```text
training_system/train.py
  → TrainingServiceV3 (V2 TrainingService if the V3 import fails)
  → training_backend="v2"
  → training_management.v2.TrainingManager
  → TrainingLoop → model → loss → backward → optimizer
```

| Name | Meaning in the current system |
|---|---|
| TrainingService V3 | Upper layer assembling caches, data loaders, and training components |
| TrainingManager V2 | Actively developed optimization backend used by the service |
| TrainingManager V3 | Separate experimental implementation, outside the supported service path |
| Neural network V5–V8 notes | Evolution markers in model components, separate from manager versions |

The V2 name does not mean the implementation contains only older features.
Current accumulation, precision, MoE auxiliary loss, and checkpoint identity fixes
are on this active path. [contracts.py](../../../training_management/contracts.py) defines its supported capabilities.

## Active components

| Component | Responsibility |
|---|---|
| [TrainingManager](../../../training_management/v2/core/training_manager.py) | Epoch lifecycle, validation, early stopping, callbacks, and resume |
| [TrainingLoop](../../../training_management/v2/core/training_loop.py) | Batch processing, autocast, accumulation, and optimizer updates |
| [LossComputation](../../../training_management/v2/core/loss_computation.py) | Token loss, PAD / `-100` masking, accuracy, and loss-derived perplexity |
| [GradientManager](../../../training_management/v2/core/gradient_manager.py) | Gradient clipping and norm tracking |
| [TrainingScheduler](../../../training_management/v2/utils/training_scheduler.py) | Warmup and epoch / metric based learning rate scheduling |
| [CheckpointManager](../../../training_management/v2/utils/checkpoint_manager.py) | Training checkpoints, best / last aliases, and retention |

## Learning contract

- Model logits have shape `[batch, sequence, vocabulary]`; targets have shape `[batch, sequence]`.
- PAD and `-100` targets are excluded from loss and accuracy.
- Class weights, label smoothing, and the entropy coefficient are read from the criterion.
- The model's already weighted MoE auxiliary loss enters the training objective once per forward pass.
- Accumulation is normalized by valid target tokens, including the final incomplete group.
- Warmup tracks completed optimizer updates; epoch changes are passed to the sampler.
- Nonfinite total objectives or gradients must not produce an invalid optimizer update.

Precision options are `auto`, `fp32`, `fp16`, and `bf16`.
CPU `fp16` requests fall back to `fp32`; explicit `bf16` uses CPU autocast.
CUDA `bf16` capability is checked; CUDA `fp16` uses GradScaler.
These are device-dependent capabilities, not GPU performance measurements.

## Checkpoints and resume

Current checkpoints contain the model, optimizer, model construction information,
and tokenizer identity. Additional training state covers the scheduler, scaler,
optimizer step count, best loss, early stopping, and Python / NumPy / Torch RNG state.

`resume_from_checkpoint()` checks model identity, tensor shapes, and tokenizer identity.
After successful restoration, training continues at the next epoch.
`epochs` is the number of additional epochs. Mid-batch resume is not provided.
Legacy records remain usable; missing extra state produces a warning about incomplete resume.
Resuming training is distinct from loading only model weights for inference.

The active `TrainingManager.resume_from_checkpoint()` propagates optimizer restoration errors.
The separate `CheckpointManager.load()` helper logs the same error as a warning and continues.
These error policies are not unified; neither path restores all training state atomically.

## Supported scope

The active service rejects `training_backend="v3"`.
Enabled EMA, SWA, SAM, Lookahead, LLRD, curriculum, scheduled sampling, focal loss,
AGC, gradient noise, and cosine restart flags raise explicit errors.
Only `distributed_strategy="none"` is supported.
The existence of [V3 source code](../../../training_management/v3/core/training_manager.py)
does not establish integration of those features into the supported training path.

## Next development cycle

1. Prevent partial restoration during resume and unify error handling with the helper loading path.
2. Remove full-memory serialization and the fixed temporary filename from the training checkpoint writer.
3. Map experimental V3 helpers to the active V2 path and define integration and compatibility contracts for selected features.
4. Report loss components separately and relate perplexity interpretation to the actual objective.

The [development roadmap](../../architecture/NEXT_DEVELOPMENT_ROADMAP.md)
prioritizes this work alongside model-core and cognitive-system development.

## Evidence and use

The historical training outputs in the [main README](../../../README.md) come from real runs.
Small contract checks do not replace those runs or measure the trained model's quality.
See the [benchmark guide](../../../benchmarks/README.md) for the scope of recorded validation.
Use the [service guide](../training_system/README-en.md) for training startup, tokenizer setup, and cache preparation.
