# Training System — from prepared data to a training run

[Türkçe](README.md) · [English](README-en.md) · [Architecture](../../architecture/CEVAHIR_ARCHITECTURE_SPEC.md)

**Matched against code:** September 20, 2026. **Active service:** `TrainingServiceV3`.

This layer assembles the tokenizer, prepared data, model, criterion, optimizer,
and training manager for a run. Gradient updates belong to
[training_management](../training_management/README-en.md).

## Actual execution path

```text
Existing tokenizer or tokenizer_management/train_bpe.py
  → training_system/prepare_cache.py
  → training_system/train.py
  → TrainingServiceV3
  → TrainingManager V2 → model training
```

[train.py](../../../training_system/train.py) selects the V3 service when it imports
successfully, otherwise it falls back to the V2 service. V3 runtime errors do not
automatically route execution to the experimental V3 TrainingManager or another backend.
The supported backend for both service paths is `training_backend="v2"`.
Service V3 is a separate version label from the neural network's V5–V8 evolution notes.

## Responsibilities and sources

| Component | Function |
|---|---|
| [prepare_cache.py](../../../training_system/prepare_cache.py) | Tokenizes and formats data into a training cache |
| [TrainingServiceV3](../../../training_system/v3/core/training_service_v3.py) | Assembles device, tokenizer, data, model, and training components |
| [ConfigManagerV3](../../../training_system/v3/core/config_manager_v3.py) | Passes service settings to model and training layers |
| [DataCacheV3](../../../training_system/v3/data/cache_v3.py) | Cache identity, metadata, and integrity verification |
| [cache_identity.py](../../../training_system/cache_identity.py) | Shared data and tokenizer content identities |
| [data_split.py](../../../training_system/data_split.py) | Train / validation split preserving source and identical-content groups |
| [DataLoader V3](../../../training_system/v3/data/dataloader_v3.py) | Dataset, length buckets, dynamic padding, and worker settings |
| [V2 TrainingService](../../../training_system/v2/core/training_service.py) | Compatibility for older service entry points |

## Preparing data

Run these commands from the repository root. Reusing the tokenizer that matches an
existing training run does not require retraining it. If a new tokenizer is needed:

```powershell
python tokenizer_management/train_bpe.py education
```

After selecting tokenizer and formatting settings, prepare the cache:

```powershell
python training_system/prepare_cache.py --data-dir education --no-clear-cache
```

`--no-clear-cache` prevents bulk clearing of existing cache files.
Without this option, the preparation tool clears the old cache.
`--max-seq-length`, `--include-whole-words`, `--include-syllables`, and `--include-sep`
affect the prepared records and must match the training configuration.

The final step is `python training_system/train.py` with suitable run settings.
The supplied profile includes 100 epochs, batch size 64, and substantial model settings;
it is not a quick check profile for a low-resource computer.
No new training run was performed to update this document.

## Cache and split contract

- The V3 service requires prepared cache data; it does not fill a missing cache by processing raw data.
- Data, tokenizer, and formatting settings contribute to cache identity.
- Checksums verify file integrity, not the semantic correctness of training examples.
- Chunks sharing a source identity remain on the same side of the split.
- Identical input / target contents also connect different sources and keep them together.
- Mixed or missing source IDs are rejected; when all IDs are absent, content grouping still applies.
- At least two independent groups are required; the requested ratio applies to groups.
- Near duplicates and different chunks of an unidentified shared document are not fully covered.

Bucket batching groups similar lengths; dynamic padding adjusts to each batch.
These mechanisms aim to reduce unnecessary padding work.
CPU and CUDA loading settings exist; realized gains depend on data and hardware.

## Model and training state

The service binds tokenizer identity to the model and training settings.
Without an explicit resume path it searches the checkpoint directory, preferring
`last.pth`, then `best.pth`. The selected record is restored by TrainingManager.
See the [training manager guide](../training_management/README-en.md) for model and optimizer restoration.
Unsupported optimization flags are rejected by the [shared contract](../../../training_management/contracts.py).

## Next development cycle

The current `_validate_alignment()` only checks whether input and target are identical
in the first example; it does not establish the correct next-token relation for every record.
Format-aware token range, masking, and alignment validation should cover the full dataset.
Split provenance and near-duplicate handling also need explicit reporting.

Detailed priorities are recorded in the [development roadmap](../../architecture/NEXT_DEVELOPMENT_ROADMAP.md).
[Historical outputs from real training](../../../README.md) and
[small verification measurements](../../../benchmarks/README.md) are presented in their own contexts.
