# Research ledger

Date: 2026-09-14. Public sources were located during discovery; no external implementation was copied. The user subsequently prioritized repairing the existing system on limited hardware. New research backends below are deferred, not accepted implementations.

| Technique | Paper / primary reference | Public implementation / license | Claimed advantage | Cevahir implementation / benchmark / result | Decision and reason |
|---|---|---|---|---|---|
| FSDP2 / distributed checkpoint | [PyTorch fully_shard](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html), [DCP](https://docs.pytorch.org/docs/2.8/distributed.checkpoint.html) | PyTorch; license must be checked before copying any source | Parameter sharding and portable training state | Existing FSDP1/DDP paths audited; GPU benchmark unavailable | Defer new backend; fix current contracts first |
| Muon | [PyTorch Muon](https://docs.pytorch.org/docs/main/generated/torch.optim.Muon.html) | `torch.optim.Muon`; reuse installed API rather than copying algorithm | Matrix-parameter optimizer alternative | No AdamW comparison yet | Defer; no demonstrated Cevahir benefit |
| MLA | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | [DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2); code/weight license review pending | Compressed latent KV representation | No implementation or benchmark | Defer; correct existing MHA/GQA cache first |
| Auxiliary-loss-free balancing / MTP | [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3); code/weight license review pending | Adaptive routing load / additional future-token objectives | Existing auxiliary MoE loss currently not consumed; no new backend or objective | Defer; repair current loss composition first |
| Gated DeltaNet / sparse attention | Source review pending | Not selected; license not evaluated | Research question, not an established Cevahir advantage | No implementation or benchmark | Defer by user scope |
| Gated residual / n-gram embedding | Source review pending | Not selected; license not evaluated | Research question, particularly Turkish morphology | No implementation or benchmark | Defer by user scope |
| Native multimodality | No specific architecture selected | No implementation copied | Separate vision-to-language fusion track | Existing processor wrappers do not implement native fusion | Defer; preserve language core |

Performance claims in papers refer to their models, hardware and experiments. None are transferred to Cevahir. Acceptance requires reproducible Cevahir measurements and regression evidence. CPU synthetic loss/perplexity measures arithmetic and optimization behavior only, not Turkish language quality.
