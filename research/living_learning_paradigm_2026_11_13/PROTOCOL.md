# CEV-11 "Paradigma" — PROTOCOL (DRAFT, not yet frozen)

Status: draft. It becomes binding when it is committed with the SHA-256 manifest of the code tree, generator
configuration, tokenizer recipe, frozen edit lists, thresholds and analyze.py, before the first confirmatory
pretraining run. Deviations after the freeze go only into AMENDMENTS.md (timestamped).

Pre-freeze adjustments by the lead (recorded here, applied to the judge's text below):
- Location: the judge proposed `experiments/cev11_paradigma/`. Following the repository convention the study lives
  in `research/living_learning_paradigm_2026_11_13/` and the report in
  `docs/research/living_learning_paradigm_2026_11_13/REPORT_TR.md`. Paths in the text below are to be read that way.
- Initialisation: the judge wrote "repo default initialisation; the residual-init bug is unconfirmed". After that
  text was written the lead confirmed and fixed it (commit 65ce430: residual projections now use std 0.02/sqrt(2L);
  bigram task 3/3 seeds faster, copy task no consistent difference). "Repo default" therefore means the fixed rule.
  The double embedding dropout is also fixed (65ce430); the study uses dropout 0.0 anyway.
- Phase 0 (measured 2026-11-13 05:45 UTC, single thread, machine lightly loaded): D192/L6 3.45M params,
  2,729 train tok/s, 8,678 eval tok/s; D192/L4 2.56M, 3,427 / 12,407 tok/s. torch 2.15.0, numpy 2.5.4.
  The depth fallback (<1,500 tok/s) does not trigger: primary depth L6.
- v* step-size grid widened from {0.5, 2.0} to {0.5, 2.0, 5.0} (selection still on discovery worlds only, by the
  same rule). Reason: the compute-realism strategist's pilot (scratch, discovery evidence, not pooled) found that
  with residual-stream norms of 169-381 a ROME-style step of 0.5 left |delta| near 60 and 0% efficacy, while a
  scaled step reached |delta| near 227 and 100% efficacy; the lead's own pilot also failed at step 0.2. The same
  pilot found that in a 4-layer model a write at the subject token steers predictions only at layer 0, while the
  prediction position works at every layer; the protocol's PRED site in the binding-site map and the map-only
  switch (section 8) already cover that outcome.
- Tokenizer fact established during implementation (cev11/tokenizers.py, property test in
  tests/evolution/test_cev11_tokenizers.py): with the repo byte-BPE, RESEG cannot occur. The informal word is one
  pre-token and the formal root is its own pre-token; BPE applies the lowest-ranked merge everywhere at each step and
  never undoes a merge, so if no merge crosses the root end the root part follows exactly the steps of BPE on the
  root alone. Hence NONIDENT = FUSED for the primary arms (0 RESEG in 15,000 random merge tables and on real prose).
  The classifier keeps all three classes as preregistered. Scratch measurement on book prose: informal FUSED
  39.5-40.1% overall (gen ~56%, acc 59-64%, dat ~45%, abl 25-27%, ins ~30%, loc ~20%), formal fused 0%.
- Analysis interpretation choices (implemented in analyze.py / cev11/stats.py, accepted by the lead before the freeze):
  (1) world-level Leak_X, like PE/IE/CE, is the mean of per-edit shares; (2) Holm always runs over the whole family
  (m = 3 primary, m = 4 secondary); an untested, insufficient or overridden member enters with p = 1; (3) the
  manipulation check is the per-world NONIDENT share over the informal target forms of STD's H1 units, averaged over
  the analysed confirmatory worlds; (4) S1 non-inferiority uses the upper end of the two-sided 95% t-interval
  (a conservative one-sided 97.5% bound) over the S1 worlds on the same edits; "not supported (locality cost)" is
  assigned only when Holm rejects but non-inferiority fails or cannot be computed; (5) S3: failure = 1 - correct/7,
  10,000 seeded within-world permutations, supported requires Holm rejection and rho > 0 with p <= 0.05;
  (6) S4: the unit is an entity with RN neighbours at layers 0 and 4 (>= 15 per world), medians over pooled
  (entity, neighbour) tau values; (7) S1 and S2 use the inflected-lived edits only (section 7); (8) a confirmatory
  (world, arm) must contain all 10 edits per lived case, otherwise the analysis stops (a crashed run is rerun).

PROTOCOL — CEV-11 'Paradigma' v1.0. Freeze as experiments/cev11_paradigma/PROTOCOL.md. Commit it, with SHA-256 of the code tree, generator configuration, tokenizer recipe, frozen edit lists, thresholds and analyze.py, before the first confirmatory pretraining run. Deviations go only into a timestamped AMENDMENTS.md. Any analysis not listed here is exploratory.

0 SCOPE (to appear verbatim in the report)
Writes are computed by an external optimiser from gold lived sentences (entity, relation, object) against a frozen base. They are stored as an additive term on one FFN down-projection (layers[l].ffn.fc2) of CevahirNeuralNetwork, with a float64 ledger for exact insert, correct, retract and erase. This is lifelong knowledge editing inside the network, not learning from raw experience. The models are small (about 3.4M parameters), the world is invented and synthetic, and the frames are templated. Persistence and controllability of the memory hold by construction and are verified (10.3). Generality and specificity are the empirical questions.

1 QUESTION
One lived sentence mentions an entity in one formal, inflected case form and is written into fc2 of one block. How often does the new fact then appear:
(a) in the entity's other case forms;
(b) in the same forms written without the apostrophe (informal Turkish), where BPE may fuse root and suffix;
(c) after a left context;
(d) in other entities that share the name's last root token?
And does the presence of a query position whose key is identical to the written key explain (b)?
Unit of inference: one independently pretrained base model (a 'world'). Probe outcome: exact greedy production of the new object.

2 MODEL
- Built with CevahirNeuralNetwork via model_management.config_schema.tiny_model_config and ModelManager.initialize, the path used by today's working pilot.
- Overrides: vocab_size 4096; embed_dim = seq_proj_dim = 192; num_heads 4; num_kv_heads 4; num_layers 6; ffn_dim None (SwiGLU, resolves to 512); RMSNorm pre-norm; RoPE; tied embeddings; dropout 0.0 (this neutralises the double embedding dropout); gradient checkpointing off; max_seq_length = max_cache_len = 64; fp32.
- Repo default initialisation. The residual-init 'bug' is unconfirmed per the lead's notes; it is recorded, not changed.
- Size: about 3.44M parameters, about 2.65M non-embedding.
- No book-bound file is modified. The memory is a forward hook on layers[l].ffn.fc2 that adds h·Mᵀ, and the KV cache is cleared after every write.
- Parameters are grouped by name, never by isinstance (dual-import hazard).
- Fallback, fixed before discovery: if Phase-0 single-thread training throughput is below 1,500 tok/s, use num_layers 4 (about 2.56M; writable layers 0–2).

3 TOKENIZER
Primary tokenizer:
- The repo's tokenizer_management/byte_bpe.ByteLevelBPE at aa307b2, unmodified. An apostrophe plus the following letters form one pre-token, so formal Ayten'den always has a token boundary at the root end.
- 4,096 entries, trained per world on the prose training split only. Invented names are therefore unseen by the tokenizer and split into subwords, as new entities do in deployment.

Classification of informal forms (computed from token character offsets, before any training run). Each (name, inflected case) written without the apostrophe is one of:
- IDENT: the tokens up to the root end are identical to the formal root tokens, so the written key exists in the query by causal masking;
- RESEG: the boundary falls at the root end, but the root is segmented differently;
- FUSED: one token spans the root–suffix boundary.
NONIDENT = RESEG ∪ FUSED.

Judge's Phase-0 probe (scratch; 3 worlds × 400 names):
- names are 2–5 tokens;
- 82–86% of names share their last root token with another name;
- 0.0% of formal forms are fused;
- 37.8–38.5% of informal forms are fused.

Manipulation check: the mean NONIDENT share among informal target forms across confirmatory worlds must be at least 25%. If it is not, H1 and H2 can be 'supported' or 'inconclusive' but not 'refuted'.

Exploratory tokenizer F_freq (10.4):
- A study-local subclass whose pre-token pattern lets a letter run continue through an apostrophe. It has its own identity, save and load.
- Trained on prose plus the formal sentences of a designated 25% 'tokenizer-frequent' entity subset, repeated 30×.
- Phase-0 evidence: an apostrophe-agnostic BPE trained on natural prose fuses 0 of 7,200 formal forms of unseen names, and learns cross-apostrophe merges only as whole-word tokens of frequent proper nouns: 12 of 3,836 merges at 4k, 102 of 16,124 at 16k. With entity-frequent text it fused 19.5–24.5% of formal forms.

4 WORLD (TR-Paradigma generator; synthetic and license-clean)
Names, per world seed w:
- 400 invented one-word names: 2–3 syllables, Turkish phonotactics, no vowel hiatus, Turkish-aware capitalisation (i→İ, ı→I), no collision with prose words or with objects.
- 50% of names end in one of 16 shared final syllables: -ten, -gül, -nur, -han, -can, -tay, -su, -er, -ay, -un, -ün, -ra, -na, -li, -ce, -se.
- 20 prefix-extension pairs, where name2 = name1 + one syllable (exploratory).

Relations (objects assigned uniformly):
- A: memleket — 48 invented towns, a syllable plus köy, pınar, dere, kent, ova, yurt, tepe, su, hisar, bük, alan or yazı;
- B: meslek — 30 occupations;
- C: en sevdiği yemek — 30 dishes.

Frames: seven subject-initial frames. The relation NP comes before the object and the object is last. The carriers keep the entity as possessor or topic.
- NOM «{S} hakkında bilinen: {NP} {O}.»
- GEN «{S}'{X} {NP} {O}.»
- DAT «{S}'{X} dair bilinen: {NP} {O}.»
- ACC «{S}'{X} tanıyan herkes bilir: {NP} {O}.»
- LOC «{S}'{X} duran kimlik kaydına göre {NP} {O}.»
- ABL «{S}'{X} bahsedenler bilir: {NP} {O}.»
- INS «{S}'{X} ilgili bilgi: {NP} {O}.»
with NP ∈ {memleketi, mesleği, en sevdiği yemek}.

Morphology: two- and four-way vowel harmony; buffer n/y; d/t after voiceless consonants; the apostrophe; no softening for proper nouns. The informal form is the same string without the apostrophe.

Left contexts: none; P1 «Bilindiği gibi, »; P2 «Dün öğrendik ki ».

Owner review: the owner reviews the carriers and a 120-form gold table asynchronously; this does not block the work. A carrier judged ungrammatical before the freeze is replaced. After the freeze, it is reported as a limitation; frame effects are reported with ROOT as the construction control.

Entity partition:
- 60 P3 entities: 2 random frames of relation A are never trained; used only for gate G4.
- 60 C0-pool entities: used only for C0.
- 280 probe-pool entities: edits and all locality probes.

5 PRETRAINING
- Each pass contains every (entity, relation, frame) cell once, except P3-held cells. Each pass re-draws the context (none / P1 / P2 = 0.50 / 0.25 / 0.25) and, for inflected frames, informal orthography with p = 0.25.
- Prose chunks (at most 60 tokens, no false EOS) make up 20% of tokens.
- Prose: docs/book/tr chapters 1–9 with code blocks stripped, plus education/ without the personal letters (mektuplar*.docx) and the health-formulation file. The owner may veto education/; the fallback is book prose only. Held-out text for bits per byte (bpb): docs/book/tr chapters 10–12 prose.
- No corpus sample is printed and no checkpoint is published.
- Each sentence is one sequence, [BOS] text [EOS], right-padded, with PAD ignored.
- Optimiser: AdamW lr 2e-3, betas (0.9, 0.98), weight decay 0.01, 100 warmup steps, cosine decay to 0.1×, batch 64 sequences, gradient clip 1.0, OMP/MKL threads 1, deterministic seeds.
- Passes: K is the smallest of {20, 25, 30, 35, 40} at which both discovery bases pass G3–G7.
- An exposure ledger is kept per (fact, frame, context, orthography), and a string-search contamination test is run (V10).

6 GATES (per base, before any edit; true greedy decoding of at most 8 tokens)
- G1: 100% round-trip on all world strings and on held-out prose.
- G3: at least 95% exact on 600 seen formal no-context cells.
- G4: at least 80% on the 120 P3-held cells.
- G5: at least 80% on 600 informal cells, reported separately for IDENT and NONIDENT.
- G6: at least 80% on 600 prefixed cells.
- G7 (editable set): at least 120 probe-pool entities correct on relation A in all 7 formal frames and in at least 3 of the 6 informal inflected frames, with no context.
- A confirmatory base that fails a gate is replaced by the next reserve world (1008, then 1009), and the replacement is reported. If reserves are exhausted, the affected hypotheses are 'inconclusive (precondition)'.
- Fallback ladder, discovery only: (1) K up to 40; (2) 300 names; (3) prose share 10%; (4) embed_dim 256. The first rung that passes on both discovery worlds is frozen.

7 EDIT ITEMS
- Per base: 70 edits on distinct editable entities, relation A, 10 per lived case (NOM to INS). The new object is drawn uniformly from towns other than the old one. The lived sentence is the formal, sentence-initial frame of the lived case.
- Hypotheses use the 60 inflected-lived edits. NOM edits feed V3 and the descriptive tables.
- Target cases for H1, H2 and S1 in an edit: the inflected cases other than the lived case where the base model produces the old object in both orthographies. At least 2 are required; otherwise the edit is excluded from these hypotheses and counted.
- Each edit is applied alone to the base and erased before the next.
- Edit lists derive deterministically from seeds, and their hashes are frozen.

8 ARMS
All arms use the same items. Each arm's layer and step size are chosen on the discovery worlds by one rule.
- STD (primary; the ROME/MEMIT standard): the key is the mean fc2 input at the last token overlapping the lemma span (the bare name), over the lived sentence and 10 fixed prose-prefixed copies. v* is optimised over the lived sentence plus 4 of those copies. In formal text this position is the root-final token.
- ROOT: the key at the last token lying wholly inside the name, from the lived sentence only, with no averaging.
- LAST (a surface-span pipeline, labelled as such): the key at the last token of the whitespace word containing the name, which is the suffix token in formal text.
- PARA-ORTH (variant-averaged keys; uses the morphology generator): the STD key, additionally averaged over the name's 7 formal and 6 informal inflected forms, leaving out each tested variant, so a variant's own key never enters the write tested on it. It uses the same v* as STD.
- FT-L: gradient fine-tuning of the same fc2 matrix (layer chosen on discovery), with an ℓ∞ clamp ε on ΔW.
- FT-ALL: fine-tuning of all parameters, a dynamic-evaluation-style living learner.
- NORM (preprocessing applied to STD's informal probes): restore the apostrophe before a valid case allomorph for names in the entity lexicon.
- BASE: no write.

Write and search details:
- Single-edit write rule for key arms (ROME closed form): M = (v* − W k̄)(C0⁻¹k̄)ᵀ / (k̄ᵀC0⁻¹k̄).
- v* search: Adam from δ = 0, step size ∈ {0.5, 2.0}, at most 120 steps, early stop when the mean target NLL is below 0.05. Objective: NLL(new object | target prompts) + 1.0 · KL(base ‖ edited) on relation B's prompt in the lived frame. Shared absolute cap: ‖δ‖ ≤ 8 × the median ‖fc2 output‖ at that layer. δ is substituted at the key position during search. Threads and tensor shapes are fixed, and results are memoised by a content hash of (base hash, layer, prompts, key rule).
- C0 (per base and layer, float64): mean h hᵀ over about 20k positions — 50% C0-pool sentences across all frames, contexts and orthographies, 50% prose — plus 1e-4 · (tr C0 / 512) · I.
- FT-L: Adam lr ∈ {1e-3, 1e-2}, at most 60 steps, ε ∈ {0.005, 0.02}.
- FT-ALL: Adam lr ∈ {1e-4, 3e-4, 1e-3}, at most 40 steps.
- FT-L and FT-ALL use the same objective as v* and stop early when NLL is below 0.05.

Selection rule (discovery worlds 0–1 only; per arm; layers 0–4 × grid; 20 discovery edits):
- Maximise mean write efficacy (WE) subject to Leak_RD ≤ 1% and Leak_C ≤ 5%. Break ties by the shallower layer, then the smaller step.
- If no configuration meets the constraints, take the maximum WE (ties to the shallower layer) and label the arm 'not installable under constraints'.

Map-only switch: if STD's best discovery WE is below 50%, H1–H3 are declared not testable, and the confirmatory budget instead runs the binding-site map (10.5) on all 8 worlds.

9 PROBES AND METRICS
Scoring:
- Exact greedy production is checked with a batched exact check: greedy decoding reproduces a candidate if and only if the argmax matches at every continuation position and is followed by '.'.
- The check is validated against true greedy decoding on a random 5% of probes (V8).

Probe families per edit:
- WF: the lived frame. An edit counts as installed if the new object is produced here.
- PF6: the 6 other formal case frames.
- PF5 and IF5: the target cases (see 7) in formal and in informal orthography; each informal form is tagged IDENT or NONIDENT.
- CF14: the 7 formal frames × {P1, P2}.
- RN: up to 8 probe-pool entities sharing the edited name's last root token, in all 7 formal frames.
- RD: 10 random probe-pool entities not in RN, in all 7 formal frames.
- SS: 20 entities queried in the lived frame with the same suffix token.
- RB and RC: the edited entity on relations B and C, in all 7 formal frames.
- PX: the prefix-extension partner (exploratory).
- OLD: residue of the old object.
- bpb on held-out prose, for 20 edits per arm.
- τ(k', k̄) = k'ᵀC0⁻¹k̄ / k̄ᵀC0⁻¹k̄, at the arm's key position and as the maximum over all prompt positions.

Metrics:
- WE.
- PE6, PE5, IE5, CE14: mean over all qualifying edits of the share of the family that produces the new object. These are unconditional and primary.
- T-versions of the same quantities over installed edits only, used for decomposition.
- Leak_X: the changed-answer rate against the base model's own greedy answer, which is computed once by true greedy decoding.
- New-object adoption rate (secondary).

10 HYPOTHESES

10.1 Primary family. Holm correction, m = 3, one-sided α = 0.05. Per-world effects d_w are computed on confirmatory worlds 1000–1007 (n = 8).
- H1 — subjects without apostrophes. d_w = PE5(STD) − IE5(STD), on the same edits and target cases. Smallest effect of interest (SESOI) m1 = 15 pp. Predicted: at least 25 pp.
- H2 — an identical-key position is necessary; fusion or re-segmentation of root and suffix is the cause. For each installed edit with at least 1 IDENT and at least 1 NONIDENT target case, g_e = mean over NONIDENT cases of (F_c − I_c) minus mean over IDENT cases of (F_c − I_c). F_c and I_c ∈ {0, 1} record whether case c's formal and informal probes produce the new object. d_w = mean of g_e. m2 = 30 pp. Predicted: at least 50 pp.
- H3 — specificity cost of the paradigm-invariant root site. d_w = Leak_RN(STD) − Leak_RD(STD), over installed edits with |RN| ≥ 1. m3 = 5 pp. Predicted: at least 10 pp.

10.2 Secondary family. Holm correction, m = 4, one-sided α = 0.05; same decision rule; n = 8.
- S1 — variant-averaged keys extrapolate to unseen variants (the claim of frontier gap G2). d_w = IE5(PARA-ORTH, leave-one-variant-out) − IE5(STD). m = 10 pp. Non-inferiority is also required: the upper 95% bound of the Leak_RD difference must be at most 1 pp, and that of the Leak_RN difference at most 5 pp; otherwise the result is 'not supported (locality cost)'. The τ-account predicts failure on NONIDENT targets; both outcomes are reported.
- S2 — prefix averaging buys robustness to left context (a replication of the ROME/CoRE rationale). d_w = CE14(STD) − CE14(ROOT). m = 10 pp.
- S3 — joint versus independent state in a lifelong stream (see 13). Knowledge in Superposition and MEMIT-Merge predict this, so it is not a novelty claim. d_w = E_end(J) − E_end(I). m = 10 pp. In addition, the pooled within-world Spearman ρ between a fact's maximal τ to other live keys and its failure under I must be greater than 0 (within-world permutation, p ≤ 0.05).
- S4 — entity identity is assembled at the root-final token as depth grows. d_w = median τ_RN at layer 0 − median τ_RN at layer 4, measured at the root-final token of sentence-initial NOM prefixes with layer-specific C0. m = 0.15.

10.3 Verification tests. These are pass/fail checks run before any confirmatory run, not hypotheses. A failure blocks publication of the H and S results until it is fixed.
- V1: with zero memory, max|Δlogit| = 0.0; detaching the hook restores 0.0.
- V2: fc2 inputs at the root-final token are identical across the 7 formal forms and the IDENT informal forms on truncated prefixes (exactly 0.0, at every layer), and within relative 1e-5 on full padded queries.
- V3: for NOM-lived edits, the ROOT, LAST and unaveraged STD positions coincide, and the ROOT and LAST writes are bitwise identical.
- V4: float64 ledger. A stream of 100 inserts, 30 corrections and 20 retractions equals the direct solve on the final belief set (relative Frobenius ≤ 1e-10); 5 different orders agree to ≤ 1e-10; erase gives M = 0 exactly; merge equals the solve on the union to ≤ 1e-10. Recomputing (k, r) in a fresh process with another thread count leaves a residue that is reported but not gated.
- V5: CEV-08-style counterexample. The histories {f1} and {f1, f2}, with r2 := ΔW({f1})·k2, give the same W (≤ 1e-12) but need different weights after retract(f1). A state holding only W cannot satisfy both; per-fact records can.
- V6: cached and uncached greedy outputs are identical after writes.
- V7: a repeated v* search gives max|Δ| = 0.0, and 8 training steps are reproducible at fixed threads.
- V8: the exact check agrees with true greedy decoding on 5% of probes, with disagreement at most 1%; otherwise that probe family uses true greedy decoding.
- V9: the morphology matches the 120-form gold table and an independent implementation on every generated form.
- V10: no edit target (entity plus new object, relation A) and no P3-held cell occurs in the training text.
- V11: tokenizer round-trip is 100%, and the tokenizer identity matches the unmodified repo class.
- V12: the IDENT / RESEG / FUSED classifier reproduces hand-labelled cases.

10.4 Exploratory analyses (descriptive only; no confirmatory claim)
- F_freq on worlds 1000–1002 (3 extra bases): PE6 of STD for tokenizer-frequent fused entities versus tokenizer-rare entities in the same bases. Predicted: at least 30 pp lower for the frequent ones.
- LAST versus ROOT.
- FT-L and FT-ALL: generality and leakage.
- NORM: recovery on informal probes.
- Informal-lived writes queried formally.
- SS and PX leakage.
- τ-AUC within each rule.
- Causal-tracing peak versus editability peak.
- Tables per frame, per token count and per cluster size.
- bpb drift.
- Segmentation of the same forms by a GPT-2-style regex.

10.5 Binding-site map (primary descriptive output; always reported)
- Worlds: discovery 0–1 and confirmatory 1000–1001.
- Sites ROOT, LAST and PRED (the last prompt token) × layers 0–4 × 30 edits, reporting WE, PE6, IE5, Leak_RN and Leak_RD.
- Causal tracing: Gaussian noise of 3σ on the name-token embeddings, then restoring the residual stream at each (layer, position).
- τ by layer on all bases.

11 STATISTICS AND DECISIONS
- Per-world estimates are computed from per-edit records.
- A world contributes to a hypothesis only if it has at least 15 qualifying edits. If fewer than 7 worlds qualify, the hypothesis is 'inconclusive (insufficient data)'.
- Test: an exact one-sided sign-flip permutation over all 2ⁿ sign patterns of (d_w − m), with H0: mean(d − m) ≤ 0. Holm step-down is applied within each family.
- Supported: Holm rejection.
- Refuted: the upper bound of the two-sided 95% t-interval of mean d_w is below m. This includes an effect of the wrong sign.
- Otherwise: inconclusive.
- Descriptive outputs: per-world values, t-intervals, and a hierarchical bootstrap (worlds, then edits; 10,000 resamples).
- Verification tests are pass/fail.
- analyze.py is frozen by hash before any confirmatory hypothesis metric is computed.

12 SEEDS
- Discovery worlds {0, 1}; confirmatory worlds {1000, …, 1007}; reserves {1008, 1009}; F_freq worlds {1000, 1001, 1002}.
- For world w: generator w, model initialisation w, data order w+1, edit list w+2, probe sampling w+3, prose prefixes w+4, life order w+5.
- All runs are single-threaded, and v* starts from δ = 0.

13 LIVES (S3)
- Run on all 8 confirmatory worlds, with the STD key at its selected layer.
- Ledger λ ∈ {0.01, 0.1, 1} is chosen on discovery lives, maximising E_end subject to Leak_RD ≤ 1%.
- One life is 100 operations on 60 probe-pool entities, in a seeded random order:
  - 60 first writes (relation A, lived cases balanced);
  - 15 second-relation writes (relation C) for entities already written;
  - 15 corrections of A;
  - 10 retractions of A.
- The same stream is applied to three state variants:
  - J: float64 ledger, W = W0 + B(λC0 + A)⁻¹. The entity-joint v* is re-optimised against W0 whenever an entity's believed set changes (MEMIT-Merge style).
  - I: independent per-fact ROME deltas against W0. Correct = replace the delta; retract = subtract it.
  - Q: sequential ROME on the current weights, with v* computed against the current model; retract = write back W0·k. Run in 4 random orders; the order spread is descriptive.
- E_end is the mean exact production of believed objects over the 7 formal frames. Also reported: retraction residue, correction residue, Leak_RN and Leak_RD.

14 STOPPING AND REPORTING
- Discovery and confirmatory runs are strictly separated.
- No hypothesis metric is inspected before all confirmatory runs finish; only gate metrics may be looked at.
- A crashed run is rerun from scratch with the same seeds.
- If compute runs short, priority is: (1) H1–H3; (2) S1–S2 (same runs); (3) S4 (forward passes only); (4) S3; (5) the confirmatory map worlds; (6) F_freq and the other exploratory items. Anything not run is reported as 'not run'.
- Every hypothesis is reported as supported, refuted or inconclusive, with per-world values. Negative results go in the abstract.
- Today's strategist, critic and judge pilots are reported separately as discovery evidence and never pooled with confirmatory results.
