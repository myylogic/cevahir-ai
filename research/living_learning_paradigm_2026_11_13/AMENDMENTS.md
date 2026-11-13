# CEV-11 — post-freeze record

This file records everything that happened to the study after the protocol was frozen (PROTOCOL.md,
freeze_manifest.sha256). No frozen file has been changed. Entries are appended in time order; "recorded output"
entries are not amendments.

- 2026-11-13 — addition (not an amendment of any frozen file): `verify_real.py`, a verification tool that repeats
  the checks V1, V2, V6 and V8 of PROTOCOL §10.3 on the study's own bases and real STD writes (the unit tests in
  tests/evolution/ cover them on tiny models only; V8 needs real probes). It imports the frozen modules and writes
  only `results/verification/real_<world>.json`. It runs on discovery worlds 0 and 1 and on confirmatory worlds
  1000 and 1001 with STD's selected layer and configuration, after the selection is recorded and before any
  confirmatory edit or life run; every confirmatory run waits for these checks and for the frozen unit tests
  (`results/verification/unit_tests.txt`). verify.py --check lists this file and AMENDMENTS.md as files that are in
  scope but not in the freeze manifest.
- 2026-11-13 — orchestration (no study code involved): the discovery selection grid was split across processes to
  use all four CPU cores. Each main selection process ran STD and then PARA-ORTH (which needs STD's in-memory v*
  results) and was stopped right after PARA-ORTH; ROOT, LAST and PRED/FT-L/FT-ALL then ran as separate
  `run_select.py --arms ...` processes, which resume from complete records. select_decide.py rejects incomplete or
  duplicated records, so the split cannot change the selection.
- 2026-11-13 09:58 UTC — incident (no study code or record changed): a descriptive look with
  `select_decide.py --allow-partial` while ROOT, LAST, PRED, FT-L and FT-ALL had no selection records yet WROTE
  results/selection.json and results/tables/discovery_map.csv instead of only printing. Cause: the completeness
  check compares the (layer, configuration) rows of the arms that have records and does not notice arms with no
  records at all. Both files were deleted immediately; no job had read them (the discovery lives and every
  confirmatory run depend on the scheduled select_decide.py runs that start only after all selection jobs have
  finished successfully, and run_confirm.py refuses a selection that lacks an arm). The recorded selection is the
  one written after the discovery lives, whose hash is appended below. The gap in the frozen script is reported
  here and left unchanged. This qualifies the orchestration entry above: the split cannot change the selection
  because the scheduled select_decide.py run starts only after all six selection jobs have exited successfully.
- 2026-11-13 11:00 UTC — verify_real.py revised before its first scheduled run, after an independent review: V8 now
  samples per probe family from the same family-level batched exact check that scoring uses (including OLD on PF6
  and RN adoption; leak probes scored by true greedy decoding are skipped and counted), pools the unedited and the
  edited state per family, records every disagreement, lists families with any disagreement as
  `true_greedy_families` (the protocol's remedy), and passes at an overall disagreement of at most 1% (per family
  only where at least 100 probes were sampled). V1 now also checks the logits after a real write was zeroed and
  after the installer's restore. V2's padded check compares each query with its own truncated key. The script exits
  non-zero only when V1 or V2 fails; V6 and V8 failures are recorded and, per PROTOCOL §10.3, block publication of
  the H and S results until resolved rather than stopping the runs. V8 here is a pre-run sample on STD writes in
  four worlds, not a check of every scored probe; the report says so.

- 2026-11-13 11:38 UTC — recorded output (not an amendment): results/selection.json sha256 83cb20052a3a406c0c79a03a16c8a4006d76113dd9a658ca65fceb85e6cba5e8, produced by the frozen select_decide.py from discovery worlds 0-1 before any confirmatory edit run.
- 2026-11-13 12:20 UTC — diagnosis recorded before any confirmatory life started (discovery evidence only; no
  frozen file or record changed): the discovery J lives collapse for every λ in the preregistered grid (E_end
  0.025-0.057, Leak_RD 63-70 of 70). An independent code audit found no implementation defect: the frozen Ledger
  rebuilt from the final belief sets reproduces the recorded lives exactly. Mechanism: (1) C0 is a per-position mean,
  so λ ≤ 1 lies two to four decades below the eigenvalues of KᵀC0⁻¹K (35.5-4362 on world 0); the grid is inert and
  the joint solve is the exact-interpolation limit (M(0.01) and M(1) differ by about 1.3%); MEMIT works at about 1e4.
  (2) The collapse itself comes from overloading one fc2 layer: about 52 writes whose residuals sit at the cap (step 5,
  cap 8 × median fc2 norm) raise the RMS of ΔW·h over C0 positions to about 2× the native fc2 output, versus 0.2× for
  one edit; a real I state on world 0 collapses too (E_end 0.065, RD 70/70). No λ from 1e-2 to 1e5 and no
  preregistered step gives Leak_RD ≤ 1% with a useful E_end. Consequence: the confirmatory S3 test stays as frozen
  (PROTOCOL §13) and is predicted to be 'refuted' because all three variants are overloaded; it is therefore not a
  test of joint versus independent state under capacity, and the report will say so. Per PROTOCOL §14 the lives run
  last (after H1-H3, S1-S2 and S4); any life run with a rescaled λ or a smaller step is exploratory. Diagnosis
  scripts and numbers: kept with the session's working files (code audit, numerical diagnosis, skeptic verdict).
