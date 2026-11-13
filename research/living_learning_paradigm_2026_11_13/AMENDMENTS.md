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
