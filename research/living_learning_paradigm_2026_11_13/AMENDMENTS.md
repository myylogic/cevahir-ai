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
