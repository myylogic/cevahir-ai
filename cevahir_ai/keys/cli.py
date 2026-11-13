"""``python -m cevahir_ai.keys``: the minimal key CLI.

  inspect <file>          is it intact, which tokenizer and architecture, compatible with the
                          current code (and a target), and if not, why
  verify <file> <key>     recompute one key of a file and compare
  explain <key | file>    what a key is, or the keys of a file
  diff <a> <b>            which tensors differ between two checkpoints
  schemes                 the scheme registry

Exit codes: 0 ok, 1 error, 2 usage, 5 verification failed (integrity, digest or compatibility).
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from cevahir_ai.domain.verdicts import PURPOSES
from cevahir_ai.errors import CvError

_MARK = {"pass": "ok  ", "fail": "FAIL", "unknown": "??  ", "unsupported": "UNSUP", "skipped": "-   "}
_YES_NO = {"yes": ("yes", "evet"), "no": ("NO", "HAYIR"), "unknown": ("unknown", "bilinmiyor")}


def _target(args):
    from cevahir_ai.adapters.inspection import Target
    target = Target(dtype=args.dtype, max_seq_length=args.max_seq_length,
                    allow_migrations=tuple(args.allow_migration or ()))
    if args.target:
        from cevahir_ai.keys import schemas
        if args.target in schemas.PRESETS:
            target.config = schemas.preset_config(args.target)
        else:
            target.config = json.loads(Path(args.target).read_text(encoding="utf-8"))
    if args.byte_bpe:
        from cevahir_ai._compat.repo import import_legacy
        target.tokenizer = import_legacy("tokenizer_management.byte_bpe").ByteLevelBPE.load(args.byte_bpe)
    elif args.vocab or args.merges:
        if not (args.vocab and args.merges):
            raise SystemExit("--vocab and --merges go together")
        from cevahir_ai._compat.repo import import_legacy
        core = import_legacy("tokenizer_management.core.tokenizer_core").TokenizerCore
        target.tokenizer = core({"vocab_path": args.vocab, "merges_path": args.merges, "read_only": True, "use_gpu": False})
    return target


def render(report):
    """Human-readable report (English labels with the Turkish answer next to the verdict)."""
    explain, keys, lines = report.explain, report.keys, []
    add = lines.append
    add(f"file        {report.path}  ({report.size:,} bytes)")
    add(f"  key       {keys.get('file')}")
    yes, evet = _YES_NO[report.intact]
    add(f"intact      {yes} / {evet}")
    for check_id in ("C1", "C3", "C12"):
        if check_id in report.facts:
            add(f"  {check_id:<4}{report.facts[check_id].name:<18} {report.facts[check_id].detail}")
    add(f"envelope    {report.envelope or ('not read at file level' if report.level == 'file' else 'unreadable')}")
    if report.level != "file":
        tokenizer = explain.get("tokenizer") or {}
        if keys.get("tok"):
            add(f"tokenizer   {keys['tok']}")
            if tokenizer.get("canonical") and tokenizer["canonical"] != str(keys["tok"]):
                add(f"  canonical {tokenizer['canonical']}" + (f"  (tracked: {tokenizer['tracked']})" if tokenizer.get("tracked") else ""))
        else:
            add("tokenizer   unknown (the file carries no tokenizer identity)")
        if keys.get("arch"):
            fields = explain["arch"]
            add(f"arch        {keys['arch']}")
            table = f"[{fields['pe_max_len']}]" if fields.get("pe_max_len") is not None else ""
            add(f"  fields    vocab={fields['vocab_size']} d={fields['embed_dim']} layers={fields['num_layers']} "
                f"heads={fields['num_heads']} kv_heads={fields['num_kv_heads']} ffn={fields['ffn_dim']} "
                f"pe={fields['pe_mode']}{table} tied={fields['tie_weights']} moe={fields['use_moe']}  [{explain['family_version']}]")
        elif explain.get("arch_undeclared"):
            add(f"arch        unknown ({explain['arch_undeclared']})")
        else:
            add("arch        unknown (the file declares no architecture)")
        if keys.get("weights"):
            weights = explain["weights"]
            add(f"weights     {keys['weights']}")
            add(f"  content   {weights['tensors']} tensors, {weights['parameters']:,} parameters, "
                f"dtype {', '.join(weights['dtypes']) or 'n/a'}")
            layout = explain.get("expected_layout") or {}
            add(f"  layout    {keys['wschema']}" + (f"  (preset {layout['preset']})" if layout.get("preset") else ""))
        for name in ("ckpt", "model"):
            if keys.get(name):
                add(f"{'checkpoint' if name == 'ckpt' else 'model':<11} {keys[name]}")
        for alias in report.aliases:
            add(f"  alias     {alias}")
        if explain.get("epoch") is not None:
            add(f"epoch       {explain['epoch']}" + (f"  metric {explain['metric']}" if explain.get("metric") is not None else ""))
    verdict = report.verdict
    add(f"verdict     {verdict.verdict}  (purpose: {verdict.purpose}, level: {report.level})")
    add(f"            {verdict.summary_tr}")
    for check in verdict.checks:
        flag = "*" if check.id in verdict.required else " "
        add(f"  {flag}{check.id:<4}{check.name:<18} {_MARK[check.status]:<5} {check.detail}")
    missing = [check_id for check_id in verdict.required if check_id not in report.facts]
    if missing:
        add(f"  not checked at level '{report.level}': {', '.join(missing)}")
    if verdict.failing:
        add("why not:")
        for check_id in verdict.failing:
            fact = report.facts.get(check_id)
            add(f"  - {check_id} {fact.name + ': ' + (fact.detail or fact.status) if fact else 'not checked'}")
    for note in report.notes:
        add(f"note        {note}")
    return "\n".join(lines)


def _cmd_inspect(args):
    from cevahir_ai.adapters.inspection import inspect_checkpoint
    report = inspect_checkpoint(args.file, level=args.level, purpose=args.purpose, target=_target(args),
                                expect_file=args.expect_file, expect_weights=args.expect_weights)
    if args.json:
        # ``default=repr``: a stored value that is no JSON type (a tensor in a metadata field) is shown, not fatal.
        print(json.dumps(report.to_dict(tensors=args.tensors), ensure_ascii=False, indent=1, sort_keys=True, default=repr))
    else:
        print(render(report))
    return 0 if report.verdict.ok else 5


def _keys_of_file(path, kinds=None):
    """Keys that can be derived from a path alone: file keys always; checkpoint keys for torch files."""
    from cevahir_ai.adapters.envelopes import sniff_pickle_globals
    from cevahir_ai.keys.schemes.file import file_bytes_key, file_key
    path = Path(path)
    keys = {"file:bytes-v1": file_bytes_key(path)}
    try:
        keys["file:utf8lf-v1"] = file_key(path, "utf8lf-v1")
    except UnicodeDecodeError:
        pass
    if sniff_pickle_globals(path)["container"] == "torch-zip":
        from cevahir_ai.adapters.inspection import inspect_checkpoint
        report = inspect_checkpoint(path, level="weights", purpose="inspect")
        for key in list(report.keys.values()) + list(report.aliases):
            keys.setdefault(f"{key.kind}:{key.scheme}", key)
    elif path.suffix == ".json":
        try:
            from cevahir_ai.keys.schemes.tok import tok_bytebpe_key
            keys["tok:bytebpe-v1"] = tok_bytebpe_key(path)
        except (ValueError, KeyError):
            pass
    return keys


def _cmd_verify(args):
    from cevahir_ai.keys.grammar import Key
    expected = Key.parse(args.key)
    found = _keys_of_file(args.file).get(f"{expected.kind}:{expected.scheme}")
    if found is None:
        print(f"cannot derive a {expected.kind}:{expected.scheme} key from {args.file}", file=sys.stderr)
        return 1
    ok = found == expected
    print(f"{'ok' if ok else 'MISMATCH'}  {found}")
    if not ok:
        print(f"expected  {expected}")
    return 0 if ok else 5


def _cmd_explain(args):
    from cevahir_ai.keys.grammar import Key, is_key
    from cevahir_ai.keys.registry import SCHEMES
    if is_key(args.what):
        key = Key.parse(args.what)
        scheme = SCHEMES[(key.kind, key.scheme)]
        print(f"{key}\n  short    {key.display()}\n  kind     {key.kind}\n  scheme   {key.scheme} ({scheme.status})\n"
              f"  meaning  {scheme.summary}\n  newline  {scheme.newline}\n  torch    {'needed' if scheme.torch else 'not needed'}\n"
              f"  derive   {scheme.derive}")
        if (key.kind, key.scheme) == ("tok", "legacy-v3"):
            from cevahir_ai.adapters.tokenizers import lookup_legacy
            tracked = lookup_legacy(key.digest)
            print(f"  alias of {tracked['key']}  (tracked tokenizer '{tracked['name']}', {tracked['how']})" if tracked
                  else "  alias of nothing on record: not a tracked tokenizer")
        return 0
    for name, key in sorted(_keys_of_file(args.what).items()):
        print(f"{name:<24} {key}")
    return 0


def _cmd_diff(args):
    from cevahir_ai.adapters.inspection import inspect_checkpoint
    from cevahir_ai.keys.torchkeys.weights import diff_tables
    left, right = (inspect_checkpoint(path, level="weights", purpose="inspect") for path in (args.a, args.b))
    if left.table is None or right.table is None:
        print("one of the files holds no readable weights", file=sys.stderr)
        return 1
    names = sorted(set(left.keys) | set(right.keys))
    same = True
    for name in names:
        a, b = left.keys.get(name), right.keys.get(name)
        if name == "file":
            continue
        equal = a == b
        same = same and equal
        print(f"{'=' if equal else '!'} {name:<10} {a.display() if a else '-':<34} {b.display() if b else '-'}")
    delta = diff_tables(left.table, right.table)
    for label, rows in (("only in a", delta["only_left"]), ("only in b", delta["only_right"])):
        for name in rows:
            print(f"  {label}: {name}")
    for name, reason in delta["changed"]:
        print(f"  changed: {name} ({reason})")
    if args.json:
        print(json.dumps(delta, ensure_ascii=False, indent=1))
    return 0 if same else 5


def _cmd_schemes(args):
    from cevahir_ai.keys.registry import SCHEMES
    for scheme in SCHEMES.values():
        print(f"{scheme.name:<24} {scheme.status:<10} {'torch' if scheme.torch else '-':<5} {scheme.newline:<5} {scheme.summary}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m cevahir_ai.keys", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="inspect a checkpoint file")
    inspect.add_argument("file")
    inspect.add_argument("--level", choices=("file", "weights", "compat"), default="compat",
                         help="file: torch-free; weights: read tensors; compat: also compare with the current code (default)")
    inspect.add_argument("--purpose", choices=PURPOSES, default="evaluate")
    inspect.add_argument("--json", action="store_true")
    inspect.add_argument("--tensors", action="store_true", help="with --json: include the per-tensor table")
    inspect.add_argument("--target", help="target architecture: a preset (tiny, train-default, cev11-primary) or a JSON config file")
    inspect.add_argument("--vocab", help="target legacy tokenizer: vocab.json")
    inspect.add_argument("--merges", help="target legacy tokenizer: merges.txt")
    inspect.add_argument("--byte-bpe", help="target byte-level BPE tokenizer file")
    inspect.add_argument("--dtype", default="float32", help="target floating dtype (default float32)")
    inspect.add_argument("--max-seq-length", type=int, help="serving context length")
    inspect.add_argument("--expect-file", help="expected file key or sha256")
    inspect.add_argument("--expect-weights", help="expected weights:sha256-v1 key")
    inspect.add_argument("--allow-migration", action="append", help="id of a migration that may be applied in memory")
    inspect.set_defaults(run=_cmd_inspect)
    verify = commands.add_parser("verify", help="recompute a key of a file and compare")
    verify.add_argument("file")
    verify.add_argument("key")
    verify.set_defaults(run=_cmd_verify)
    explain = commands.add_parser("explain", help="explain a key, or list the keys of a file")
    explain.add_argument("what")
    explain.set_defaults(run=_cmd_explain)
    diff = commands.add_parser("diff", help="compare two checkpoints tensor by tensor")
    diff.add_argument("a")
    diff.add_argument("b")
    diff.add_argument("--json", action="store_true")
    diff.set_defaults(run=_cmd_diff)
    schemes = commands.add_parser("schemes", help="list the key schemes")
    schemes.set_defaults(run=_cmd_schemes)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):      # Windows consoles default to a legacy code page
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError, OSError):
            pass
    logging.disable(logging.WARNING)      # legacy modules log at import; a CLI answer must stay readable
    try:
        return args.run(args)
    except CvError as error:
        print(f"error [{error.category}]: {error}", file=sys.stderr)
        return error.exit_code
    except (FileNotFoundError, KeyError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        logging.disable(logging.NOTSET)
