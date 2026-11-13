"""``python -m cevahir_ai.keys.schemas --check | --write``: golden weight layouts."""
import argparse
import sys

from cevahir_ai.keys import schemas


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m cevahir_ai.keys.schemas", description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="compare the snapshots with the current code")
    mode.add_argument("--write", action="store_true", help="rewrite the snapshots from the current code")
    args = parser.parse_args(argv)
    if args.write:
        schemas.write()
        print(f"wrote {len(schemas.PRESETS)} layout snapshots to {schemas.LAYOUT_DIR}")
        return 0
    findings = schemas.check()
    for finding in findings:
        print(finding, file=sys.stderr)
    if findings:
        print("weight layouts changed: revert, or bump ARCHITECTURE_VERSION, add a migration and run --write",
              file=sys.stderr)
        return 5
    print(f"{len(schemas.PRESETS)} layout snapshots match the current code")
    return 0


if __name__ == "__main__":
    sys.exit(main())
