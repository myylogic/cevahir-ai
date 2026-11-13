"""``python -m cevahir_ai``: version and a passthrough to the key CLI (the console arrives later)."""
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["keys"]:
        from cevahir_ai.keys.cli import main as keys_main
        return keys_main(argv[1:])
    if argv in (["--version"], ["-V"]):
        from cevahir_ai import __version__
        print(__version__)
        return 0
    print("usage: python -m cevahir_ai --version | keys <inspect|verify|explain|diff|schemes> ...", file=sys.stderr)
    return 0 if argv in (["-h"], ["--help"]) else 2


if __name__ == "__main__":
    sys.exit(main())
