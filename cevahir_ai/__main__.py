"""``python -m cevahir_ai`` and the ``cevahir`` command: the console, and a passthrough to the key CLI.

    cevahir                         the interactive console (see ``cevahir -Help``)
    cevahir -Command "..."          one line of the console language
    cevahir -File script.ps1        a script
    cevahir keys <inspect|...>      the key CLI of ``python -m cevahir_ai.keys``
    cevahir --version
"""
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["keys"]:
        from cevahir_ai.keys.cli import main as keys_main
        return keys_main(argv[1:])
    from cevahir_ai.console.main import main as console_main
    return console_main(argv)


if __name__ == "__main__":
    sys.exit(main())
