"""The interactive console: prompt, multi-line input, history and Tab completion.

Line editing comes from ``readline`` where Python has it (Linux, macOS, Colab terminals). On
Windows, where it does not, the small editor in ``lineedit_win`` supplies history and Tab; with
``CEVAHIR_LINE_EDITOR=plain``, or when input is not a terminal, plain ``input()`` is used.
"""
import os
import sys

from cevahir_ai.console.parser import is_incomplete

PROMPT, CONTINUATION = "PS cevahir> ", ">> "
HISTORY_LIMIT = 2000


def banner(session):
    from cevahir_ai import __version__
    from cevahir_ai.ops.governed import BOOTSTRAP_POLICY
    turkish = session.culture == "tr"
    try:
        home, host = str(session.store.home), session.store.host_id() or ("(henüz yok)" if turkish else "(none yet)")
    except Exception as error:
        home, host = f"({error})", "-"
    lines = [f"Cevahir AI {__version__}", f"  home    {home}", f"  host    {host}"]
    if turkish:
        lines += [f"  politika {BOOTSTRAP_POLICY}: okuma ve kayıt eylemlerine izin verilir ve kaydedilir; onay katmanı henüz yok",
                  "  yardım  Get-Help   |   komutlar  Get-Command   |   çıkış  exit"]
    else:
        lines += [f"  policy  {BOOTSTRAP_POLICY}: read and register actions are allowed and recorded; no approvals yet",
                  "  help    Get-Help   |   commands  Get-Command   |   leave  exit"]
    return "\n".join(lines) + "\n"


def history_path(session):
    try:
        return os.path.join(str(session.store.home), "console_history")
    except Exception:
        return None


def _readline_reader(session):
    """``input``-based reader with readline history and completion, or None when readline is missing."""
    try:
        import readline
    except ImportError:
        return None, lambda: None
    from cevahir_ai.console.completion import complete

    state = {"matches": []}

    def completer(text, index):
        if index == 0:
            try:
                line, end = readline.get_line_buffer(), readline.get_endidx()
                state["matches"] = [item.text for item in complete(line, end, session)]
            except Exception:
                state["matches"] = []
        return state["matches"][index] if index < len(state["matches"]) else None

    readline.set_completer(completer)
    readline.set_completer_delims(" \t\n|;(){},'\"")
    readline.parse_and_bind("bind ^I rl_complete" if "libedit" in (readline.__doc__ or "") else "tab: complete")
    path = history_path(session)
    if path and os.path.isfile(path):
        try:
            readline.read_history_file(path)
        except OSError:
            pass

    def save():
        target = history_path(session)
        if target and os.path.isdir(os.path.dirname(target)):
            try:
                readline.set_history_length(HISTORY_LIMIT)
                readline.write_history_file(target)
            except OSError:
                pass

    return input, save


def _windows_reader(session):
    try:
        import msvcrt
    except ImportError:
        return None, lambda: None
    from cevahir_ai.console.completion import complete
    from cevahir_ai.console.lineedit_win import LineEditor
    history, path = [], history_path(session)
    if path and os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as stream:
                history = [line.rstrip("\n") for line in stream][-HISTORY_LIMIT:]
        except OSError:
            history = []

    def write(text):
        sys.stdout.write(text)
        sys.stdout.flush()

    editor = LineEditor(msvcrt.getwch, write, lambda line, cursor: [item.text for item in complete(line, cursor, session)],
                        history)

    def save():
        target = history_path(session)
        if target and os.path.isdir(os.path.dirname(target)):
            try:
                with open(target, "w", encoding="utf-8", newline="\n") as stream:
                    stream.write("\n".join(editor.history[-HISTORY_LIMIT:]) + "\n")
            except OSError:
                pass

    return editor.read_line, save


def make_reader(session, stdin=None):
    """``(read(prompt) -> line, save_history())`` for this terminal."""
    stdin = stdin or sys.stdin
    interactive = hasattr(stdin, "isatty") and stdin.isatty()
    plain = session.environ.get("CEVAHIR_LINE_EDITOR", "").lower() == "plain"
    if interactive and not plain:
        reader, save = _readline_reader(session)
        if reader is None and os.name == "nt":
            reader, save = _windows_reader(session)
        if reader is not None:
            return reader, save
    return input, lambda: None


def run(session, read=None, save=None, show_banner=True):
    """The read-eval-print loop. Returns the exit code of the session."""
    if read is None:
        read, save = make_reader(session)
    if show_banner:
        session.write(banner(session))
    code = 0
    while True:
        buffer = ""
        try:
            while True:
                line = read(CONTINUATION if buffer else PROMPT)
                buffer = f"{buffer}\n{line}" if buffer else line
                if not buffer.strip() or not is_incomplete(buffer):
                    break
        except KeyboardInterrupt:
            session.write("\n")
            continue
        except EOFError:
            session.write("\n")
            break
        if not buffer.strip():
            continue
        if buffer.strip().lower() in ("exit", "quit"):
            break
        if session._store is not None:
            session._store.invalidate()          # another process may have registered something meanwhile
        session.codes.clear()
        session.exit_requested = False
        try:
            code = session.run(buffer)
        except Exception as error:                  # a defect in a command must not end the session
            session.write_error_stream(f"cevahir: internal error: {type(error).__name__}: {error}")
            code = 1
        if session.exit_requested:
            break
    if save:
        save()
    return code
