"""``cevahir``: the console entry point.

    cevahir                              interactive console (or a script on standard input)
    cevahir -Command "Get-Artifact | Format-Table"     one line; also -c
    cevahir -File runbook.ps1 -Name prod               a script file with arguments for its param( )
    cevahir -Invoke < request.json                     one cmdlet call as JSON (the PowerShell module uses it)

    -OutputFormat Text|Json    Json: the objects left on the pipeline, as one JSON array on stdout
    -Culture tr|en             language of help and of the banner
    -Home <dir>                another CEVAHIR_HOME for this run
    -Width <n>                 table width
    -NoProfile                 do not run <CEVAHIR_HOME>/profile.ps1
    -NoLogo                    no banner
    -Version, -Help

Exit codes: 0 ok, 1 error, 2 parse or usage error, 3 policy denied, 4 approval required (reserved),
5 verification failed, 130 interrupted. Starting the console imports no torch; a cmdlet that needs
torch imports it on first use.
"""
import json
import os
import sys

USAGE = __doc__.split("\n\n", 1)[1].split("Exit codes")[0].rstrip()
_FLAGS = {"-command": "command", "-c": "command", "-file": "file", "-f": "file", "-outputformat": "output",
          "-o": "output", "-culture": "culture", "-home": "home", "-width": "width"}
_SWITCHES = {"-noprofile": "noprofile", "-nologo": "nologo", "-invoke": "invoke", "-version": "version",
             "--version": "version", "-v": "version", "-help": "help", "--help": "help", "-h": "help", "-?": "help"}


class Usage(Exception):
    pass


def parse_arguments(argv):
    options = {"command": None, "file": None, "arguments": [], "output": "text", "culture": None, "home": None,
               "width": None, "noprofile": False, "nologo": False, "invoke": False, "version": False, "help": False}
    index = 0
    while index < len(argv):
        token = argv[index]
        lowered = token.lower()
        index += 1
        if lowered in _SWITCHES:
            options[_SWITCHES[lowered]] = True
        elif lowered in _FLAGS:
            name = _FLAGS[lowered]
            if index >= len(argv):
                raise Usage(f"{token} needs a value")
            if name == "command":
                options["command"] = " ".join(argv[index:])
                break
            options[name] = argv[index]
            index += 1
            if name == "file":
                options["arguments"] = list(argv[index:])
                break
        else:
            raise Usage(f"unknown argument: {token}")
    if options["output"].lower() not in ("text", "json"):
        raise Usage("-OutputFormat is Text or Json")
    options["output"] = options["output"].lower()
    if options["culture"] is not None and options["culture"].lower() not in ("tr", "en"):
        raise Usage("-Culture is tr or en")
    if options["width"] is not None:
        if not options["width"].isdigit():
            raise Usage("-Width is a number")
        options["width"] = int(options["width"])
    return options


def configure_streams():
    """Never let an unencodable character end the session: a legacy code page (cp1254) replaces it;
    a pipe is written as UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream.isatty():
                stream.reconfigure(errors="replace")
            else:
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def invoke(session, request):
    """Run one cmdlet from a JSON request ``{"command", "parameters", "input"}``. Nothing in the request
    is parsed as script text: parameter values are data. Returns the JSON-ready result."""
    from cevahir_ai.console import binder, registry
    from cevahir_ai.console.commands.core import from_json
    from cevahir_ai.console.errors import CommandNotFound
    from cevahir_ai.console.runtime import EXPECTED, Context
    from cevahir_ai.domain.records import plain
    records, name = [], request.get("command") if isinstance(request, dict) else None
    try:
        if not isinstance(request, dict) or not isinstance(request.get("command"), str):
            raise ValueError('the request must be {"command": "...", "parameters": {...}, "input": [...]}')
        command = registry.find(request["command"])
        if command is None or not command.module:
            raise CommandNotFound(f"'{request['command']}' is not a Cevahir cmdlet")
        name = command.name
        parameters = {} if request.get("parameters") is None else request["parameters"]
        if not isinstance(parameters, dict):
            raise ValueError("parameters must be an object")
        arguments = [(str(name), from_json(value)) for name, value in parameters.items()]
        bound = binder.bind(command, arguments)
        context = Context(session, command, verbose=bool(bound.get("Verbose")), whatif=bool(bound.get("WhatIf")),
                          error_action=bound.get("ErrorAction"))
        items = request.get("input")
        if command.kind == "stream":
            stream = command.function(context, binder.bind_input(command, bound), iter([from_json(i) for i in items or []]))
            records.extend(stream)
        elif not items:
            records.extend(command.function(context, binder.bind_input(command, bound)))
        else:
            for item in items:
                try:
                    records.extend(command.function(context, binder.bind_input(command, bound, from_json(item))))
                except EXPECTED as error:
                    context.error(error)
    except EXPECTED as error:
        session.report(error, name if isinstance(name, str) else None)
    return {"records": [plain(record, typed=True) for record in records],
            "errors": [record.to_dict() for record in reversed(session.errors)],
            "warnings": list(session.warnings), "exit": session.exit_code()}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        options = parse_arguments(argv)
    except Usage as error:
        print(f"cevahir: {error}\n\n{USAGE}", file=sys.stderr)
        return 2
    if options["version"]:
        from cevahir_ai import __version__
        print(__version__)
        return 0
    if options["help"]:
        print(USAGE)
        return 0
    configure_streams()
    import logging
    from cevahir_ai.console.runtime import Session
    environ = dict(os.environ)
    if options["home"]:
        environ["CEVAHIR_HOME"] = options["home"]
    interactive = not (options["command"] or options["file"] or options["invoke"]) and sys.stdin.isatty()
    session = Session(environ=environ, culture=options["culture"], width=options["width"], interactive=interactive,
                      output="json" if options["output"] == "json" or options["invoke"] else "text")
    logging.disable(logging.WARNING)          # legacy modules log at import; console output must stay readable
    try:
        if options["invoke"]:
            try:
                request = json.loads(sys.stdin.buffer.read().decode("utf-8-sig"))
            except (ValueError, UnicodeDecodeError) as error:
                print(f"cevahir: -Invoke needs one JSON request on standard input: {error}", file=sys.stderr)
                return 2
            session.stderr = _Discard()
            result = invoke(session, request)
            sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
            return result["exit"]
        profile = None if options["noprofile"] else _profile(session)
        if profile:
            session.run_file(profile)
            session.variables["erroractionpreference"] = "Continue"
        if options["file"]:
            code = session.run_file(options["file"], options["arguments"])
        elif options["command"] is not None:
            code = session.run(options["command"])
        elif interactive:
            from cevahir_ai.console import repl
            code = repl.run(session, show_banner=not options["nologo"])
        else:
            code = session.run(sys.stdin.read())
        if session.output == "json":
            from cevahir_ai.domain.records import plain
            sys.stdout.write(json.dumps([plain(item, typed=True) for item in session.collected], ensure_ascii=False) + "\n")
        return code
    except KeyboardInterrupt:
        return 130
    except Exception as error:                 # a defect: say so plainly, never a half-printed traceback by default
        if os.environ.get("CEVAHIR_DEBUG"):
            raise
        print(f"cevahir: internal error: {type(error).__name__}: {error} (set CEVAHIR_DEBUG=1 for the traceback)",
              file=sys.stderr)
        return 1
    finally:
        logging.disable(logging.NOTSET)


class _Discard:
    def write(self, text):
        return len(text)

    def flush(self):
        pass


def _profile(session):
    try:
        path = os.path.join(str(session.store.home), "profile.ps1")
    except Exception:
        return None
    return path if os.path.isfile(path) else None


if __name__ == "__main__":
    sys.exit(main())
