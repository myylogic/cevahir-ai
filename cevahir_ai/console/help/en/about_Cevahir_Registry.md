TOPIC
    about_Cevahir_Registry

SHORT DESCRIPTION
    The registry, the audit log and lineage under CEVAHIR_HOME.

WHERE
    CEVAHIR_HOME is outside the repository: %LOCALAPPDATA%\Cevahir on Windows,
    ${XDG_STATE_HOME:-~/.local/state}/cevahir on Linux and macOS, Drive on Colab when it is
    mounted. The CEVAHIR_HOME environment variable, or cevahir -Home <dir>, overrides it.
    Get-Environment shows the home in use.

ARTEFACTS AND KEYS
    Register-Artifact reads existing files and writes one manifest per artefact, named by the
    artefact's content-derived key (cev:<kind>:<scheme>:<sha256>). Nothing is moved or renamed.
    Tables show the short form (ckpt:3f9a1c0b7d2e); every command accepts the short form, the
    full key, an alias (a legacy tokenizer digest) or the path of a registered file.

    A manifest never contains a path. Where a file is on this machine is kept separately, so
    two byte-identical copies (last.pth, best.pth) are one artefact with two locations, and a
    file copied from another machine is recognised by its bytes alone.

STATES
    registered    recorded here
    imported      came in a bundle; not verified on this machine yet
    verified      the last Test-Checkpoint established its digests
    quarantined   the last test found other bytes than the registered ones, or an unsafe envelope

AUDIT LOG
    Every acting command and every verification verdict is appended to this machine's chain
    (audit/<host-id>/). Each event carries the hash of the event before it. Test-AuditLog
    recomputes the chain: an edited, removed, inserted or reordered line is reported with its
    sequence number; a chain cut at the end is reported through its head record.

    Limits, stated plainly: a process running as you can read the MAC key and rewrite the
    whole chain consistently. Detecting that needs a signed copy of the head kept off the
    machine. That, policies as data, and approvals are the next phase; today the built-in
    bootstrap policy allows read and register actions and records them.

LINEAGE
    Get-Lineage shows which keys produced an artefact (-Direction Up) or what was produced from
    it (-Direction Down). It is read from the audit events. For files registered after the fact
    only what the file itself states is known: the tokenizer of a checkpoint, the corpus digest
    in a cache name. Runs and caches enter lineage when training is run through the console
    (next phase).

MOVING BETWEEN MACHINES
    Export-Artifact -To x.cvb [-IncludePayload] [-IncludeAudit], then Import-Artifact x.cvb on
    the other machine. Import verifies everything before it writes anything. A bundle is not
    signed: import proves it is internally consistent, not who made it.

SEE ALSO
    Get-Help Register-Artifact -Full, Get-Help Test-AuditLog -Full, Get-Help about_Cevahir_Subset
