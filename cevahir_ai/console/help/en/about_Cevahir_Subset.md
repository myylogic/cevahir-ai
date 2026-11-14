TOPIC
    about_Cevahir_Subset

SHORT DESCRIPTION
    The console language is a strict subset of PowerShell. What it accepts means what it means in
    PowerShell; what it does not accept is rejected at parse time (error id NotInCevahirSubset,
    exit code 2). Nothing is approximated.

WHAT IT HAS
    Commands        Verb-Noun cmdlets with -Parameter value, -Switch, -Name:value, positional
                    arguments, lists (a, b, c). Names and parameters ignore case; a parameter can
                    be shortened to a unique prefix. Get-CvArtifact is the same as Get-Artifact.
    Pipeline        cmd | cmd | cmd. Objects flow, not text. A line that ends in | continues.
    Variables       $x = <pipeline>, $x += <value>, $x.Property, $x[0], "text $x and $($x.Key)"
                    Automatic: $_ $PSItem $true $false $null $Error $LASTEXITCODE $? $CevahirHome
                    $PolicyKey $PWD $PSScriptRoot; $ErrorActionPreference can be set.
    Statements      if / elseif / else, foreach ($x in ...) { }, throw, exit <code>,
                    param( ) at the start of a script file.
    Expressions     -eq -ne -gt -ge -lt -le -like -notlike -match -notmatch -contains -notcontains
                    -in -notin (and -ceq, -clike ... for case-sensitive), -and -or -not !,
                    + - * / %, numbers with KB MB GB, 'literal' and "expanding" strings,
                    @( ) arrays, @{ Name = value } hashtables.
    String methods  .Substring .StartsWith .EndsWith .Contains .ToLower .ToUpper .Trim .Split
                    .Replace, and the properties .Length and .Count.
    Scripts         cevahir -File runbook.ps1 -Name prod; .\other.ps1 from inside a script.
                    Files are UTF-8; CRLF and LF both work. Scripts stop at the first error.
    Comments        # to the end of the line, <# ... #>

SCRIPT BLOCKS ARE NOT CODE
    A script block { ... } is accepted only as the filter of Where-Object, ForEach-Object and
    Sort-Object and as the Expression of a calculated property. It holds exactly one expression
    over $_ : comparisons, -and/-or/-not, arithmetic, property access, literals and the string
    methods above. Inside a block there are no commands, no assignments, no statements, no
    pipelines; ( ) and $( ) hold expressions only.

        Get-Checkpoint | Where-Object { $_.Epoch -ge 3 -and $_.State -ne 'quarantined' }
        Get-Artifact   | Select-Object Short, @{Name='Files'; Expression={ $_.Locations.Count }}

    To run a cmdlet for each object, pipe into the cmdlet:

        Get-Checkpoint | Test-Checkpoint            (not: ForEach-Object { Test-Checkpoint $_ })

    A string is never evaluated as code. There is no Invoke-Expression.

WHAT IS REJECTED ON PURPOSE
    function, filter, class, enum, switch, while, do, for, break, continue, return,
    try / catch / finally, trap; casts and type literals ([int], [System.IO.File]);
    static members (::); the & and . invocation operators; redirection (> >> 2>) and
    background jobs; here-strings; splatting; $env: and other scoped variables;
    the -f, -replace, -split, -join, -is, -as and bitwise operators; members whose
    name starts with an underscore; methods other than the string methods above.

THE CONSOLE NEVER STARTS ANOTHER PROGRAM
    A name that is not a cmdlet of the console (python, git, rm) is CommandNotFoundException.
    To keep output in a file use Out-File.

EXIT CODES
    0 success; 1 error (includes not found); 2 parse or usage error; 3 policy denied;
    4 approval required (reserved); 5 verification failed (Test-Checkpoint, Test-AuditLog);
    130 interrupted. exit <n> in a script sets the code.

SEE ALSO
    Get-Help about_Cevahir_Registry, Get-Command, Get-Help <command> -Full
