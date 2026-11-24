# Publishing research results

This file records how Muhammed Yasin Yılmaz's research results are brought to GitHub and published as archive
versions with a DOI carrying his name. It schedules nothing; it is the contract of future research rounds.

## From a new result to a release

1. **Keep what was there.** Keep the earlier research code, reports, measurements and negative results. Put a new
   claim and experiment in a new dated folder. Where an earlier reading changes, explain why in the new text, and do
   not delete what was found before. A document withdrawn by its author is replaced by a note that names the
   release holding its full text.
2. **State the contribution.** Write the contribution to the main question and what is out of scope. Name the
   strongest fitting baseline, the information each side had, the total cost, the negative control and the limit of
   test leakage. Do not present an idea known in the literature as a new principle.
3. **Run and check.** Run the experiment, and compare the recorded result with an independent computation or a
   meaningful counter-example. Do not call a hash check a reproduction. Record the new verification date and source
   version apart.
4. **Catalog and editions.** Add the work's identifier, the real author, the date, the abstract and the path of the
   original record to the catalog. `python scripts/build_publications.py --write` builds each edition from its
   source; it does not change the original research records. Review titles and interpretations against the sources.
5. **Book and reviewed sources.** Update the book where it is affected, review the links and symbols in the code,
   and then renew the explicit review record with `python scripts/check_book.py --record-reviewed-sources`. CI must
   never renew that record by itself.
6. **Checks.** `python scripts/check_book.py`, `python scripts/build_publications.py` and
   `python scripts/check_publications.py` must pass, and the reproduction commands for the changed experiment must be
   run. Do not present tests that were not run as run. Never put credentials, private data or personal local
   sessions in a release.
7. **Metadata.** Match the version and author in the catalog, CITATION.cff and .zenodo.json. Do not invent an ORCID,
   affiliation, co-author, reviewer or signature the author did not give. .zenodo.json takes precedence in GitHub
   archiving.
8. **Release.** Commit the reviewed files, push to GitHub and check CI. Make a new tag and GitHub Release for a
   meaningful, completed research package, not for every intermediate step. Never move an old tag or force-push.
9. **Verify the archive.** From Zenodo's actual published record, verify the version DOI, the all-versions DOI, the
   author, title and version, and that the files match the tag. Do not present a DOI reserved in a draft as published.
   If the GitHub release exists but Zenodo failed, keep the release open and complete it from the actual error.
10. **Record the DOI.** Add the real DOI to the citation file, the release record and the README links. The DOI
    identifies the tagged archive version, not the later commit that carries it. Finish with a clean working tree and
    local main equal to origin/main.

## Collection and individual papers

The texts, sources and code are archived under one software and research archive DOI per release. Each text has a CEV
identifier, which is not a DOI. If a text is later archived as a separate preprint, its own record and DOI and its
relation to the collection are added. The same DOI is never used as if it were the separate DOI of different works.

## What verification means

The author attribution rests on the project owner's request. Publishing with the authority of a GitHub session is
not a cryptographic commit signature or an external identity check. Finite scans, source digests and CI are
technical checks; peer review and scientific novelty review are separate. The AI-assistance statement is in every
edition.

Sources: [referencing and citing content on GitHub](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content),
[enabling a repository in Zenodo](https://help.zenodo.org/docs/github/enable-repository/),
[Zenodo metadata precedence](https://help.zenodo.org/docs/github/describe-software/zenodo-json/).
