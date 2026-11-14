"""Registry cmdlets: artefacts, keys, lineage and bundles. Thin adapters over ``cevahir_ai.ops``."""
import os

from cevahir_ai.console.errors import ParameterBindingError
from cevahir_ai.console.registry import Param, cmdlet
from cevahir_ai.keys.grammar import KEY_RE

KINDS = ("auto", "ckpt", "tok", "cache", "corpus", "evalset", "file")
ARTIFACT_KINDS = ("ckpt", "model", "tok", "cache", "corpus", "evalset", "file")
STATES = ("registered", "imported", "verified", "quarantined")
_KEY_HELP = ("A key, an alias, a short form (ckpt:3f9a1c0b) or the path of a registered file.",
             "Anahtar, takma ad, kısa biçim (ckpt:3f9a1c0b) ya da kayıtlı bir dosyanın yolu.")


@cmdlet("Get-Artifact", module=True, output="Cevahir.Artifact", params=[
    Param("Key", position=0, pipeline="both", aliases=("Id",), help=_KEY_HELP),
    Param("Kind", choices=ARTIFACT_KINDS, help=("Only artefacts of this kind.", "Yalnız bu türdeki eserler.")),
    Param("State", choices=STATES, help=("Only artefacts in this state.", "Yalnız bu durumdaki eserler.")),
], synopsis=("Lists the artefacts registered in CEVAHIR_HOME.", "CEVAHIR_HOME'da kayıtlı eserleri listeler."),
    description=("Reads manifests only: no torch, no payload file is opened. State is derived from the audit "
                 "events: registered, imported (came in a bundle, not verified here), verified, quarantined.",
                 "Yalnız manifestleri okur: torch gerekmez, hiçbir yük dosyası açılmaz. Durum denetim kaydından "
                 "türetilir: registered, imported (paketle geldi, burada doğrulanmadı), verified, quarantined."),
    examples=[("Get-Artifact", "Everything registered.", "Kayıtlı her şey."),
              ("Get-Artifact -Kind ckpt | Where-Object State -ne verified", "Checkpoints not verified yet.",
               "Henüz doğrulanmamış checkpoint'ler.")])
def get_artifact(context, p):
    from cevahir_ai.ops import artifacts
    yield from artifacts.get(context.store, p["Key"], p["Kind"], p["State"])


@cmdlet("Register-Artifact", module=True, output="Cevahir.Artifact", action_class="mutate", whatif=True,
        torch="to read the tensors of a checkpoint that is not registered yet", params=[
    Param("Path", "list", position=0, mandatory=True, pipeline="both", aliases=("FullName",),
          help=("Files or folders to register.", "Kaydedilecek dosyalar ya da klasörler.")),
    Param("Kind", choices=KINDS, default="auto",
          help=("What to look for. auto: checkpoints, tokenizers and caches. corpus, evalset and file must be asked for.",
                "Ne aranacağı. auto: checkpoint, tokenizer ve önbellek. corpus, evalset ve file açıkça istenmelidir.")),
    Param("Recurse", "switch", help=("Also scan sub-folders.", "Alt klasörleri de tara.")),
], synopsis=("Registers existing files: checkpoints, tokenizers, caches, corpora.",
             "Var olan dosyaları kaydeder: checkpoint, tokenizer, önbellek, korpus."),
    description=("Nothing is moved or rewritten. Each artefact gets a manifest keyed by its content-derived key "
                 "(provenance legacy-import), its file gets a location, and one audit event records it. A file "
                 "whose bytes are already known only gets a new location (no torch).",
                 "Hiçbir şey taşınmaz ya da yeniden yazılmaz. Her eser, içerikten türetilen anahtarıyla bir manifest "
                 "(köken: legacy-import), dosyası bir konum alır ve tek bir denetim olayı yazılır. Baytları zaten "
                 "bilinen dosya yalnız yeni bir konum alır (torch gerekmez)."),
    examples=[("Register-Artifact .\\saved_models\\checkpoints -Recurse", "Register every checkpoint under a folder.",
               "Bir klasör altındaki bütün checkpoint'leri kaydet."),
              ("Register-Artifact .\\education -Kind corpus", "Register a folder as a corpus.",
               "Bir klasörü korpus olarak kaydet."),
              ("Register-Artifact .\\data -Recurse -WhatIf", "Show what would be scanned.", "Neyin taranacağını göster.")])
def register_artifact(context, p):
    from cevahir_ai.ops import artifacts
    for path in p["Path"]:
        path = os.path.expanduser(path)
        if context.what_if("Register-Artifact", f"{path} (kind {p['Kind']}{', recurse' if p['Recurse'] else ''})"):
            continue
        context.verbose_message(f"scanning {path}")
        try:
            yield from artifacts.register(context.store, path, p["Kind"], p["Recurse"], warn=context.warning)
        except (OSError, ValueError) as error:
            context.error(error)


@cmdlet("Get-Lineage", module=True, output="Cevahir.LineageEdge", params=[
    Param("Key", position=0, mandatory=True, pipeline="both", aliases=("Id",), help=_KEY_HELP),
    Param("Direction", choices=("Up", "Down", "Both"), default="Up",
          help=("Up: what it was made from. Down: what was made from it.",
                "Up: neyden üretildiği. Down: ondan ne üretildiği.")),
    Param("Depth", "int", help=("How many levels to follow (default: all).", "Kaç düzey izlenir (varsayılan: hepsi).")),
], synopsis=("Shows which keys produced an artefact, or what it produced.",
             "Bir eseri hangi anahtarların ürettiğini ya da ondan ne üretildiğini gösterir."),
    description=("Lineage is read from the audit events, so it cannot disagree with the audit log.",
                 "Soy ağacı denetim olaylarından okunur; denetim kaydıyla çelişemez."),
    examples=[("Get-Lineage model:51c0aa -Direction Up | Format-Table -View Tree", "From a model back to its tokenizer.",
               "Modelden tokenizer'ına doğru."),
              ("Get-Tokenizer | Get-Lineage -Direction Down", "Everything trained with a tokenizer.",
               "Bir tokenizer ile eğitilen her şey.")])
def get_lineage(context, p):
    from cevahir_ai.ops import artifacts
    yield from artifacts.lineage_of(context.store, p["Key"], p["Direction"], p["Depth"])


@cmdlet("Get-Key", module=True, output="Cevahir.KeyInfo", torch="for a checkpoint file that is not registered", params=[
    Param("Path", position=0, pipeline="property", aliases=("FullName",),
          help=("A file: every key derivable from it. A key or short form is also accepted here.",
                "Dosya: ondan türetilebilen bütün anahtarlar. Buraya anahtar ya da kısa biçim de yazılabilir.")),
    Param("Key", pipeline="property", help=("A key to describe.", "Açıklanacak anahtar.")),
    Param("Explain", "switch", help=("Add what the registry knows about the key (members, inputs, fields).",
                                     "Kaydın anahtar hakkında bildiklerini ekle (üyeler, girdiler, alanlar).")),
], synopsis=("Shows the keys of a file, or what a key means.", "Bir dosyanın anahtarlarını ya da bir anahtarın anlamını gösterir."),
    examples=[("Get-Key .\\data\\merges_lib\\merges.txt", "File keys of a text file.", "Bir metin dosyasının dosya anahtarları."),
              ("Get-Key ckpt:3f9a1c0b -Explain | Format-List", "What the registry knows about a checkpoint key.",
               "Kaydın bir checkpoint anahtarı hakkında bildikleri.")])
def get_key(context, p):
    from cevahir_ai.ops import keys
    path, key = p["Path"], p["Key"]
    if path is not None and key is None and not os.path.isfile(os.path.expanduser(path)):
        from cevahir_ai.store.catalog import SHORT_RE
        if KEY_RE.match(path) or SHORT_RE.match(path):      # a drive letter (C:\...) is never a key kind
            path, key = None, path
    if path is None and key is None:
        raise ParameterBindingError("Get-Key needs a file path or a key")
    yield from keys.keys_of(context.store, os.path.expanduser(path) if path else None, key, p["Explain"])


@cmdlet("Resolve-Key", module=True, output="Cevahir.KeyInfo", params=[
    Param("Ref", position=0, mandatory=True, pipeline="both", aliases=("Key", "Id"), help=_KEY_HELP),
], synopsis=("Resolves a reference to the full key of a registered artefact.",
             "Bir başvuruyu kayıtlı eserin tam anahtarına çözer."),
    examples=[("Resolve-Key ckpt:3f9a | Select-Object -ExpandProperty Key", "The full key behind a short form.",
               "Kısa biçimin arkasındaki tam anahtar.")])
def resolve_key(context, p):
    from cevahir_ai.ops import artifacts, keys
    artifact = artifacts.resolve(context.store, p["Ref"])
    yield from keys.keys_of(context.store, None, artifact.key)


@cmdlet("Compare-Key", module=True, output="Cevahir.KeyDiff", torch="for a checkpoint file that is not registered", params=[
    Param("Reference", position=0, mandatory=True, help=_KEY_HELP),
    Param("Difference", position=1, mandatory=True, help=_KEY_HELP),
], synopsis=("Compares two artefacts member by member.", "İki eseri üye üye karşılaştırır."),
    description=("For checkpoints it names the architecture fields and the tensors that differ.",
                 "Checkpoint'lerde farklı olan mimari alanlarını ve tensörleri adlandırır."),
    examples=[("Compare-Key ckpt:3f9a1c0b ckpt:a17e55d0 | Where-Object Equal -eq $false",
               "Only the members that differ.", "Yalnız farklı olan üyeler.")])
def compare_key(context, p):
    from cevahir_ai.ops import keys
    rows = keys.compare(context.store, p["Reference"], p["Difference"])
    if any(not row.Equal for row in rows):
        context.verbose_message("the two sides differ")
    yield from rows


@cmdlet("Get-KeyScheme", module=True, output="Cevahir.KeyScheme", params=[
    Param("Kind", position=0, help=("Only schemes of this kind (tok, ckpt, weights ...).",
                                    "Yalnız bu türün şemaları (tok, ckpt, weights ...).")),
    Param("Status", choices=("primary", "legacy", "signature", "occurrence"),
          help=("Only schemes with this status.", "Yalnız bu durumdaki şemalar.")),
], synopsis=("Lists the key schemes.", "Anahtar şemalarını listeler."),
    examples=[("Get-KeyScheme -Kind tok", "The tokenizer schemes.", "Tokenizer şemaları.")])
def get_key_scheme(context, p):
    from cevahir_ai.ops import keys
    yield from keys.scheme_records(p["Kind"], p["Status"])


@cmdlet("Export-Artifact", module=True, kind="stream", output="Cevahir.Bundle", action_class="mutate", whatif=True, params=[
    Param("Key", "list", position=0, aliases=("Id",), help=("Artefacts to export (or pipe them in).",
                                                            "Dışa aktarılacak eserler (ya da boruyla verin).")),
    Param("To", "path", mandatory=True, help=("The .cvb file to write.", "Yazılacak .cvb dosyası.")),
    Param("IncludePayload", "switch", help=("Also pack the payload files (the checkpoint itself).",
                                            "Yük dosyalarını da paketle (checkpoint'in kendisi).")),
    Param("IncludeAudit", "switch",
          help=("Also pack this machine's audit chain. It holds every recorded action, with local paths.",
                "Bu makinenin denetim zincirini de paketle. Zincir, yerel yollarıyla kayıtlı her eylemi içerir.")),
], synopsis=("Writes artefacts to a .cvb bundle for another machine.", "Eserleri başka bir makine için .cvb paketine yazar."),
    description=("Manifests of the registered ancestors (tokenizer of a checkpoint) travel too, so lineage "
                 "survives the move. The bundle is not signed.",
                 "Kayıtlı ataların manifestleri de (checkpoint'in tokenizer'ı) taşınır; soy ağacı taşımadan sonra "
                 "da kalır. Paket imzalı değildir."),
    examples=[("Get-Checkpoint -Best | Export-Artifact -To D:\\transfer\\best.cvb -IncludePayload",
               "Pack the best checkpoint with its file.", "En iyi checkpoint'i dosyasıyla paketle.")])
def export_artifact(context, p, items):
    from cevahir_ai.console.expr import member, to_text
    from cevahir_ai.domain.records import Record
    from cevahir_ai.ops import artifacts
    references = list(p["Key"] or [])
    for item in items:
        key = member(item, "Key") if isinstance(item, (Record, dict)) else item
        if key is not None:
            references.append(to_text(key))
    if not references:
        raise ParameterBindingError("Export-Artifact needs at least one artefact (-Key or pipeline input)")
    if context.what_if("Export-Artifact", f"{len(references)} artefact(s) -> {p['To']}"):
        return
    yield artifacts.export(context.store, references, p["To"], p["IncludePayload"], p["IncludeAudit"])


@cmdlet("Import-Artifact", module=True, output="Cevahir.Artifact", action_class="mutate", whatif=True, params=[
    Param("Path", "path", position=0, mandatory=True, pipeline="both", aliases=("FullName",),
          help=("The .cvb bundle.", ".cvb paketi.")),
    Param("Destination", "path", help=("Where to unpack payload files (default: blobs/ in CEVAHIR_HOME).",
                                       "Yük dosyalarının açılacağı yer (varsayılan: CEVAHIR_HOME içinde blobs/).")),
], synopsis=("Imports a .cvb bundle after verifying all of it.", "Bir .cvb paketini tamamını doğruladıktan sonra içe aktarır."),
    description=("Nothing is written unless every manifest hashes to its name, every payload to its file key, and "
                 "an included audit chain verifies and does not contradict a copy already held.",
                 "Her manifest adına, her yük dosya anahtarına özetlenmedikçe ve paketteki denetim zinciri doğrulanıp "
                 "eldeki kopyayla çelişmedikçe hiçbir şey yazılmaz."),
    examples=[("Import-Artifact D:\\transfer\\best.cvb | Test-Checkpoint -Level Compat -Purpose evaluate",
               "Import, then verify here.", "İçe aktar, sonra burada doğrula.")])
def import_artifact(context, p):
    from cevahir_ai.ops import artifacts
    if context.what_if("Import-Artifact", p["Path"]):
        return
    yield from artifacts.import_bundle(context.store, p["Path"], p["Destination"])
