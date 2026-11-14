"""Checkpoint, model, tokenizer and cache cmdlets. Thin adapters over ``cevahir_ai.ops.checkpoints``."""
from cevahir_ai.console.errors import ParameterBindingError
from cevahir_ai.console.registry import Param, cmdlet
from cevahir_ai.domain.verdicts import PURPOSES

_KEY_HELP = ("A key, an alias, a short form (ckpt:3f9a1c0b) or the path of a registered file.",
             "Anahtar, takma ad, kısa biçim (ckpt:3f9a1c0b) ya da kayıtlı bir dosyanın yolu.")


@cmdlet("Get-Checkpoint", module=True, output="Cevahir.Checkpoint", params=[
    Param("Key", position=0, pipeline="both", aliases=("Id",), help=_KEY_HELP),
    Param("Path", "path", pipeline="property", aliases=("FullName",),
          help=("The checkpoint registered from this file.", "Bu dosyadan kaydedilen checkpoint.")),
    Param("Latest", "switch", help=("Only the checkpoint whose file was written last.",
                                    "Yalnız dosyası en son yazılan checkpoint.")),
    Param("Best", "switch", help=("Only the checkpoint with the lowest Metric.",
                                  "Yalnız en düşük Metric değerli checkpoint.")),
], synopsis=("Lists registered checkpoints.", "Kayıtlı checkpoint'leri listeler."),
    description=("Torch-free: epoch, metric, architecture and tokenizer come from the manifest.",
                 "torch gerekmez: epoch, metrik, mimari ve tokenizer manifestten gelir."),
    examples=[("Get-Checkpoint | Sort-Object Epoch -Descending | Select-Object -First 3", "The three newest epochs.",
               "En yeni üç epoch."),
              ("Get-Checkpoint -Latest | Test-Checkpoint -Level Compat -Purpose evaluate", "Verify the newest checkpoint.",
               "En yeni checkpoint'i doğrula.")])
def get_checkpoint(context, p):
    from cevahir_ai.ops import checkpoints
    yield from checkpoints.get_checkpoints(context.store, p["Key"], p["Latest"], p["Best"], p["Path"])


@cmdlet("Test-Checkpoint", module=True, output="Cevahir.CheckpointVerdict", aliases=("inspect",),
        torch="for -Level Weights and -Level Compat", params=[
    Param("Key", position=0, pipeline="both", aliases=("Id",), help=_KEY_HELP),
    Param("Path", "path", pipeline="property", aliases=("FullName",),
          help=("A checkpoint file (registered or not).", "Bir checkpoint dosyası (kayıtlı olsun olmasın).")),
    Param("Level", choices=("File", "Weights", "Compat"), default="File",
          help=("File: bytes and envelope safety, no torch. Weights: also read the tensors. Compat: also compare "
                "with the layout the current code builds and with -Target.",
                "File: baytlar ve zarf güvenliği, torch'suz. Weights: tensörler de okunur. Compat: ayrıca bugünkü "
                "kodun kurduğu düzenle ve -Target ile karşılaştırılır.")),
    Param("Purpose", choices=PURPOSES, default="inspect",
          help=("What the checkpoint is to be used for; it decides which checks are required.",
                "Checkpoint'in ne için kullanılacağı; hangi denetimlerin gerektiğini belirler.")),
    Param("Target", help=("Target architecture: a preset (tiny, train-default, cev11-primary) or a JSON config file.",
                          "Hedef mimari: ön ayar (tiny, train-default, cev11-primary) ya da JSON yapılandırma dosyası.")),
    Param("Vocab", "path", help=("Target legacy tokenizer: vocab.json (with -Merges).",
                                 "Hedef eski tokenizer: vocab.json (-Merges ile birlikte).")),
    Param("Merges", "path", help=("Target legacy tokenizer: merges.txt.", "Hedef eski tokenizer: merges.txt.")),
    Param("ByteBpe", "path", help=("Target byte-level BPE tokenizer file.", "Hedef bayt düzeyi BPE tokenizer dosyası.")),
    Param("Dtype", default="float32", help=("Target floating dtype.", "Hedef kayan nokta türü.")),
    Param("MaxSeqLength", "int", help=("Serving context length.", "Sunum bağlam uzunluğu.")),
    Param("AllowMigration", "list", help=("Ids of migrations that may be applied in memory.",
                                          "Bellekte uygulanabilecek göçlerin kimlikleri.")),
    Param("NoRecord", "switch", help=("Do not append the verdict to the audit log.",
                                      "Kararı denetim kaydına ekleme.")),
], synopsis=("Runs the compatibility gate on a checkpoint.", "Bir checkpoint üzerinde uyumluluk kapısını çalıştırır."),
    description=("A registered checkpoint is tested against its manifest: the bytes on disk must be the registered "
                 "file (C1) and the tensors the registered weights (C3). Test-Checkpoint never throws for an "
                 "incompatible checkpoint: the verdict says so, and the exit code of the console is 5. The verdict "
                 "is appended to the audit log (artifact.verified).",
                 "Kayıtlı checkpoint manifestine karşı sınanır: diskteki baytlar kayıtlı dosya (C1), tensörler kayıtlı "
                 "ağırlıklar (C3) olmalıdır. Uyumsuz checkpoint için hata fırlatılmaz: karar bunu söyler ve konsolun "
                 "çıkış kodu 5 olur. Karar denetim kaydına eklenir (artifact.verified)."),
    examples=[("Get-Checkpoint | Test-Checkpoint | Format-Table", "File-level check of every checkpoint (no torch).",
               "Her checkpoint'in dosya düzeyi denetimi (torch'suz)."),
              ("Test-Checkpoint -Path .\\last.pth -Level Compat -Purpose serve | Format-List Verdict, Summary, Checks",
               "Is this file fit for serving, and if not, why.", "Bu dosya sunuma uygun mu, değilse neden."),
              ("Get-Checkpoint -Latest | Test-Checkpoint -Level Compat -Purpose evaluate | Format-Table -View Checks",
               "One row per check.", "Her denetim için bir satır.")])
def test_checkpoint(context, p):
    from cevahir_ai.ops import checkpoints
    if p["Key"] is None and p["Path"] is None:
        raise ParameterBindingError("Test-Checkpoint needs a checkpoint key or -Path")
    target = None
    if p["Level"] != "File":
        target = checkpoints.build_target(p["Target"], p["Vocab"], p["Merges"], p["ByteBpe"], p["Dtype"],
                                          p["MaxSeqLength"], p["AllowMigration"])
    verdict = checkpoints.test_checkpoint(context.store, p["Key"], None if p["Key"] is not None else p["Path"],
                                          p["Level"], p["Purpose"], target, record=not p["NoRecord"])
    if not verdict.Ok:
        context.failed(5)
    yield verdict


@cmdlet("Get-Model", module=True, output="Cevahir.Model", params=[
    Param("Key", position=0, pipeline="both", aliases=("Id",), help=_KEY_HELP),
], synopsis=("Lists servable model identities (weights + architecture + tokenizer).",
             "Sunulabilir model kimliklerini listeler (ağırlıklar + mimari + tokenizer)."),
    description=("A model key exists only for checkpoints whose architecture and tokenizer are known. Publishing "
                 "does not exist yet: Published is always false.",
                 "Model anahtarı yalnız mimarisi ve tokenizer'ı bilinen checkpoint'ler için vardır. Yayınlama henüz "
                 "yoktur: Published her zaman false'tur."),
    examples=[("Get-Model | Format-List Key, Checkpoints", "Models and the checkpoints that hold them.",
               "Modeller ve onları taşıyan checkpoint'ler.")])
def get_model(context, p):
    from cevahir_ai.ops import checkpoints
    yield from checkpoints.get_models(context.store, p["Key"])


@cmdlet("Get-Tokenizer", module=True, output="Cevahir.Tokenizer", params=[
    Param("Key", position=0, pipeline="both", aliases=("Id",), help=_KEY_HELP),
    Param("Tracked", "switch", help=("The tokenizers pinned in governance/registry/tokenizers.json instead.",
                                     "Bunun yerine governance/registry/tokenizers.json içinde sabitlenen tokenizer'lar.")),
], synopsis=("Lists registered tokenizers.", "Kayıtlı tokenizer'ları listeler."),
    examples=[("Get-Tokenizer -Tracked | Format-List Key, Aliases", "The tracked tokenizer and its legacy digests.",
               "İzlenen tokenizer ve eski özetleri.")])
def get_tokenizer(context, p):
    from cevahir_ai.ops import checkpoints
    yield from checkpoints.get_tokenizers(context.store, p["Key"], p["Tracked"])


@cmdlet("Get-Cache", module=True, output="Cevahir.Cache", params=[
    Param("Key", position=0, pipeline="both", aliases=("Id",), help=_KEY_HELP),
], synopsis=("Lists registered training caches.", "Kayıtlı eğitim önbelleklerini listeler."),
    examples=[("Get-Cache | Where-Object Sidecar -eq stale", "Caches whose .sha256 sidecar is stale.",
               ".sha256 yan dosyası eskimiş önbellekler.")])
def get_cache(context, p):
    from cevahir_ai.ops import checkpoints
    yield from checkpoints.get_caches(context.store, p["Key"])
