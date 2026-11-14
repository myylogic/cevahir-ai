"""Audit cmdlets. Thin adapters over ``cevahir_ai.ops.audit``."""
from cevahir_ai.console.registry import Param, cmdlet


@cmdlet("Get-AuditEvent", module=True, output="Cevahir.AuditEvent", params=[
    Param("Last", "int", help=("Only the last n events.", "Yalnız son n olay.")),
    Param("Since", help=("Events since a duration ago (14d, 2h, 30m) or a date (2026-11-13).",
                         "Bir süre öncesinden (14d, 2h, 30m) ya da bir tarihten (2026-11-13) bu yana olaylar.")),
    Param("Event", help=("Event name, wildcards allowed (artifact.*).", "Olay adı, joker karakter olabilir (artifact.*).")),
    Param("Action", help=("The cmdlet that acted, wildcards allowed (Register-*).",
                          "Eylemi yapan cmdlet, joker karakter olabilir (Register-*).")),
    Param("Target", pipeline="property", aliases=("Key",),
          help=("Only events about this artefact (key, alias or short form).",
                "Yalnız bu eserle ilgili olaylar (anahtar, takma ad ya da kısa biçim).")),
    Param("HostId", help=("The chain of another machine held in this home (default: this machine).",
                          "Bu evde tutulan başka bir makinenin zinciri (varsayılan: bu makine).")),
], synopsis=("Reads the audit log.", "Denetim kaydını okur."),
    description=("Every acting command (register, import, export) and every verification verdict is recorded with "
                 "who acted, the decision of the policy and the outcome. Plain listings are not recorded.",
                 "Eylem yapan her komut (kayıt, içe ve dışa aktarma) ve her doğrulama kararı; kimin yaptığı, politika "
                 "kararı ve sonuçla birlikte kaydedilir. Yalın listelemeler kaydedilmez."),
    examples=[("Get-AuditEvent -Last 5 | Format-Table Seq, Event, Action, Decision, Outcome", "The last five events.",
               "Son beş olay."),
              ("Get-Checkpoint -Latest | Get-AuditEvent | Format-List Seq, Time, Event, OutcomeDetail",
               "The history of one checkpoint.", "Tek bir checkpoint'in geçmişi.")])
def get_audit_event(context, p):
    from cevahir_ai.ops import audit
    yield from audit.get_events(context.store, p["Last"], p["Since"], p["Event"], p["Action"], p["Target"], p["HostId"])


@cmdlet("Test-AuditLog", module=True, output="Cevahir.AuditVerdict", params=[
    Param("HostId", position=0, help=("Only this chain (default: this machine's and every imported chain).",
                                      "Yalnız bu zincir (varsayılan: bu makinenin ve içe aktarılan her zincir).")),
    Param("Repair", "switch",
          help=("Move a torn last line (crash damage) to quarantine and rebuild a missing head; record it. "
                "Edits and removals are never repaired.",
                "Yarım kalmış son satırı (çökme hasarı) karantinaya taşı, eksik baş kaydını yeniden kur ve bunu kaydet. "
                "Düzenleme ve silmeler asla onarılmaz.")),
], synopsis=("Verifies the hash chain of the audit log.", "Denetim kaydının özet zincirini doğrular."),
    description=("Recomputes every event hash, every link, the sequence numbers, the MACs (for this machine's own "
                 "chain) and the head record. An edited, removed, inserted or reordered line is reported with its "
                 "sequence number; a chain cut at the end is reported through its head. Exit code 5 when a chain "
                 "does not verify. A process running as the same user can still rewrite chain, head and MACs "
                 "together: detecting that needs anchors kept off the machine, which do not exist yet.",
                 "Her olay özetini, her bağı, sıra numaralarını, MAC'leri (bu makinenin kendi zinciri için) ve baş "
                 "kaydını yeniden hesaplar. Düzenlenen, silinen, eklenen ya da yeri değiştirilen satır sıra "
                 "numarasıyla bildirilir; sondan kesilen zincir baş kaydı üzerinden bildirilir. Zincir "
                 "doğrulanmazsa çıkış kodu 5'tir. Aynı kullanıcıyla çalışan bir süreç zinciri, baş kaydını ve "
                 "MAC'leri birlikte yeniden yazabilir: bunu yakalamak için makine dışında tutulan çapalar gerekir; "
                 "henüz yoktur."),
    examples=[("Test-AuditLog", "Verify every chain in this home.", "Bu evdeki her zinciri doğrula."),
              ("Test-AuditLog | Select-Object -ExpandProperty Findings | Format-Table Seq, Segment, Line, Problem",
               "What exactly is wrong, and where.", "Tam olarak neyin, nerede yanlış olduğu.")])
def test_audit_log(context, p):
    from cevahir_ai.ops import audit
    for verdict in audit.test_log(context.store, p["HostId"], p["Repair"]):
        if not verdict.Ok:
            context.failed(5)
        yield verdict
