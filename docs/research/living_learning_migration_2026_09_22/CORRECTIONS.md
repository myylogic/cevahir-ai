# Tur 16 denetim düzeltmesi

İlk ana çalıştırma sonrası bağımsız kod denetimi, `reserve_8` yönteminin gecikme kuralını kurarken sekiz tampon bitini yeniden işlediğini, fakat `install_replay_bits` sayacına bunu eklemediğini buldu. Bu satırın kaynak maliyeti 0 yerine 8 olmalıydı. Tahmin, durum kümesi, etiket sırası veya edinim sonucu değişmedi.

İlk sonuç ve o çalıştırmanın kaynağı silinmedi: [ilk sonuç](../../../research/living_learning_migration_2026_09_22/results/migration_learning_initial_cost_omission.json), [ilk kaynak](../../../research/living_learning_migration_2026_09_22/results/initial_source_cost_omission.txt). Bunlar maliyet eksikliğini içeren tarihsel kayıtlardır; geçerli ana sonuç olarak kullanılmaz.

Sayaç düzeltilirken ilk sürümün devam eden tam tekrar işlemi eski/yeni kaynak-sonuç eşleşmesi değiştiği için başarısız oldu; bu çalıştırma geçmiş bir doğrulama olarak kabul edilmedi. Sayaç düzeltildikten sonra 576 yol yeniden üretildi ve tam tekrar denetimi yeni kaynakla yeniden başlatıldı. Son denetim, ilk ve düzeltilmiş çalıştırmanın bütün kalibrasyon kayıtları ile 37.440 ana akış tahmin kaydının aynı olduğunu ayrıca kontrol eder. Yalnız sayaç ve bundan türeyen durum/kaynak hash'leri değişebilir. Mevcut önceki araştırma sonuçlarından hiçbiri geri alınmadı.
