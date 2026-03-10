# -*- coding: utf-8 -*-
"""
Topic Configuration - Konu Bazlı Scraping Yapılandırması (İleri Seviye)

Bu dosya, topic_based_scraper.py için konu bazlı sayfa listeleri ve kategorileri içerir.
Her konu için 1000+ sayfa toplamak üzere optimize edilmiştir.

HESAPLAMA:
- Deep mod: Kategori başına 200 sayfa
- 1000 sayfa için: 5+ kategori yeterli
- Her konu için 10-20 kategori eklenmiştir
- Toplam potansiyel: 2000-4000 sayfa/konu (deep mod)
"""

# ============================================================================
# KONU BAZLI SAYFA LİSTELERİ (Genişletilmiş)
# ============================================================================
# Her konu için önceden tanımlı Wikipedia sayfa başlıkları
# Bu sayfalar kategori bazlı toplama ile birleştirilir
# ============================================================================

TOPIC_PAGES = {
    "tarih": [
        # Genel tarih
        "Tarih", "Dünya_tarihi", "Türk_tarihi", "Osmanlı_İmparatorluğu",
        "Türkiye_Cumhuriyeti", "Selçuklu_İmparatorluğu", "Anadolu_Selçuklu_Devleti",
        "Bizans_İmparatorluğu", "Roma_İmparatorluğu", "Antik_Yunan", "Mısır_uygarlığı",
        # Dönemler
        "Antik_çağ", "Orta_çağ", "Yeni_çağ", "Yakın_çağ", "Modern_çağ",
        # Önemli olaylar
        "Sanayi_Devrimi", "Fransız_İhtilali", "Amerikan_Bağımsızlık_Savaşı",
        "I._Dünya_Savaşı", "II._Dünya_Savaşı", "Soğuk_Savaş", "Berlin_Duvarı",
        # Türk tarihi detayları
        "Anadolu_Selçuklu_Devleti", "Büyük_Selçuklu_İmparatorluğu", "Timur_İmparatorluğu",
        "Memlük_Sultanlığı", "Safevi_Devleti", "Babür_İmparatorluğu"
    ],
    "islam_tarihi": [
        "İslam_tarihi", "İslam", "Peygamber", "Muhammed", "Hicret",
        "Dört_Halife", "Ebu_Bekir", "Ömer", "Osman", "Ali",
        "Emeviler", "Abbasiler", "Osmanlı_İmparatorluğu", "Endülüs",
        "İslam_bilim", "İslam_felsefesi", "İslam_edebiyatı", "İslam_sanatı",
        "Kur'an", "Hadis", "Sünnet", "Fıkıh", "Kelam", "Tasavvuf",
        "Mescid-i_Nebevi", "Kabe", "Mekke", "Medine", "Kudüs"
    ],
    "dinler_tarihi": [
        "Din", "Dinler_tarihi", "İslam", "Hristiyanlık", "Yahudilik",
        "Budizm", "Hinduizm", "Taoizm", "Şinto", "Zerdüştlük", "Maniheizm",
        "Sihizm", "Jainizm", "Bahailik", "Rastafaryanizm", "Paganizm"
    ],
    "dunya_tarihi": [
        "Dünya_tarihi", "Antik_çağ", "Orta_çağ", "Yeni_çağ", "Yakın_çağ",
        "Sanayi_Devrimi", "Fransız_İhtilali", "Amerikan_Bağımsızlık_Savaşı",
        "I._Dünya_Savaşı", "II._Dünya_Savaşı", "Soğuk_Savaş",
        "Rönesans", "Reformasyon", "Aydınlanma_Çağı", "Sömürgecilik"
    ],
    "hastaliklar_tarihi": [
        "Hastalık", "Tıp_tarihi", "Veba", "Kolera", "Çiçek_hastalığı",
        "Grip", "Tüberküloz", "Sıtma", "AIDS", "COVID-19", "Pandemi",
        "Salgın", "Epidemi", "Bulaşıcı_hastalık", "Kronik_hastalık"
    ],
    "bilim_tarihi": [
        "Bilim_tarihi", "Bilim", "Fizik", "Kimya", "Biyoloji", "Matematik",
        "Astronomi", "Tıp", "Teknoloji", "Mühendislik", "Bilimsel_devrim"
    ],
    "edebiyat_tarihi": [
        "Edebiyat_tarihi", "Türk_edebiyatı", "Dünya_edebiyatı", "Şiir",
        "Roman", "Hikâye", "Tiyatro", "Deneme", "Edebiyat_akımları"
    ],
    "sanat_tarihi": [
        "Sanat_tarihi", "Resim", "Heykel", "Mimarlık", "Müzik", "Tiyatro",
        "Sinema", "Fotoğraf", "Dans", "Sanat_akımları"
    ],
    "kimya": [
        "Kimya", "Genel_kimya", "Organik_kimya", "İnorganik_kimya", "Fiziksel_kimya",
        "Analitik_kimya", "Biyokimya", "Kuantum_kimyası", "Teorik_kimya",
        "Periyodik_tablo", "Element", "Bileşik", "Karışım", "Saf_madde",
        "Molekül", "Atom", "İyon", "İzotop", "İzobar", "İzoelektronik",
        "Kimyasal_tepkime", "Kimyasal_denklem", "Stokiyometri", "Kimyasal_denge",
        "Asit", "Baz", "Tuz", "pH", "Tampon_çözelti", "Nötrleşme",
        "Bağ", "Kovalent_bağ", "İyonik_bağ", "Metalik_bağ", "Hidrojen_bağı",
        "Reaksiyon", "Ekzotermik_reaksiyon", "Endotermik_reaksiyon", "Kataliz",
        "Çözelti", "Çözücü", "Çözünen", "Doygunluk", "Çözünürlük",
        "Oksidasyon", "İndirgeme", "Redoks", "Elektrokimya", "Elektroliz",
        "Organik_bileşik", "Hidrokarbon", "Alkan", "Alken", "Alkin", "Aromatik",
        "Fonksiyonel_grup", "Alkol", "Eter", "Aldehit", "Keton", "Karboksilik_asit",
        "Polimer", "Monomer", "Plastik", "Kauçuk", "Selüloz",
        "Enzim", "Protein", "Karbonhidrat", "Lipit", "Nükleik_asit"
    ],
    "saglik": [
        "Sağlık", "Tıp", "Hastalık", "Tedavi", "İlaç", "Cerrahi",
        "Anatomi", "Fizyoloji", "Patoloji", "Farmakoloji", "Hijyen",
        "Beslenme", "Egzersiz", "Mental_sağlık", "Halk_sağlığı"
    ],
    "kitalar": [
        "Kıta", "Asya", "Avrupa", "Afrika", "Kuzey_Amerika", "Güney_Amerika",
        "Antarktika", "Okyanusya", "Avustralya", "Coğrafya", "Yeryüzü",
        "Kara_parçası", "Kıtalar_arası", "Levha_tektoniği"
    ],
    "uzay": [
        "Uzay", "Uzay_araştırmaları", "Uzay_keşfi", "Uzay_teknolojisi",
        "NASA", "ESA", "Roscosmos", "JAXA", "CNSA", "ISRO",
        "Roket", "Roket_motoru", "Yakıt", "İtki", "Yörünge", "Fırlatma",
        "Uydu", "Yapay_uydu", "İletişim_uydu", "Gözlem_uydu", "GPS",
        "Uzay_istasyonu", "Uluslararası_Uzay_İstasyonu", "Mir", "Skylab",
        "Uzay_mekiği", "Uzay_aracı", "Uzay_gemisi", "Uzay_aracı_tasarımı",
        "Astronot", "Kozmonot", "Uzay_yürüyüşü", "Uzay_giysisi",
        "Ay", "Ay_keşfi", "Apollo_programı", "Ay_taşı", "Ay_yüzeyi",
        "Mars", "Mars_keşfi", "Mars_rover", "Mars_insanlı_keşif", "Mars_kolonileşme",
        "Gezegen_keşfi", "Venüs", "Jüpiter", "Satürn", "Uranüs", "Neptün",
        "Asteroit", "Kuyruklu_yıldız", "Meteor", "Göktaşı",
        "Uzay_teleskobu", "Hubble", "James_Webb", "Kepler", "TESS",
        "Uzay_haberleşme", "Derin_uzay_haberleşme", "Radyo_teleskop",
        "Uzay_çöplüğü", "Uzay_çevre_koruma", "Uzay_hukuku"
    ],
    "astronomi": [
        "Astronomi", "Gözlemsel_astronomi", "Teorik_astronomi", "Radyo_astronomi",
        "Güneş_sistemi", "Güneş", "Güneş_fiziği", "Güneş_rüzgarı", "Güneş_patlaması",
        "Merkür", "Venüs", "Dünya", "Mars", "Jüpiter", "Satürn", "Uranüs", "Neptün", "Plüton",
        "Gezegen", "İç_gezegen", "Dış_gezegen", "Gaz_devi", "Buz_devi", "Cüce_gezegen",
        "Uydu", "Ay", "Io", "Europa", "Ganymede", "Callisto", "Titan", "Enceladus",
        "Asteroit_kuşağı", "Kuiper_kuşağı", "Oort_bulutu", "Kuyruklu_yıldız",
        "Yıldız", "Yıldız_oluşumu", "Yıldız_evrimi", "Yıldız_türleri", "Yıldız_sınıflandırması",
        "Güneş", "Kırmızı_dev", "Beyaz_cüce", "Nötron_yıldızı", "Pulsar", "Kara_delik",
        "Galaksi", "Gökada", "Gökada_türleri", "Sarmal_galaksi", "Eliptik_galaksi",
        "Samanyolu", "Andromeda", "Galaksi_kümeleri", "Süperkümeler",
        "Evren", "Kozmoloji", "Big_Bang", "Kozmik_mikrodalga_arka_plan_ışıması",
        "Kara_delik", "Olay_ufku", "Hawking_ışıması", "Kara_delik_merkezi",
        "Nebula", "Yıldız_oluşum_bölgesi", "Gezegenimsi_bulutsu", "Süpernova_kalıntısı",
        "Yıldız_kümeleri", "Açık_küme", "Küresel_küme",
        "Karanlık_madde", "Karanlık_enerji", "Kozmik_ışın", "Gama_ışın_patlaması",
        "Exoplanet", "Yaşanabilir_bölge", "Dünya_dışı_yaşam", "SETI"
    ],
    "fizik": [
        "Fizik", "Klasik_fizik", "Modern_fizik", "Kuantum_fiziği", "Kuantum_mekaniği",
        "Görelilik", "Özel_görelilik", "Genel_görelilik", "Einstein", "Newton",
        "Maxwell", "Planck", "Bohr", "Heisenberg", "Schrödinger", "Dirac",
        "Elektromanyetizma", "Elektrik", "Manyetizma", "Elektromanyetik_dalga",
        "Termodinamik", "Entropi", "Isı", "Sıcaklık", "Enerji", "İş",
        "Mekanik", "Klasik_mekanik", "Kuantum_mekaniği", "İstatistiksel_mekanik",
        "Optik", "Işık", "Foton", "Lazer", "Spektroskopi", "Girişim",
        "Akustik", "Ses", "Dalga", "Frekans", "Rezonans",
        "Nükleer_fizik", "Atom_fiziği", "Parçacık_fiziği", "Standart_model",
        "Kuvvet", "Kütleçekim", "Elektromanyetik_kuvvet", "Güçlü_nükleer_kuvvet", "Zayıf_nükleer_kuvvet",
        "Hareket", "Hız", "İvme", "Momentum", "Açısal_momentum",
        "Dalga", "Parçacık", "Dalga-parçacık_ikiliği", "Belirsizlik_ilkesi",
        "Fizik_kanunları", "Newton_kanunları", "Termodinamik_kanunları", "Korunum_kanunları",
        "Enerji", "Kinetik_enerji", "Potansiyel_enerji", "Enerji_korunumu",
        "Alan", "Elektrik_alanı", "Manyetik_alan", "Kütleçekim_alanı",
        "Parçacık", "Elektron", "Proton", "Nötron", "Kuark", "Lepton", "Bozon"
    ],
    "biyoloji": [
        "Biyoloji", "Genel_biyoloji", "Moleküler_biyoloji", "Hücre_biyolojisi",
        "Hücre", "Hücre_zarı", "Sitoplazma", "Nükleus", "Mitokondri", "Ribozom",
        "DNA", "RNA", "mRNA", "tRNA", "rRNA", "Gen", "Genom", "Kromozom",
        "Genetik", "Kalıtım", "Mendel", "Genotip", "Fenotip", "Mutasyon",
        "Evrim", "Doğal_seçilim", "Darwin", "Lamarck", "Fosil", "Türleşme",
        "Tür", "Taksonomi", "Sınıflandırma", "Filogeni", "Evrimsel_akrabalık",
        "Ekosistem", "Biyosfer", "Habitat", "Niş", "Besin_zinciri", "Enerji_akışı",
        "Bitki", "Fotosentez", "Klorofil", "Kök", "Gövde", "Yaprak", "Çiçek",
        "Hayvan", "Omurgalı", "Omurgasız", "Memeli", "Kuş", "Balık", "Sürüngen",
        "İnsan", "Anatomi", "Fizyoloji", "Sistem", "Dolaşım_sistemi", "Solunum_sistemi",
        "Botanik", "Zooloji", "Mikrobiyoloji", "Bakteri", "Virüs", "Mantar",
        "İmmünoloji", "Bağışıklık", "Antikor", "Aşı", "Enfeksiyon",
        "Nörobiyoloji", "Sinir_sistemi", "Nöron", "Beyin", "Sinaps",
        "Endokrinoloji", "Hormon", "Metabolizma", "Enzim", "Homeostaz"
    ],
    "cografya": [
        "Coğrafya", "Dünya", "Kıta", "Okyanus", "Deniz", "Dağ", "Nehir",
        "Göl", "Çöl", "Orman", "İklim", "Hava_durumu", "Jeoloji",
        "Harita", "Enlem", "Boylam", "Ekvator", "Kutup"
    ],
    "ekonomi": [
        "Ekonomi", "İktisat", "Para", "Para_birimi", "Enflasyon",
        "Faiz", "Borsa", "Ticaret", "İhracat", "İthalat", "GSYİH",
        "İşsizlik", "Kalkınma", "Küreselleşme", "Piyasa"
    ],
    "felsefe": [
        "Felsefe", "Filozof", "Platon", "Aristoteles", "Descartes",
        "Kant", "Nietzsche", "Etik", "Metafizik", "Epistemoloji",
        "Mantık", "Estetik", "Siyaset_felsefesi", "Din_felsefesi"
    ],
    "psikoloji": [
        "Psikoloji", "Zihin", "Beyin", "Davranış", "Biliş", "Duygu",
        "Hafıza", "Öğrenme", "Kişilik", "Gelişim_psikolojisi",
        "Sosyal_psikoloji", "Klinik_psikoloji", "Nöropsikoloji"
    ],
    "sosyoloji": [
        "Sosyoloji", "Toplum", "Kültür", "Sosyal_yapı", "Sosyal_değişim",
        "Sınıf", "Toplumsal_cinsiyet", "Etnisite", "Din", "Eğitim",
        "Aile", "İş", "Kent", "Kırsal", "Göç"
    ],
    "matematik": [
        "Matematik", "Sayı", "Geometri", "Cebir", "Analiz", "İstatistik",
        "Olasılık", "Trigonometri", "Kalkülüs", "Lineer_cebir",
        "Topoloji", "Sayı_teorisi", "Matematiksel_kanıt"
    ],
    "teknoloji": [
        "Teknoloji", "Bilgisayar", "Yazılım", "Donanım", "İnternet",
        "Yapay_zeka", "Makine_öğrenmesi", "Robotik", "Nanoteknoloji",
        "Biyoteknoloji", "Telekomünikasyon", "Elektronik", "Mühendislik"
    ],
    "muzik": [
        "Müzik", "Nota", "Enstrüman", "Orkestra", "Klasik_müzik",
        "Caz", "Rock", "Pop", "Türk_müziği", "Halk_müziği",
        "Opera", "Bale", "Koro", "Solist", "Bestekar"
    ],
    "spor": [
        "Spor", "Futbol", "Basketbol", "Voleybol", "Tenis", "Yüzme",
        "Atletizm", "Olimpiyat", "Dünya_kupası", "Şampiyonluk",
        "Antrenman", "Fitness", "Sağlık", "Rekabet"
    ],
    "icatlar": [
        "İcat", "Buluş", "İnovasyon", "Teknoloji", "Mucit", "Patent",
        "Televizyon", "Radyo", "Telefon", "Telefon", "İnternet", "Bilgisayar",
        "Elektrik", "Ampul", "Pil", "Batarya", "Jeneratör", "Motor",
        "Otomobil", "Uçak", "Tren", "Gemi", "Bisiklet", "Motorsiklet",
        "Yazı", "Matbaa", "Kağıt", "Mürekkep", "Kalem", "Daktilo",
        "Tıp_icatları", "Röntgen", "MR", "CT", "Ultrason", "EKG",
        "Tarım_icatları", "Traktör", "Biçerdöver", "Sulama", "Gübre",
        "İletişim_icatları", "Telgraf", "Faks", "E-posta", "SMS", "WhatsApp",
        "Enerji_icatları", "Güneş_paneli", "Rüzgar_türbini", "Hidroelektrik",
        "Uzay_icatları", "Roket", "Uydu", "Uzay_teleskobu", "Mars_rover"
    ],
    "elektrik": [
        "Elektrik", "Elektrik_akımı", "Voltaj", "Direnç", "Güç", "Enerji",
        "Elektrik_devresi", "Seri_bağlantı", "Paralel_bağlantı", "Ohm_kanunu",
        "Elektrik_alanı", "Elektrik_yükü", "Elektron", "Proton", "İyon",
        "Elektromanyetizma", "Manyetik_alan", "Elektromanyetik_kuvvet",
        "Jeneratör", "Motor", "Transformatör", "Alternatör", "Dinamo",
        "Pil", "Batarya", "Akü", "Şarj", "Deşarj", "Elektroliz",
        "Elektrik_üretimi", "Termik_santral", "Hidroelektrik", "Nükleer_santral",
        "Elektrik_iletimi", "Yüksek_gerilim", "Elektrik_şebekesi", "Dağıtım",
        "Elektrik_güvenliği", "Topraklama", "Sigorta", "Kesici",
        "Elektronik", "Yarı_iletken", "Transistör", "Diyot", "Entegre_devre"
    ],
    "elektronik": [
        "Elektronik", "Elektronik_devre", "Yarı_iletken", "Transistör", "Diyot",
        "Entegre_devre", "Mikroçip", "Mikroişlemci", "Bellek", "RAM", "ROM",
        "Dijital_elektronik", "Analog_elektronik", "Sinyal_işleme", "Filtre",
        "Amplifikatör", "Osilatör", "Multivibratör", "Flip-flop",
        "Mikrodenetleyici", "Arduino", "Raspberry_Pi", "Gömülü_sistem",
        "Sensör", "Aktüatör", "Transdüser", "Encoder", "Decoder",
        "İletişim", "Modülasyon", "Demodülasyon", "Anten", "Alıcı", "Verici",
        "Güç_elektroniği", "Güç_kaynağı", "Regülatör", "Dönüştürücü",
        "Elektronik_cihaz", "Bilgisayar", "Telefon", "Tablet", "Televizyon",
        "Robotik", "Otomasyon", "Kontrol_sistemi", "PLC"
    ],
    "kuantum_fizigi": [
        "Kuantum_fiziği", "Kuantum_mekaniği", "Kuantum_teorisi", "Kuantum_kuramı",
        "Planck", "Einstein", "Bohr", "Heisenberg", "Schrödinger", "Dirac",
        "Kuantum", "Foton", "Elektron", "Parçacık", "Dalga", "Dalga-parçacık_ikiliği",
        "Belirsizlik_ilkesi", "Heisenberg_belirsizlik_ilkesi", "Gözlem_etkisi",
        "Schrödinger_denklemi", "Dalga_fonksiyonu", "Kuantum_durumu", "Süperpozisyon",
        "Dolanıklık", "Kuantum_dolanıklığı", "EPR_paradoksu", "Bell_eşitsizliği",
        "Kuantum_tünelleme", "Kuantum_sıçrama", "Enerji_seviyesi", "Orbital",
        "Spin", "Açısal_momentum", "Manyetik_moment", "Pauli_dışlama_ilkesi",
        "Kuantum_alan_kuramı", "Standart_model", "Kuantum_elektrodinamiği",
        "Kuantum_kromodinamiği", "Parçacık_fiziği", "Kuark", "Lepton", "Bozon",
        "Kuantum_bilgisayar", "Kubit", "Kuantum_algoritma", "Kuantum_şifreleme",
        "Kuantum_optik", "Lazer", "Fotonik", "Kuantum_iletişim",
        "Kuantum_ölçüm", "Gözlem", "Çöküş", "Kuantum_zıplama"
    ],
    "fizik_kanunlari": [
        "Fizik_kanunları", "Doğa_kanunları", "Fizik_ilkesi", "Fizik_teorisi",
        "Newton_kanunları", "Hareket_kanunları", "Kütleçekim_kanunu",
        "Termodinamik_kanunları", "Sıfırıncı_kanun", "Birinci_kanun", "İkinci_kanun", "Üçüncü_kanun",
        "Enerji_korunumu", "Momentum_korunumu", "Açısal_momentum_korunumu",
        "Yük_korunumu", "Lepton_sayısı_korunumu", "Baryon_sayısı_korunumu",
        "Ohm_kanunu", "Kirchhoff_kanunları", "Faraday_kanunu", "Lenz_kanunu",
        "Maxwell_denklemleri", "Elektromanyetizma", "Elektromanyetik_dalga",
        "Einstein_denklemleri", "Görelilik", "Özel_görelilik", "Genel_görelilik",
        "Kuantum_kanunları", "Schrödinger_denklemi", "Heisenberg_belirsizlik_ilkesi",
        "Pauli_dışlama_ilkesi", "Fermi-Dirac_istatistiği", "Bose-Einstein_istatistiği",
        "Stefan-Boltzmann_kanunu", "Wien_kanunu", "Planck_kanunu",
        "Hareket_denklemleri", "Lagrange_denklemleri", "Hamilton_denklemleri"
    ],
    "bilim_insanlari": [
        "Bilim_insanı", "Fizikçi", "Kimyager", "Biyolog", "Matematikçi", "Astronom",
        "Einstein", "Newton", "Galileo", "Kepler", "Copernicus", "Darwin",
        "Curie", "Marie_Curie", "Pierre_Curie", "Rutherford", "Bohr", "Planck",
        "Maxwell", "Faraday", "Tesla", "Edison", "Bell", "Watt",
        "Pasteur", "Fleming", "Koch", "Lister", "Harvey", "Vesalius",
        "Mendel", "Watson", "Crick", "Franklin", "McClintock", "Goodall",
        "Hawking", "Feynman", "Dirac", "Heisenberg", "Schrödinger", "Pauli",
        "Lavoisier", "Mendeleev", "Dalton", "Avogadro", "Arrhenius",
        "Türk_bilim_insanları", "Cahit_Arf", "Aziz_Sancar", "Feza_Gürsey"
    ],
    "mucitler": [
        "Mucit", "Buluş", "İcat", "Patent", "İnovasyon",
        "Edison", "Tesla", "Bell", "Watt", "Steam_engine", "Buhar_makinesi",
        "Wright_kardeşler", "Uçak", "Otomobil", "Ford", "Benz",
        "Gutenberg", "Matbaa", "Yazı", "Kağıt", "Çin",
        "Einstein", "Görelilik", "Planck", "Kuantum", "Bohr",
        "Curie", "Radyoaktivite", "Röntgen", "X-ışını", "Fleming", "Penisilin",
        "Pasteur", "Pastörizasyon", "Koch", "Bakteri", "Lister", "Antiseptik",
        "Marconi", "Radyo", "Baird", "Televizyon", "Zuse", "Bilgisayar",
        "Berners-Lee", "World_Wide_Web", "İnternet", "Gates", "Microsoft",
        "Jobs", "Apple", "Musk", "SpaceX", "Tesla_Motors"
    ]
}

# ============================================================================
# KONU BAZLI KATEGORİLER (1000+ Sayfa İçin Genişletilmiş)
# ============================================================================
# Her konu için Wikipedia kategorileri
# Deep mod: Kategori başına 200 sayfa
# 1000 sayfa için: 5+ kategori yeterli
# Her konu için 10-20 kategori eklenmiştir (2000-4000 sayfa potansiyeli)
# ============================================================================

TOPIC_CATEGORIES = {
    "tarih": [
        # Ana kategoriler
        "Tarih", "Türk_tarihi", "Dünya_tarihi",
        # Dönemler
        "Antik_çağ", "Orta_çağ", "Yeni_çağ", "Yakın_çağ",
        # Bölgesel tarih
        "Anadolu_tarihi", "Avrupa_tarihi", "Asya_tarihi", "Afrika_tarihi",
        "Amerika_tarihi", "Ortadoğu_tarihi",
        # Özel konular
        "Savaş_tarihi", "Siyaset_tarihi", "Ekonomi_tarihi", "Kültür_tarihi",
        "Sanat_tarihi", "Edebiyat_tarihi", "Bilim_tarihi", "Teknoloji_tarihi",
        # İmparatorluklar
        "Osmanlı_İmparatorluğu", "Roma_İmparatorluğu", "Bizans_İmparatorluğu",
        "Selçuklu_İmparatorluğu", "Moğol_İmparatorluğu"
    ],
    "islam_tarihi": [
        "İslam_tarihi", "İslam", "İslam_bilim",
        # Dönemler
        "Dört_Halife_dönemi", "Emeviler", "Abbasiler", "Osmanlı_İmparatorluğu",
        # Bölgeler
        "Endülüs", "Anadolu_İslam_tarihi", "Ortadoğu_İslam_tarihi",
        # Konular
        "İslam_felsefesi", "İslam_edebiyatı", "İslam_sanatı", "İslam_mimarlığı",
        "İslam_hukuku", "İslam_teolojisi", "Tasavvuf", "İslam_bilim_insanları"
    ],
    "dinler_tarihi": [
        "Din", "Dinler_tarihi",
        # Dinler
        "İslam", "Hristiyanlık", "Yahudilik", "Budizm", "Hinduizm",
        # Konular
        "Din_felsefesi", "Din_sosyolojisi", "Din_antropolojisi", "Din_psikolojisi"
    ],
    "dunya_tarihi": [
        "Dünya_tarihi", "Tarih",
        # Dönemler
        "Antik_çağ", "Orta_çağ", "Yeni_çağ", "Yakın_çağ",
        # Olaylar
        "Savaşlar", "Devrimler", "Keşifler", "Göçler"
    ],
    "hastaliklar_tarihi": [
        "Hastalık", "Tıp_tarihi", "Pandemi",
        # Hastalık türleri
        "Bulaşıcı_hastalıklar", "Kronik_hastalıklar", "Genetik_hastalıklar",
        # Tarihsel salgınlar
        "Veba", "Kolera", "Grip", "COVID-19"
    ],
    "bilim_tarihi": [
        "Bilim_tarihi", "Bilim",
        # Bilim dalları
        "Fizik_tarihi", "Kimya_tarihi", "Biyoloji_tarihi", "Matematik_tarihi",
        "Astronomi_tarihi", "Tıp_tarihi", "Teknoloji_tarihi"
    ],
    "edebiyat_tarihi": [
        "Edebiyat_tarihi", "Türk_edebiyatı",
        # Dönemler
        "Klasik_edebiyat", "Modern_edebiyat", "Çağdaş_edebiyat",
        # Türler
        "Şiir", "Roman", "Hikâye", "Tiyatro", "Deneme"
    ],
    "sanat_tarihi": [
        "Sanat_tarihi", "Sanat",
        # Sanat dalları
        "Resim", "Heykel", "Mimarlık", "Müzik", "Tiyatro", "Sinema",
        # Dönemler
        "Klasik_sanat", "Modern_sanat", "Çağdaş_sanat"
    ],
    "kimya": [
        "Kimya", "Organik_kimya", "İnorganik_kimya", "Fiziksel_kimya",
        "Analitik_kimya", "Biyokimya", "Kuantum_kimyası", "Teorik_kimya",
        "Kimyasal_tepkime", "Bağ", "Element", "Bileşik", "Molekül", "Atom",
        # Alt kategoriler
        "Kimyasal_maddeler", "Kimyasal_süreçler", "Kimyasal_araçlar",
        "Kimyasal_endüstri", "Kimyasal_güvenlik", "Kimyasal_analiz"
    ],
    "saglik": [
        "Sağlık", "Tıp", "Hastalık", "Halk_sağlığı",
        # Tıp dalları
        "İç_hasalıkları", "Cerrahi", "Pediatri", "Jinekoloji", "Kardiyoloji",
        "Nöroloji", "Onkoloji", "Psikiyatri", "Dermatoloji", "Ortopedi"
    ],
    "kitalar": [
        "Kıta", "Coğrafya", "Yeryüzü",
        # Kıtalar
        "Asya", "Avrupa", "Afrika", "Kuzey_Amerika", "Güney_Amerika",
        "Antarktika", "Okyanusya"
    ],
    "uzay": [
        "Uzay", "Uzay_araştırmaları", "Uzay_teknolojisi", "Roket", "Uydu",
        "Uzay_istasyonu", "Uzay_mekiği", "Astronot", "Mars_keşfi", "Ay_keşfi",
        # Alt kategoriler
        "Uzay_araçları", "Uzay_misyonları", "Uzay_ajansları", "Uzay_teknolojileri",
        "Gezegen_keşfi", "Asteroit_keşfi", "Uzay_teleskopları"
    ],
    "astronomi": [
        "Astronomi", "Güneş_sistemi", "Gezegen", "Yıldız", "Galaksi", "Gökada",
        "Evren", "Kozmoloji", "Big_Bang", "Kara_delik", "Nötron_yıldızı",
        "Nebula", "Yıldız_kümeleri", "Karanlık_madde", "Karanlık_enerji",
        # Alt kategoriler
        "Gezegenler", "Uydular", "Asteroitler", "Kuyruklu_yıldızlar",
        "Yıldız_türleri", "Galaksi_türleri", "Kozmik_olaylar"
    ],
    "fizik": [
        "Fizik", "Klasik_fizik", "Modern_fizik", "Kuantum_fiziği", "Kuantum_mekaniği",
        "Görelilik", "Özel_görelilik", "Genel_görelilik", "Elektromanyetizma",
        "Termodinamik", "Mekanik", "Optik", "Akustik", "Nükleer_fizik", "Parçacık_fiziği",
        "Fizik_kanunları", "Enerji", "Kuvvet", "Hareket", "Dalga",
        # Alt kategoriler
        "Fizik_deneyleri", "Fizik_teorileri", "Fizik_araçları", "Fizik_uygulamaları"
    ],
    "biyoloji": [
        "Biyoloji", "Moleküler_biyoloji", "Hücre_biyolojisi", "Genetik",
        "Evrim", "Ekosistem", "Anatomi", "Fizyoloji", "Botanik", "Zooloji",
        "Mikrobiyoloji", "İmmünoloji", "Nörobiyoloji", "Endokrinoloji",
        # Alt kategoriler
        "Canlı_türleri", "Biyolojik_süreçler", "Biyolojik_sistemler", "Biyolojik_araçlar"
    ],
    "cografya": [
        "Coğrafya", "Dünya", "Jeoloji", "İklim",
        # Alt kategoriler
        "Fiziki_coğrafya", "Beşeri_coğrafya", "Ekonomik_coğrafya", "Siyasi_coğrafya",
        "Kıtalar", "Okyanuslar", "Dağlar", "Nehirler", "Göller", "Çöller"
    ],
    "ekonomi": [
        "Ekonomi", "İktisat", "Ticaret", "Piyasa",
        # Alt kategoriler
        "Makroekonomi", "Mikroekonomi", "Uluslararası_ekonomi", "Finans",
        "Bankacılık", "Borsa", "Para_politikası", "Maliye_politikası"
    ],
    "felsefe": [
        "Felsefe", "Etik", "Metafizik", "Epistemoloji",
        # Alt kategoriler
        "Felsefe_akımları", "Filozoflar", "Felsefe_tarihi", "Felsefe_dalları",
        "Mantık", "Estetik", "Siyaset_felsefesi", "Din_felsefesi"
    ],
    "psikoloji": [
        "Psikoloji", "Zihin", "Davranış", "Biliş",
        # Alt kategoriler
        "Psikoloji_akımları", "Psikoloji_dalları", "Psikologlar", "Psikolojik_bozukluklar",
        "Bilişsel_psikoloji", "Sosyal_psikoloji", "Klinik_psikoloji", "Gelişim_psikolojisi"
    ],
    "sosyoloji": [
        "Sosyoloji", "Toplum", "Kültür", "Sosyal_yapı",
        # Alt kategoriler
        "Sosyoloji_akımları", "Sosyologlar", "Sosyal_olaylar", "Sosyal_değişim",
        "Toplumsal_sınıflar", "Kültürel_antropoloji", "Sosyal_psikoloji"
    ],
    "matematik": [
        "Matematik", "Geometri", "Cebir", "Analiz",
        # Alt kategoriler
        "Matematik_dalları", "Matematikçiler", "Matematik_teorileri", "Matematik_uygulamaları",
        "Sayı_teorisi", "Topoloji", "İstatistik", "Olasılık"
    ],
    "teknoloji": [
        "Teknoloji", "Bilgisayar", "Yapay_zeka", "Mühendislik",
        # Alt kategoriler
        "Bilgisayar_bilimi", "Yazılım_geliştirme", "Donanım", "Ağ_teknolojileri",
        "Yapay_zeka_uygulamaları", "Robotik", "Nanoteknoloji", "Biyoteknoloji"
    ],
    "muzik": [
        "Müzik", "Enstrüman", "Klasik_müzik", "Türk_müziği",
        # Alt kategoriler
        "Müzik_türleri", "Müzisyenler", "Müzik_teorisi", "Müzik_enstrümanları",
        "Opera", "Bale", "Orkestra", "Koro"
    ],
    "spor": [
        "Spor", "Futbol", "Olimpiyat", "Atletizm",
        # Alt kategoriler
        "Spor_dalları", "Sporcular", "Spor_organizasyonları", "Spor_tarihi",
        "Futbol", "Basketbol", "Voleybol", "Tenis", "Yüzme", "Atletizm"
    ],
    "icatlar": [
        "İcat", "Buluş", "İnovasyon", "Teknoloji", "Mucit", "Patent",
        "Elektrik", "Elektronik", "Uzay", "Tıp", "İletişim", "Enerji",
        # Alt kategoriler
        "Tarihsel_icatlar", "Modern_icatlar", "Mucitler", "İnovasyon_alanları"
    ],
    "elektrik": [
        "Elektrik", "Elektrik_akımı", "Voltaj", "Direnç", "Elektromanyetizma",
        "Jeneratör", "Motor", "Pil", "Batarya", "Elektrik_üretimi", "Elektrik_iletimi",
        # Alt kategoriler
        "Elektrik_sistemleri", "Elektrik_araçları", "Elektrik_güvenliği", "Elektrik_endüstrisi"
    ],
    "elektronik": [
        "Elektronik", "Yarı_iletken", "Transistör", "Entegre_devre", "Mikroçip",
        "Dijital_elektronik", "Analog_elektronik", "Mikrodenetleyici", "Sensör",
        # Alt kategoriler
        "Elektronik_cihazlar", "Elektronik_araçlar", "Elektronik_endüstrisi", "Elektronik_uygulamaları"
    ],
    "kuantum_fizigi": [
        "Kuantum_fiziği", "Kuantum_mekaniği", "Kuantum_teorisi",
        "Belirsizlik_ilkesi", "Schrödinger_denklemi", "Dolanıklık",
        "Kuantum_bilgisayar", "Kubit", "Kuantum_optik",
        # Alt kategoriler
        "Kuantum_teorileri", "Kuantum_deneyleri", "Kuantum_uygulamaları", "Kuantum_araçları"
    ],
    "fizik_kanunlari": [
        "Fizik_kanunları", "Newton_kanunları", "Termodinamik_kanunları",
        "Enerji_korunumu", "Maxwell_denklemleri", "Einstein_denklemleri",
        "Kuantum_kanunları", "Hareket_denklemleri",
        # Alt kategoriler
        "Klasik_fizik_kanunları", "Modern_fizik_kanunları", "Kuantum_kanunları", "Termodinamik_kanunları"
    ],
    "bilim_insanlari": [
        "Bilim_insanı", "Fizikçi", "Kimyager", "Biyolog", "Matematikçi",
        "Einstein", "Newton", "Darwin", "Curie", "Tesla", "Hawking",
        # Alt kategoriler
        "Fizikçiler", "Kimyagerler", "Biyologlar", "Matematikçiler", "Astronomlar", "Türk_bilim_insanları"
    ],
    "mucitler": [
        "Mucit", "Buluş", "İcat", "Patent", "Edison", "Tesla", "Bell",
        "Wright_kardeşler", "Gutenberg", "Berners-Lee", "Musk",
        # Alt kategoriler
        "Tarihsel_mucitler", "Modern_mucitler", "İcat_alanları", "Patent_sistemleri"
    ]
}

# ============================================================================
# YARDIMCI FONKSİYONLAR
# ============================================================================

def get_all_topics() -> list:
    """Tüm mevcut konuları döndürür."""
    return list(TOPIC_PAGES.keys())

def get_topic_pages(topic: str) -> list:
    """Belirli bir konu için önceden tanımlı sayfa listesini döndürür."""
    return TOPIC_PAGES.get(topic, [])

def get_topic_categories(topic: str) -> list:
    """Belirli bir konu için kategori listesini döndürür."""
    return TOPIC_CATEGORIES.get(topic, [])

def add_topic(topic_name: str, pages: list, categories: list) -> None:
    """
    Yeni bir konu ekler.
    
    Args:
        topic_name: Konu adı
        pages: Önceden tanımlı sayfa listesi
        categories: Kategori listesi
    """
    TOPIC_PAGES[topic_name] = pages
    TOPIC_CATEGORIES[topic_name] = categories

def add_pages_to_topic(topic_name: str, pages: list) -> None:
    """Mevcut bir konuya yeni sayfalar ekler."""
    if topic_name in TOPIC_PAGES:
        TOPIC_PAGES[topic_name].extend(pages)
    else:
        TOPIC_PAGES[topic_name] = pages

def add_categories_to_topic(topic_name: str, categories: list) -> None:
    """Mevcut bir konuya yeni kategoriler ekler."""
    if topic_name in TOPIC_CATEGORIES:
        TOPIC_CATEGORIES[topic_name].extend(categories)
    else:
        TOPIC_CATEGORIES[topic_name] = categories

# ============================================================================
# TOKEN HESAPLAMA BİLGİLERİ
# ============================================================================
"""
TOKEN HESAPLAMA:
- Ortalama Wikipedia sayfası: 2000-5000 token (Türkçe)
- 1000 sayfa x 2000 token = 2 milyon token (minimum)
- 1000 sayfa x 5000 token = 5 milyon token (maksimum)
- 20 konu x 2-5 milyon = 40-100 milyon token (toplam)
- Milyarlarca token için: 200+ konu veya çok daha fazla sayfa gerekir

GÜNLÜK KONUŞMA DİLİ:
- 14 milyon token + altyazı verisi = İyi bir başlangıç
- Altyazı verisi günlük konuşma için ÇOK ÖNEMLİ
- Wikipedia daha çok formal/bilimsel dil içerir
- Altyazı verisi informal/günlük konuşma dilini sağlar
- Önerilen: 14M Wikipedia + 5-10M altyazı = 19-24M token (yeterli)
- Daha fazla veri = Daha iyi performans (ama azalan getiri yasası geçerli)
"""
