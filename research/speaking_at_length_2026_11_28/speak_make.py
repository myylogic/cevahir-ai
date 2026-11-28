# -*- coding: utf-8 -*-
"""The yardstick for speaking at length, written and frozen before the mechanism (28 November 2026).

    python speak_make.py          # writes speak.json and speak.sha256

A life and questions about it, graded a sentence at a time:

- ``lessons``: the 301 sentences told in the repository's 24 lessons (the owner's, written before this study), told
  one at a time, in order.
- ``texts``: four short expository texts written for this set (a clock, a guitar, a bicycle, a bridge), each read as
  one text, a sentence a line, with no paragraph breaks (so the kind of a sentence is not given by the layout). Each
  sentence carries a tag here (what, parts, works, made, care) that the mind never sees.
- ``shown``: dialogues watched. For the clock and the guitar, seven forms of request with their answers; for four
  subjects of the lessons, "X hakkında ne biliyorsun ?" answered with every told sentence that names X.
- ``asked``: the same forms about the bicycle and the bridge, which were never shown; the same request about nine
  other subjects of the lessons; and one request in no shown form (the owner's question of 28 November).

A key is three lists of told sentences: ``must`` (what the answer is), ``may`` (of the same thing, beside the
question: said, it is counted apart) and everything else told is wrong. An answer is cut into sentences at its
closing marks and each is looked up among the told sentences.
"""
import glob, hashlib, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

TEXTS = {
 "saat": [
  ("what", "saat zamanı ölçen ve gösteren bir araçtır ."),
  ("what", "mekanik saat pil kullanmadan çalışır ."),
  ("parts", "mekanik saat yay , dişliler , maşa , kadran , akrep ve yelkovandan oluşur ."),
  ("parts", "kadran saatin yüzüdür ve rakamları taşır ."),
  ("parts", "akrep saati , yelkovan dakikayı gösterir ."),
  ("works", "kurulan yay enerjiyi biriktirir ve yavaş yavaş dişlilere aktarır ."),
  ("works", "maşa dişlilerin hızını düzenler ve saatin tik tak sesini çıkarır ."),
  ("works", "saatin tasarımında dişlilerin oranı akrebin ve yelkovanın hızını belirler ."),
  ("made", "saat dişlileri pirinçten yapılır ."),
  ("made", "yay sertleştirilmiş çelikten üretilir ."),
  ("made", "kadranın önündeki cam safir ya da mineral camdan yapılır ."),
  ("care", "mekanik saat her gün kurulmalıdır ."),
  ("care", "saat sudan ve mıknatıstan uzak tutulmalıdır ."),
 ],
 "gitar": [
  ("what", "gitar telleri parmakla ya da penayla çalınan bir çalgıdır ."),
  ("what", "klasik gitarın altı teli vardır ."),
  ("parts", "gitar gövde , sap , teller , eşik ve burgulardan oluşur ."),
  ("parts", "gövde gitarın en büyük parçasıdır ve sesi büyütür ."),
  ("parts", "sap üzerinde perdeler bulunur ."),
  ("parts", "burgular tellerin gerginliğini ayarlar ."),
  ("works", "tele vurulduğunda tel titreşir ve bu titreşim gövdenin içinde yankılanır ."),
  ("works", "telin boyu kısaldıkça ses incelir ."),
  ("works", "gitarın tasarımında gövdenin biçimi sesin rengini belirler ."),
  ("made", "gövde ladin ya da sedir ağacından yapılır ."),
  ("made", "klasik gitarın telleri naylondan , akustik gitarın telleri çelikten yapılır ."),
  ("made", "sap için sert ve dayanıklı maun ağacı kullanılır ."),
  ("care", "teller eskidiğinde değiştirilmelidir ."),
  ("care", "gitar nemden ve sıcaktan korunmalıdır ."),
 ],
 "bisiklet": [
  ("what", "bisiklet iki tekerlekli ve pedalla çalışan bir taşıttır ."),
  ("what", "bisikleti süren kişinin kas gücü onu hareket ettirir ."),
  ("parts", "bisiklet kadro , tekerlekler , zincir , pedallar , gidon ve frenlerden oluşur ."),
  ("parts", "kadro bisikletin iskeletidir ve bütün parçaları bir arada tutar ."),
  ("parts", "tekerlekler jant , teller ve lastikten oluşur ."),
  ("parts", "gidon ön tekerleği yönlendirir ."),
  ("works", "pedallar çevrildiğinde zincir gücü arka tekerleğe aktarır ."),
  ("works", "vites dişlileri yokuşta daha az güçle pedal çevirmeyi sağlar ."),
  ("works", "bisikletin tasarımında ağırlık merkezi alçak tutulur ve böylece denge kolaylaşır ."),
  ("works", "kadronun üçgen biçimi yükü eşit dağıtır ve sağlamlık verir ."),
  ("made", "bisiklet kadrosu çelik , alüminyum ya da karbon fiberden yapılır ."),
  ("made", "alüminyum hafif olduğu için en çok kullanılan kadro malzemesidir ."),
  ("made", "lastikler kauçuktan yapılır ve içleri havayla doldurulur ."),
  ("made", "zincir ve dişliler aşınmaya dayanıklı sertleştirilmiş çelikten üretilir ."),
  ("care", "zincir düzenli olarak yağlanmalıdır ."),
  ("care", "frenler her sürüşten önce denetlenmelidir ."),
 ],
 "köprü": [
  ("what", "köprü bir engelin üzerinden geçmeyi sağlayan bir yapıdır ."),
  ("what", "köprüler nehirlerin , vadilerin ve yolların üzerine kurulur ."),
  ("parts", "asma köprü kuleler , ana halatlar , askı halatları ve tabliyeden oluşur ."),
  ("parts", "tabliye araçların ve yayaların üzerinden geçtiği parçadır ."),
  ("parts", "kuleler ana halatları taşır ."),
  ("works", "tabliyenin yükü askı halatlarıyla ana halatlara , oradan da kulelere aktarılır ."),
  ("works", "kemer köprüde yük kemerin iki ucundaki ayaklara dağılır ."),
  ("works", "köprünün tasarımında rüzgâr ve deprem yükleri hesaba katılır ."),
  ("made", "halatlar binlerce ince çelik telden yapılır ."),
  ("made", "kuleler ve ayaklar betonarmeden yapılır ."),
  ("made", "eski köprüler taştan ve ahşaptan yapılırdı ."),
  ("care", "çelik parçalar paslanmaya karşı düzenli olarak boyanmalıdır ."),
  ("care", "köprüler belirli aralıklarla denetlenmelidir ."),
 ],
}
ORDER = ["saat", "gitar", "bisiklet", "köprü"]
FORMS = {            # the thing as the request names it: plain, as what is told, as whose care
 "saat": ("saat", "saati", "saatin"), "gitar": ("gitar", "gitarı", "gitarın"),
 "bisiklet": ("bisiklet", "bisikleti", "bisikletin"), "köprü": ("köprü", "köprüyü", "köprünün")}
REQUESTS = [         # form, the request, the tags its answer has
 ("one sentence", "{0} nedir ?", None),
 ("parts", "{0} hangi parçalardan oluşur ?", ("parts",)),
 ("works", "{0} nasıl çalışır ?", ("works",)),
 ("made", "{0} hangi malzemelerden yapılır ?", ("made",)),
 ("care", "{2} bakımı nasıl yapılır ?", ("care",)),
 ("all", "{1} anlat", ("what", "parts", "works", "made", "care")),
 ("two kinds", "{0} hangi parçalardan oluşur ve hangi malzemelerden yapılır ?", ("parts", "made")),
]
OWNER = "bisiklet kaç parçadan oluşur parçalarının tasarım dinamiği nedir hangi malzemeler kullanılmalıdır"
LESSON_SUBJECTS = {  # the forms of each subject in what was told, chosen by reading the words of the lessons
 "toplum": "toplum toplumdur toplumu toplumudur", "kalp": "kalp kalpten", "su": "su sudan suyun",
 "kuvvet": "kuvvet kuvvete kuvvetin",
 "güneş": "güneş güneşe güneşin", "dünya": "dünya dünyanın dünyaya dünyayı", "kan": "kan kana kanda kandan kanı kanın",
 "hücre": "hücre hücreden hücreleri hücrelerin hücrelerini hücrenin hücresidir",
 "enerji": "enerji enerjinin enerjisidir enerjisiyle enerjiyi", "gezegen": "gezegen gezegendir gezegenler",
 "cevahir": "cevahir", "virüs": "virüs", "veri": "veri verinin veride veriden verisini veriyle"}
LESSONS_SHOWN = ["toplum", "kalp", "su", "kuvvet"]


def main():
    lessons = []
    for path in sorted(glob.glob(os.path.join(REPO, "research", "living_mind_2026_11_22", "dersler", "*.txt"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.strip() and not line.startswith("#") and not line.startswith("?") and "\t" not in line:
                    lessons.append(line.strip())
    told = lessons + [s for name in ORDER for _, s in TEXTS[name]]
    shown, asked = [], []
    for subject, forms in LESSON_SUBJECTS.items():
        forms = set(forms.split())
        must = [s for s in told if forms & set(re.findall(r"\w+", s.lower()))]
        item = {"world": "lessons", "thing": subject, "form": "what is known of it",
                "q": "%s hakkında ne biliyorsun ?" % subject, "must": must, "may": []}
        (shown if subject in LESSONS_SHOWN else asked).append(item)
    for name in ORDER:
        text = TEXTS[name]
        for form, request, tags in REQUESTS:
            must = [text[0][1]] if tags is None else [s for t, s in text if t in tags]
            item = {"world": "texts", "thing": name, "form": form, "q": request.format(*FORMS[name]), "must": must,
                    "may": [s for _, s in text if s not in must]}
            (shown if name in ("saat", "gitar") else asked).append(item)
    text = TEXTS["bisiklet"]
    must = [s for t, s in text if t in ("parts", "works", "made")]
    asked.append({"world": "texts", "thing": "bisiklet", "form": "in no shown form", "q": OWNER, "must": must,
                  "may": [s for _, s in text if s not in must]})
    out = {"what": "speaking at length: a life and questions, graded a sentence at a time", "lessons": lessons,
           "texts": {name: [s for _, s in TEXTS[name]] for name in ORDER}, "order": ORDER, "shown": shown, "asked": asked}
    blob = json.dumps(out, ensure_ascii=False, indent=1).encode("utf-8")
    with open(os.path.join(HERE, "speak.json"), "wb") as f:
        f.write(blob)
    digest = hashlib.sha256(blob).hexdigest()
    with open(os.path.join(HERE, "speak.sha256"), "w") as f:
        f.write(digest + "  speak.json\n")
    print(json.dumps({"told": len(told), "lessons": len(lessons), "shown": len(shown), "asked": len(asked),
                      "must_sentences_asked": sum(len(a["must"]) for a in asked), "sha256": digest}, ensure_ascii=False))
    for a in asked:
        print("  %-8s %-20s must %2d  may %2d  %s" % (a["world"], a["form"], len(a["must"]), len(a["may"]), a["q"][:70]))


if __name__ == "__main__":
    main()
