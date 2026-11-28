# -*- coding: utf-8 -*-
"""A third set for speaking at length, written and frozen before two changes to the telling grain were made (how a
word ends as a mark of a kind of sentence; a request in no known form read through the marks of the known forms):
a windmill and a pencil, each named once in a sentence told before its text; the seven shown forms and one request
in no shown form of each. 28 November 2026.

    python speak3_make.py         # writes speak3.json and speak3.sha256

The life is the second set's life with two sentences told and two texts read before the dialogues.
"""
import hashlib, json, os
import speak_make as S

HERE = os.path.dirname(os.path.abspath(__file__))
MENTIONS = ["köylüler buğdayı değirmene götürür .", "öğrenciler sınavda kalem kullanır ."]
TEXTS = {
 "değirmen": [
  ("what", "yel değirmeni rüzgârın gücüyle tahıl öğüten bir yapıdır ."),
  ("what", "yel değirmeni rüzgârlı tepelere kurulur ."),
  ("parts", "yel değirmeni kanatlar , mil , dişliler ve değirmen taşlarından oluşur ."),
  ("parts", "kanatlar değirmenin rüzgârı yakalayan parçasıdır ."),
  ("parts", "mil kanatları dişlilere bağlar ."),
  ("works", "rüzgâr estiğinde kanatlar döner ve mil bu dönüşü dişlilere aktarır ."),
  ("works", "dişliler üstteki taşı döndürür ve tahıl iki taşın arasında ezilir ."),
  ("works", "değirmenin tasarımında kanatların eğimi rüzgârdan alınan gücü belirler ."),
  ("made", "kanatlar ahşap çıtalardan ve yelken bezinden yapılır ."),
  ("made", "değirmen taşları sert kayadan yontulur ."),
  ("made", "gövde taştan ya da tuğladan yapılır ."),
  ("care", "dişliler düzenli olarak yağlanmalıdır ."),
  ("care", "yıpranan yelken bezleri değiştirilmelidir ."),
 ],
 "kalem": [
  ("what", "kurşun kalem yazı yazmaya ve çizim yapmaya yarayan bir araçtır ."),
  ("what", "kurşun kalemin yazısı silgiyle silinebilir ."),
  ("parts", "kurşun kalem ahşap gövde , grafit uç ve silgiden oluşur ."),
  ("parts", "uç kalemin kâğıda iz bırakan parçasıdır ."),
  ("parts", "silgi kalemin arka ucuna takılır ."),
  ("works", "uç kâğıda sürtüldüğünde grafit tanecikleri kâğıdın yüzeyinde kalır ."),
  ("works", "uç yumuşadıkça çizgi koyulaşır ."),
  ("works", "kalemin tasarımında altıgen gövde kalemin masadan yuvarlanmasını önler ."),
  ("made", "gövde sedir ağacından yapılır ."),
  ("made", "uç grafit ile kilin karışımından üretilir ."),
  ("made", "silgi kauçuktan yapılır ."),
  ("care", "kalemin ucu köreldiğinde açılmalıdır ."),
  ("care", "kalem yere düşürülmemelidir ."),
 ],
}
FORMS = {"değirmen": ("değirmen", "değirmeni", "değirmenin"), "kalem": ("kalem", "kalemi", "kalemin")}
FREE = {"değirmen": ("değirmen neyden yapılır ve bakımı için neler yapılmalıdır", ("made", "care")),
        "kalem": ("kalemin parçaları ve tasarımı nedir hangi malzemeden üretilir", ("parts", "works", "made"))}


def main():
    second = json.load(open(os.path.join(HERE, "speak2.json"), encoding="utf-8"))
    asked = []
    for name, text in TEXTS.items():
        for form, request, tags in S.REQUESTS:
            must = [text[0][1]] if tags is None else [s for t, s in text if t in tags]
            asked.append({"world": "texts", "thing": name, "form": form, "q": request.format(*FORMS[name]),
                          "must": must, "may": [s for _, s in text if s not in must]})
        q, tags = FREE[name]
        must = [s for t, s in text if t in tags]
        asked.append({"world": "texts", "thing": name, "form": "in no shown form", "q": q, "must": must,
                      "may": [s for _, s in text if s not in must]})
    out = {"what": "speaking at length, the third set", "lessons": second["lessons"] + MENTIONS,
           "texts": dict(second["texts"], **{n: [s for _, s in t] for n, t in TEXTS.items()}),
           "order": second["order"] + list(TEXTS), "shown": second["shown"], "asked": asked}
    blob = json.dumps(out, ensure_ascii=False, indent=1).encode("utf-8")
    open(os.path.join(HERE, "speak3.json"), "wb").write(blob)
    digest = hashlib.sha256(blob).hexdigest()
    open(os.path.join(HERE, "speak3.sha256"), "w").write(digest + "  speak3.json\n")
    print(json.dumps({"asked": len(asked), "must": sum(len(a["must"]) for a in asked), "sha256": digest}))


if __name__ == "__main__":
    main()
