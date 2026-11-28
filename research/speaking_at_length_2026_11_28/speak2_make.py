# -*- coding: utf-8 -*-
"""A second set for speaking at length, written and frozen after the first candidate failed and before the second
was run on it (28 November 2026): two things the first set does not hold (a tent, a torch), each named once in a
sentence told before its text, so that "the first sentence that names it" and "the first sentence of its text"
differ; and two requests in no shown form.

    python speak2_make.py         # writes speak2.json and speak2.sha256

The life is the first set's life (lessons, four texts, the eighteen dialogues shown) with two sentences told and
two texts read before the dialogues. Keys as in speak_make.py.
"""
import hashlib, json, os
import speak_make as S

HERE = os.path.dirname(os.path.abspath(__file__))
MENTIONS = ["kampçılar geceyi çadırda geçirir .", "madenciler karanlık tünelde fener taşır ."]
TEXTS = {
 "çadır": [
  ("what", "çadır kumaştan yapılan taşınabilir bir barınaktır ."),
  ("what", "çadır kamp yapanları yağmurdan ve rüzgârdan korur ."),
  ("parts", "çadır direkler , dış kumaş , zemin örtüsü , kazıklar ve gergi iplerinden oluşur ."),
  ("parts", "direkler çadırın iskeletidir ."),
  ("parts", "kazıklar çadırı toprağa bağlar ."),
  ("works", "gergi ipleri gerildiğinde kumaş düzleşir ve çadır rüzgâra karşı dayanır ."),
  ("works", "çadırın tasarımında kubbe biçimi rüzgârın yükünü dağıtır ."),
  ("works", "eğimli yüzey yağmur suyunun akıp gitmesini sağlar ."),
  ("made", "dış kumaş su geçirmeyen naylon ya da polyesterden yapılır ."),
  ("made", "direkler hafif alüminyumdan ya da cam elyafından üretilir ."),
  ("made", "kazıklar çelikten yapılır ."),
  ("care", "çadır kuruyken katlanmalıdır ."),
  ("care", "fermuarlar zorlanmadan açılıp kapatılmalıdır ."),
 ],
 "fener": [
  ("what", "el feneri karanlıkta yolu aydınlatan taşınabilir bir ışık kaynağıdır ."),
  ("what", "el feneri elektriğini pillerden alır ."),
  ("parts", "el feneri gövde , piller , ampul , yansıtıcı , cam ve düğmeden oluşur ."),
  ("parts", "düğme fenerin açılıp kapanmasını sağlayan parçadır ."),
  ("parts", "yansıtıcı ampulün arkasında durur ."),
  ("works", "düğmeye basıldığında devre kapanır ve piller ampule akım gönderir ."),
  ("works", "yansıtıcı ışığı toplar ve ileriye doğru yöneltir ."),
  ("works", "fenerin tasarımında yansıtıcının eğriliği ışık demetinin genişliğini belirler ."),
  ("made", "gövde alüminyumdan ya da sert plastikten yapılır ."),
  ("made", "yansıtıcı parlatılmış metalden üretilir ."),
  ("made", "öndeki cam çizilmeye dayanıklı sertleştirilmiş camdan yapılır ."),
  ("care", "piller uzun süre kullanılmayacaksa çıkarılmalıdır ."),
  ("care", "fener nemli yerde saklanmamalıdır ."),
 ],
}
FORMS = {"çadır": ("çadır", "çadırı", "çadırın"), "fener": ("fener", "feneri", "fenerin")}
FREE = {"çadır": ("çadır nelerden oluşur ve neyden yapılır", ("parts", "made")),
        "fener": ("fenerin parçaları nelerdir tasarımı nasıldır hangi malzemeden üretilir", ("parts", "works", "made"))}


def main():
    first = json.load(open(os.path.join(HERE, "speak.json"), encoding="utf-8"))
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
    out = {"what": "speaking at length, the second set", "lessons": first["lessons"] + MENTIONS,
           "texts": dict(first["texts"], **{n: [s for _, s in t] for n, t in TEXTS.items()}),
           "order": first["order"] + list(TEXTS), "shown": first["shown"], "asked": asked}
    blob = json.dumps(out, ensure_ascii=False, indent=1).encode("utf-8")
    open(os.path.join(HERE, "speak2.json"), "wb").write(blob)
    digest = hashlib.sha256(blob).hexdigest()
    open(os.path.join(HERE, "speak2.sha256"), "w").write(digest + "  speak2.json\n")
    print(json.dumps({"asked": len(asked), "must": sum(len(a["must"]) for a in asked), "sha256": digest}))


if __name__ == "__main__":
    main()
