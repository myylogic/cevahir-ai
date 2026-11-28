# -*- coding: utf-8 -*-
"""A fourth set, written and frozen before the reading of a request through its own words was changed a third time
(a word never met in a request is followed only if it names fewer told sentences than the thing does): a ladder and
an umbrella, each named once in a sentence told before its text; the seven shown forms and one request in no shown
form of each. No form of the reading has been run on it. 28 November 2026.

    python speak4_make.py         # writes speak4.json and speak4.sha256

The life is the third set's with two sentences told and two texts read before the dialogues (two things shown).
Fixed before the first run on this set (speak_run.py --set 4):

 1. no wrong sentence;
 2. requests in no shown form: the mind speaks of both; at least a quarter of what they ask for is said; of what
    is said for them no more than a third is beside the question;
 3. the rest as the code before the reading (f05c10c) says it on this set: one sentence, the whole text, and for
    the kinds not fewer right and not more beside.
"""
import hashlib, json, os
import speak_make as S

HERE = os.path.dirname(os.path.abspath(__file__))
MENTIONS = ["ustalar çatıya merdivenle çıkar .", "yolcular yağmurda şemsiye açar ."]
TEXTS = {
 "merdiven": [
  ("what", "merdiven yüksek yerlere çıkmayı sağlayan bir araçtır ."),
  ("what", "katlanır merdiven kolayca taşınır ."),
  ("parts", "merdiven iki yan kol , basamaklar ve ayaklardan oluşur ."),
  ("parts", "basamaklar merdivenin üzerine basılan parçasıdır ."),
  ("parts", "ayaklar yan kolların ucuna takılır ."),
  ("works", "merdiven duvara yaslandığında ağırlık yan kollardan zemine aktarılır ."),
  ("works", "açılı durduğunda merdiven devrilmez ."),
  ("works", "merdivenin tasarımında basamakların aralığı adım boyuna göre belirlenir ."),
  ("made", "yan kollar alüminyumdan ya da ahşaptan yapılır ."),
  ("made", "basamaklar oluklu metalden üretilir ."),
  ("made", "ayaklar kaymayı önleyen kauçuktan yapılır ."),
  ("care", "merdivenin vidaları düzenli olarak sıkılmalıdır ."),
  ("care", "çatlayan basamaklar değiştirilmelidir ."),
 ],
 "şemsiye": [
  ("what", "şemsiye yağmurdan ve güneşten koruyan bir eşyadır ."),
  ("what", "katlanır şemsiye çantaya sığar ."),
  ("parts", "şemsiye sap , teller , kumaş ve sürgüden oluşur ."),
  ("parts", "kumaş şemsiyenin yağmuru tutan parçasıdır ."),
  ("parts", "sürgü sapın üzerinde kayar ."),
  ("works", "sürgü yukarı itildiğinde teller açılır ve kumaşı gerer ."),
  ("works", "gerilen kumaş suyu kenarlara akıtır ."),
  ("works", "şemsiyenin tasarımında tellerin esnekliği rüzgâra dayanıklılığı belirler ."),
  ("made", "teller çelikten ya da fiberden yapılır ."),
  ("made", "kumaş su geçirmeyen polyesterden üretilir ."),
  ("made", "sap ahşaptan ya da plastikten yapılır ."),
  ("care", "ıslak şemsiye açık bırakılarak kurutulmalıdır ."),
  ("care", "paslanan teller yağlanmalıdır ."),
 ],
}
FORMS = {"merdiven": ("merdiven", "merdiveni", "merdivenin"), "şemsiye": ("şemsiye", "şemsiyeyi", "şemsiyenin")}
FREE = {"merdiven": ("merdivenin basamakları neyden üretilir ve bakımı için ne yapılmalıdır", ("made", "care")),
        "şemsiye": ("şemsiye kaç parçadan oluşur kumaşı ve telleri hangi malzemeden yapılır", ("parts", "made"))}


def main():
    third = json.load(open(os.path.join(HERE, "speak3.json"), encoding="utf-8"))
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
    out = {"what": "speaking at length, the fourth set", "lessons": third["lessons"] + MENTIONS,
           "texts": dict(third["texts"], **{n: [s for _, s in t] for n, t in TEXTS.items()}),
           "order": third["order"] + list(TEXTS), "shown": third["shown"], "asked": asked}
    blob = json.dumps(out, ensure_ascii=False, indent=1).encode("utf-8")
    open(os.path.join(HERE, "speak4.json"), "wb").write(blob)
    digest = hashlib.sha256(blob).hexdigest()
    open(os.path.join(HERE, "speak4.sha256"), "w").write(digest + "  speak4.json\n")
    print(json.dumps({"asked": len(asked), "must": sum(len(a["must"]) for a in asked), "sha256": digest}))


if __name__ == "__main__":
    main()
