# g4 — Prior-work digest: concepts (the owner's observation notes), relations (round 12), context (round 14), interactions (round 18)

Repository root: `/workspace/cevahir-ai` (read-only). All paths below are relative to that root unless absolute.
Digest written 2026-11-14 for the "new core" design (learning and running as one process; weights as graded
beliefs; concepts formed when needed; the system imagines and measures how real its constructions are; the way
of constructing improves and transfers).

---

## 0. What was read, how the documents map, integrity

**Named folders.**

| Study | Docs (`docs/research/<study>/`) | Code + results (`research/<study>/`) |
|---|---|---|
| `living_learning_concepts_2026_09_22` | `OBSERVATION_AND_HYPOTHESES_TR.md`, `CROSS_CONNECTIONS_AND_TRANSFER_TR.md` | **Does not exist.** `research/living_learning_concepts_2026_09_22` is absent; the two notes say themselves "Yeni kod veya sayısal deney yapılmadı". This is a record of the owner's observation, not an experiment. |
| `living_learning_relations_2026_09_22` (round 12) | `REPORT_TR.md`, `PROTOCOL.md`, `MATHEMATICAL_AUDIT_TR.md`, `LITERATURE_TR.md`, `NEXT_HYPOTHESES_TR.md` | `relation_learning.py`, `structural_witness.py`, `verify_relations.py`, `analyze_results.py`, `final_audit.py`, `plot_results.py`, `README.md`; `results/{relation_learning,paired_summary,structural_witness,verification,final_audit,preservation_baseline}.json`, `composition_curves.{png,svg}` |
| `living_learning_context_2026_09_22` (round 14) | `REPORT_TR.md`, `PROTOCOL.md`, `MATHEMATICAL_AUDIT_TR.md`, `ACTIVE_EXPERIENCE_PROTOCOL.md`, `CONTINUOUS_SYSTEM_TARGET_TR.md`, `THEORY_NOVELTY_BOUNDARY_TR.md` | `context_learning.py`, `active_experience.py`, `verify_context.py`, `verify_active.py`, `analyze_results.py`, `final_audit.py`, `README.md`; `results/{context_learning,active_experience,paired_summary,verification,active_verification,final_audit,preservation_baseline}.json` |
| `living_learning_interactions_2026_09_22` (round 18) | `REPORT_TR.md`, `PROTOCOL.md`, `MATHEMATICS_AND_PRIOR_ART_TR.md`, `WHOLE_PROBLEM_AND_NEXT_TEST_TR.md` | `interaction_learning.py`, `verify_interactions.py`, `analyze_results.py`, `diagnose_coverage.py`, `final_audit.py`, `README.md`; `results/{interaction_learning,summary,coverage_diagnostics,verification,final_audit,preservation_baseline}.json`. Depends on round 17: `research/living_learning_repair_2026_09_22/repair_learning.py` + its `results/repair_learning.json` (imported at `interaction_learning.py:14-18`). |

**Place in the programme.** Round 10 = geometry, 11 = library, *then the two concept notes*, 12 = relations,
13 = consolidation, 14 = context, 15 = temporal, 16 = migration, 17 = repair, 18 = interactions, 19 = distinctions,
20 = live_revision, 21 = deduction, 22 = crossmap, 23 = campaign (read off the first lines of every `REPORT*.md`).
The concept notes are the *origin* of this line of rounds: round 12's protocol and report start from them
(`docs/research/living_learning_relations_2026_09_22/PROTOCOL.md:9`, `REPORT_TR.md:13`), and the "whole problem"
notes of rounds 18, 19 and 20 return to them as the still-open target
(`docs/research/living_learning_interactions_2026_09_22/WHOLE_PROBLEM_AND_NEXT_TEST_TR.md:26`,
`docs/research/living_learning_distinctions_2026_09_23/WHOLE_PROBLEM_TR.md`,
`docs/research/living_learning_live_revision_2026_09_23/WHOLE_PROBLEM_TR.md` — the last two belong to other
groups; five files in total cite the concept folder).

**No paper edition.** None of `docs/publications/papers/01…10` or `docs/publications/working-notes/future-state-review.md`
references any of these four folders (grep for the folder names and for "Tur 12/14/18" returns nothing). These are
local, unpublished rounds; every report states that no commit, push or publication was made (relations
`REPORT_TR.md:114`, context `REPORT_TR.md:132`, interactions `REPORT_TR.md:80`).
`docs/research/RESEARCH_LEDGER.md` is a 2026-09-14 technique ledger (FSDP2, Muon, MLA…) and is unrelated.

**Integrity checks I ran (read-only, seconds; scripts in
`/tmp/claude-0/-workspace-cevahir-ai/79909244-c51d-5307-ad00-1a1e5ef77203/scratchpad/newcore/prior/_checks_g4/`).**
All 53 artifact hashes stored in the three `results/final_audit.json` files (19 relations + 19 context + 15
interactions: every `.py`, `.md` and result file of the three studies) equal the sha256 of the current files
(`g4_integrity.py` → `g4_integrity.json`: `mismatching: []` three times). `git status` was clean before and after;
no `__pycache__` was created (I never imported the repo's modules). The studies recorded base HEAD
`f4eb1dc550b3efc8b2a62fc1c5db262c804b219c`; today's HEAD is `28bcf0b…` — the files of these studies are
nevertheless byte-identical to what their audits hashed. Original environment: Python 3.14.3 + NumPy 2.4.2
(`final_audit.json › environment`); this machine has Python 3.13.18 + NumPy 2.5.4. **I did not rerun any
experiment**; every number below is read from the stored result JSONs (and re-aggregated by me from
`paired_summary.json` / `summary.json`, where it matched the reports to the last printed digit).

**Substrate fact.** No neural network, no gradient, no embedding anywhere in this group. NumPy is used as a table
engine. Everything "learned" is one of: a 6-entry transition table; a set of surviving candidate permutations
with fixed prior weights; a vector of 16,384 integer error counters; count tables; nine or thirty-seven measured
F17 effect vectors; one integer preference (1 or 2). The laws below are therefore functional statements about
*state*, not about a substrate, and none of them is evidence for or against a neural core.

**Evidence grades used in this digest.**
- **[EXACT]** finite algebraic/arithmetic witness, proof, or exhaustive finite check (in the study's own code or
  documents).
- **[EMP n=24]** measured over 24 paired lives. This group reports **descriptive** paired counts only: no
  confidence intervals, no significance tests, one configuration per study (`paired_summary.json ›
  descriptive_only: true`; relations `REPORT_TR.md:33`).
- **[SUGGESTED]** stated or interpreted by the study without a dedicated test.
- **[ANALYST]** my inference or my tiny check — *not* a result of the owner's studies (collected in §6).

Every report in this group opens by saying that the main problem is open and that no new principle is claimed
("Yaşarken öğrenmenin genel ilkesi veya genel yapay zekâ çözülmüş değildir", relations `REPORT_TR.md:9`; "Bu,
yeni temel kuram veya yeni bir öğrenme ilkesi bulunduğu iddiası değildir", context `REPORT_TR.md:9`; "Ana yaşayan
öğrenme problemi çözülmedi", interactions `REPORT_TR.md:5`). Turkish is kept where the words are decision labels
or the owner's verbatim observation.

**The owner's verbatim observations recorded in this group** (the "pencere" the triggering request refers to):

| # | Where | Topic |
|---|---|---|
| Q1 | `docs/research/living_learning_concepts_2026_09_22/OBSERVATION_AND_HYPOTHESES_TR.md:9` | how a concept ("aile") develops, splits, re-regulates |
| Q2 | same file `:13` | instruction: extend it in his own form, "bu pencereyi boşa atma" |
| Q3 | `…concepts…/CROSS_CONNECTIONS_AND_TRANSFER_TR.md:9` | cross-links between knowledge clusters → new thoughts, fast adaptation |
| Q4 | `docs/research/living_learning_relations_2026_09_22/NEXT_HYPOTHESES_TR.md:9` | flexible structure, stable references, continuity without rupture |
| Q5 | `docs/research/living_learning_context_2026_09_22/ACTIVE_EXPERIENCE_PROTOCOL.md:7` | a model that keeps learning during inference improves itself from the moment it starts |
| Q6 | `docs/research/living_learning_context_2026_09_22/REPORT_TR.md:116` | "sinir ağlarının çalışma prensiblerini yamalamak gibi bir niyetim yok … bambaşka bir buluş" |

---

## 1. `living_learning_concepts_2026_09_22` — the owner's observation of how meaning reorganises (two notes, no experiment)

Both notes are *records of an observation and its research opening*, written 22 Sept 2026 between round 11 and
round 12. They contain no code, no data, no result. Their status sentence: "Bu kayıt yeni bir deney sonucu veya
tamamlanmış mimari değildir" (`OBSERVATION_AND_HYPOTHESES_TR.md:5`); "Yeni kod veya sayısal deney yapılmadı"
(`CROSS_CONNECTIONS_AND_TRANSFER_TR.md:89`). They matter for the new core more than their size suggests: they are
the only place in the programme where the *target phenomenon* (concepts that split, merge, cross-link and are
re-evaluated by experience) is described in the owner's own terms, and they fix the obligations any later
"concept formation" claim must meet.

### 1A. `OBSERVATION_AND_HYPOTHESES_TR.md` — "Deneyimle yeniden örgütlenen anlam"

#### (1) Question
How does a topic develop in a mind over a life: which experience creates which distinction, changes an old
distinction, and how does that change take part in later understanding and learning? (`:21`: "hangi deneyimin
hangi ayrımı oluşturduğu, eski ayrımı değiştirdiği ve bu değişikliğin sonraki anlama/öğrenmeye nasıl katıldığı").

#### (2) Mechanism / state representation actually used
**None, on purpose.** "Yeni bir mimari seçilmedi; grafik, kümeleme, embedding, tensor büyümesi veya belirli bir
biyolojik mekanizma peşinen cevap sayılmadı" (`:113`). What the note fixes instead is the description of the
state that must be explained.

The owner's observation, verbatim (`:9`):

> bir insan bebekken farklı bir davranış gösterir çocukken başka ergenken başka yetişkinken başka erişkinden başka yaşlıyken başka zihinsel aktivitelerde bulunur. . zihinde bir konu nasıl gelişir. yaşanmışlıkla konular zihinde kümelenir belirli alt kategorilere ayrılır temsil uzaylarında mesela benim zihnimden örneklem de bulunalım aile kavramı dün 25 farklı olayı barındırırken 12 farklı duygu düşünceyi çağrıştırırken şuan 12 farklı dalı ve 22 farklı duyguyu barındırıyor bu veri uzayı daralıp genişleyebilir katmenl olarak sınırsız bir alt katman kapasitesi var bu uzayın. aynı şekilde zihnimdeki her olay ve olaylar zinciri ilk defa yaşanan olauy ise olayına göre kategorize edilp zamanla kategorisi regüle edilir ve anlam ağacı bu sayede genişlerken geçmiş deneyimlerimde aktif harıramda kalmasada aktif nöral ağda yaşamı devam ediyor.

and his instruction on how to treat it (`:13`): "… sen benim söylediğim biçimdende genişlet. bu pencereyi boşa
atma." The numbers (25 events / 12 feelings → 12 branches / 22 feelings) are kept as a personal illustration of
a concept's inner structure changing, not as a neural measurement (`:15`).

The note's reading of the observation (researcher's interpretation, kept apart from the quote):

- **Working proposition** (`:25`): "Bir deneyim yalnız temsil edilen yeni bir içerik olmayabilir; önceki ve
  sonraki deneyimlerin temsil edilme, ilişkilendirilme ve değerlendirilme biçimini değiştiren bir iç düzenleme
  nedeni olabilir." It adds at once that this is not yet a learning algorithm (`:27`).
- **Five forms of development** (`:37-41`): (1) *ayrıştırma* — events held in one group separate when a new
  distinction explains future outcomes better; (2) *birleştirme* — situations kept under different names turn
  out to share a relation; a common expectation or super-concept may form; (3) *çapraz ilişkilendirme* — the same
  event relates to family, trust, autonomy, responsibility at once; "Tek ebeveynli ağaç bir anlatım aracı
  olabilir; zorunlu hesap yapısı değildir"; (4) *yeniden değerlendirme* — the content of what happened stays, its
  meaning/importance/valence changes ("Ne oldu?" vs "onu şimdi nasıl yorumluyorum?"); (5) *ayrıntıdan kalıcı
  etkiye dönüşme* — the day and the sentences become inaccessible while an expectation, relation or learning
  tendency remains; this is **not** assumed to mean that all details are preserved.
- **Four kinds of change that must not be mixed** (table `:47-52`):

  | Change | How research must separate it |
  |---|---|
  | event knowledge (new event, or new evidence about an old one) | real content change vs change of interpretation |
  | conceptual structure (split, merge, re-relate) | look for a *computational difference in new situations*, not a renamed label |
  | evaluation (feeling, importance, expectation, value link) | the same category need not carry the same value in every context; not reduced in advance to one reward number |
  | access / transient state (another side of the same knowledge is salient today) | an effect that disappears when the transient state returns to normal vs persistent learning, separated by intervention |

  Plus a fifth axis: the developmental/internal state of the receiving system changes too; not all differences
  are explained by amount of data or one age number (`:54`).

#### (3) Experimental design
None. One finite thought experiment (§5 of the note, below) and four primary sources checked on 22 Sept 2026
(`:104-107`): Mack, Love & Preston 2016 (concept-relevant features and hippocampal similarity reorganise with the
categorisation demand); Schlichting, Mumford & Preston 2015 (integration and separation patterns coexist);
Audrain & McAndrews 2022 (detail fades while schematic structure persists over three days); Bethlehem et al.
2022 (lifespan brain charts). The note states that these neither prove nor refute the observation (`:102`).

#### (4) Results and decision labels
No result. Seven hypotheses, all **untested** at the time: "Bu hipotezlerden hiçbiri henüz yeni deneyle
doğrulanmış değildir" (`:90`).

| # | Hypothesis (`:76-88`) | Test the note demands |
|---|---|---|
| H1 | content and structure change together: new experience can change *which past situations are treated as same or different*, not only output parameters | paired streams that do / do not require the distinction; measure split/regroup, future performance and resource cost; true categories never given as update input |
| H2 | new knowledge changes an old interpretation while the old event's content stays fixed | separate queries for old content, current interpretation, unseen relational inference; unnecessary change of unrelated past counts as an error |
| H3 | recall vs lasting effect: with the raw event record inaccessible, learned state still affects future interpretation/acquisition | continue in a truly raw-record-free process; transport / corrupt / reset the remaining learned state. "Yalnız arşivden eski satır aramak bu hipotezin tanığı değildir" |
| H4 | concept vs evaluation: similarity of two events and the value given to them are not the same object | same event family with different outcome/evaluation in different contexts, acquired without needless category proliferation |
| H5 | reflexive effect: the old structure shapes the first reading of a new event, the new event can change the old structure | escape from a first wrong reading; use of accumulated non-confirming evidence; no wrecking of the whole structure by one noisy event |
| H6 | open-ended but resource-stated differentiation: number and depth of distinctions not given in advance; memory and compute finite; growth, merge, reuse and loss measured together | "Literal sonsuz kapasite varsayımı kabul edilmez. Sabit ağırlık boyutuyla yeni temsil örgüsü edinmek de mümkün bir adaydır; her yeni kavram için yeni tensör boyutu veya yeni nöron gerektiği ileri sürülmez." |
| H7 | meaning of event *chains*: the same multiset of events with a different order, prior expectation or later explanation can carry a different relational meaning | streams built from the same events but different order/conditions that need different future expectations, with controls where order truly does not matter; no causality from co-occurrence; the right temporal relation never given as a label |

#### (5) Law / invariant claimed
- **K1 — cross-cutting witness [EXACT, by inspection]** (`:60-68`). Four events with two accessible features,
  `e00=(0,0), e01=(0,1), e10=(1,0), e11=(1,1)`. If the first need depends on feature 1 the useful partition is
  `{e00,e01} | {e10,e11}`; if a later need depends on feature 2 it is `{e00,e10} | {e01,e11}`. The second
  partition is not a refinement of the first: it cuts both old groups. *Scope, in the note's words:* "Bu tanık,
  tek bir hiyerarşinin imkânsızlığını kanıtlamaz; bir kez verilmiş kategorilere yalnız ekleme yapan tek-okumalı
  düzenin yeterli kabul edilemeyeceğini gösterir" — a system may keep the four singleton leaves and build several
  groupings over them, or use context-dependent readings.
- **K2 — information limit [EXACT, re-application; "yeni teorem olarak sunulmaz"]** (`:70`). If the old state
  kept only feature 1 for each event and all discriminating information about feature 2 is inaccessible, the new
  grouping cannot be derived from the old state; a new observation or another preserved trace is needed. This is
  the state-transport limit of `docs/research/living_learning_reassessment_2026_09_20/STATE_TRANSPORT.md`
  (digested by another group) applied to concepts.
- **K3 — two different claims** (`:72`): "Ayrıntılı anı erişilmese de etkisi kalır" is not the same claim as "anı
  daha sonra gerekebilecek bütün ayrımlar için eksiksiz saklanır". The hard part is deciding *within experience*
  which distinctions to keep with which resource and which to merge.
- **K4 — the measure** (`:43`): fewer branches is not automatically loss (a more general relation may express
  them); more branches is not automatically better understanding (noise or over-reaction to a single event makes
  needless distinctions). "Ölçü, dal sayısı yerine açıklama, öngörü, bağlama uygun karar ve sonraki edinim
  üzerindeki etkiler olmalıdır."

#### (6) Limits
*Admitted:* not an experiment; no architecture; the numbers are an illustration; no biological claim; a matching
paper does not mean the observation is explained, and the absence of one is neither a refutation nor a proof of
originality (`:5`). The observation's "sınırsız bir alt katman kapasitesi" is explicitly *not* adopted literally (H6).
*Limits I see [ANALYST]:* (a) the witness assumes the two binary features are already *accessible*; forming the
feature is exactly the unsolved step ("concepts formed when needed"); (b) "evaluation/feeling" has no operational
definition yet, only the demand not to collapse it into category or reward; (c) the five forms are a taxonomy —
nothing here says which signal should trigger which form.

#### (7) Implications for a core where learning and running are one process
- **Requirement.** The concept store must admit *several simultaneous groupings over the same event leaves*
  (K1). A structure that only appends children under existing categories is ruled out as *the* computation
  structure; a tree may remain a way of displaying it (`:39`).
- **Requirement.** Four separately addressable kinds of change — fact content, grouping, evaluation,
  access/transient state (`:47-52`). A core whose "weights are graded beliefs" has to say which of the four a
  given weight carries, and needs an intervention test that separates a transient access effect (gone when the
  inner state returns) from a persistent change.
- **Requirement.** Retention is a decision about future distinctions (K2, K3): merging is irreversible with
  respect to the distinctions it drops, so "what to keep, what to merge, what new evidence to seek" is part of
  the learned state, not a storage detail.
- **Warning.** A hand-drawn `aile → güven → destek` tree into which events are placed "kullanıcının sorduğu
  oluşum ve düzenlenmeyi açıklamaz" (`:21`). Any experiment in which the researcher supplies the true categories,
  even as update input, does not test H1.
- **Warning.** Success is not counted in branches or concepts created (K4); it is measured on explanation,
  prediction, context-appropriate decision and *later acquisition*.
- **Open question.** H1–H7 as a whole. Within this group only H3 receives a partial computational instance
  (round 12 keeps the acquired transition structure and drops the raw source episodes:
  `research/living_learning_relations_2026_09_22/verify_relations.py:215` asserts `"source_observations" not in
  payload["state"]`; relations `REPORT_TR.md:19` adds that this is not a model of conscious recall vs implicit
  memory). Later rounds 19 (distinctions) and 22 (crossmap) say they return to these notes — see the other
  groups' digests.

### 1B. `CROSS_CONNECTIONS_AND_TRANSFER_TR.md` — "Ayrı anlam örgülerinin birleşmesi"

#### (1) Question
Can relations formed *between* separately developed bodies of knowledge produce a new, testable inference and
make a new topic faster to learn — and what must be shown before "a connection was formed" may be called that?

#### (2) Mechanism / state representation actually used
None chosen. The owner's observation, verbatim (`:9`):

> ilişkiler duygular yaşayış bilgiler insanın doğumdan ölüme dek edindiği tüm deneyimler zihinde bu şekilde işlenir. nöral yollar ile bu bilgi kümeleri çapraz bağlarla bağlanırsa eğer beyinde yeni düşüncelerde ortaya çıkaiblir birbirinden bağımsız gelişmiş düşünce bilgi yığınları tek bir nokta veya anlam fikir  üzerinden birleşebilir bu sayede insan çok kapsamlı bir konu hakkında bile derinlemesine bilgi sahibi olmasada hemen hemen diğer konularda edindiği bilgiler aracılığı ile yeni konuuya hızlıca adapte olup öğrenebilir.

Reading (`:15-19`): this is not "putting two archives in one place"; what is noticed is *which role* the elements
of the two bodies play, so that a relation in one yields an applicable expectation in the other. **Working
proposition** (`:19`): "Deneyim, yalnız tek tek bilgi üretmekle kalmayıp başka deneyimlerin birbirine
çevrilebilmesini sağlayan ilişkiler de oluşturabilir; bu ilişkiler yeni düşüncelerin ve daha hızlı sonraki
edinimin kaynağı olabilir." The "common point" need not be a word or object; it can be a dependency, an order, a
boundary, a reciprocity, a shared resource, or how one change affects another (`:23`). Illustration (the
researcher's, not the owner's life): kitchen orders piling up at one slow stage → the candidate relation
input–processing–waiting–capacity carried to a new processing system lets one ask the right questions earlier;
if the real fault is a synchronisation error the same analogy misleads and the limit of the mapping must be
found by new evidence (`:27-29`).

#### (3) Experimental design
None. Three primary sources (`:81-83`): Gentner 1983 (structure-mapping; assumes explicit relational
representations and mapping rules); Gick & Holyoak 1983 (schema induction; only abstract and bibliographic record
verified, full text not accessible, so no numbers used); Shohamy & Wagner 2008 (generalisation to new pairings
tied to integration during learning; a narrow task family).

#### (4) Results and decision labels
No result. "Bu maddeler deney yükümlülükleridir; henüz uygulanmış veya geçmiş sayılmaz" (`:67`).

#### (5) Law / invariant claimed
- **K5 — connection is not inference [EXACT, logical witness]** (`:35-40`). `A before B` and `B before C` support
  `A before C` inside one consistent time order; `A knows B` and `B knows C` do not entail `A knows C`. Both are
  drawn `A—B—C`. "Buradaki fark bağlantı sayısı değil, ilişkinin anlamı ve kullanılma koşuludur." *Scope:* shows
  that a computational account cannot be completed by adding links alone; says nothing about how biological
  connections hold such things.
- **K6 — shared feeling is not shared structure** (`:42`): a common emotion may make a memory accessible or a
  candidate important, but two situations evoking the same feeling is no proof that they carry the same causal
  structure. The roles of emotion in access, importance and evaluation are to be studied separately and not
  used as a truth label.
- **K7 — four measures that do not substitute for each other** (table `:46-51`): *recall* (access to acquired
  content); *recombination* (separately acquired parts used in a new structure); *new inference* (a new, testable
  expectation produced from relations **without an example that teaches the target result directly**); *fast new
  acquisition* (durable/generalisable success with fewer examples or less compute **under the same target
  evidence and resources**). A system can explain a new topic fluently with familiar words and still fail to
  produce a correct new prediction; one correct guess does not show a relation that can be reused; fetching the
  target result secretly from an archive is not inference or transfer (`:53`).
- **K8 — the strong form** (`:55`): the new connection should not answer once but become "daha sonra başka
  bağlantıların da kurulabileceği kalıcı bir ilişki" — new thought as material for later learning. How long that
  recursion can be sustained under finite resources is stated as open.
- **Seven obligations for any bridge experiment** (`:59-65`): (1) *separate acquisition* — sources learned from
  different experiences; the later common label or full solution not stored beforehand; (2) *role and relation
  mapping* — targets that look similar but carry a different relation vs targets that look different but carry
  the right one; word similarity alone is not a control; (3) *generate, then verify* — record expectations about
  untaught target results first, then measure which are corrected and which become permanent; (4) *intervention*
  — disable the acquired relation while keeping the source skills, with an equal-size unrelated-link control,
  and check that the intervention damaged nothing else; (5) *compute and evidence matching* — source learning
  and link-finding costs included; "fewer target examples" and "same examples with less compute" reported
  separately; (6) *correctable transfer* — when the first link is wrong, find the right scope from later
  evidence without deleting the relation that is valid in the source; (7) *persistent reuse* — the acquired link
  carried into a new process and used in a **third** task; "Bu test geçmeden yalnız geçici çağrışım için yaşam
  boyu yeni kapasite iddiası kurulamaz".

#### (6) Limits
*Admitted:* "tüm deneyimler" and "nöral yollar" are parts of the owner's general hypothesis; no narrow cognitive
experiment confirms them for a whole life and brain (`:11`). Supplying the right common relation by hand
"keşif ile hazır şablon kullanımını karıştırır" (`:67`). The earlier negative-transfer result is kept: correct
source knowledge carried to an unsuitable target can make acquisition harder (`:73`). The state-transport limit
applies: "eksik ayrım sırf daha çok bağlantı çizilerek geri üretilemez" (`:75`).
*Limits I see [ANALYST]:* the note gives no criterion for when two bodies of knowledge *should* be bridged
(the trigger), and no cost model for maintaining bridges; both are deferred to experiments.

#### (7) Implications for the core
- **Requirement.** A link in the core must carry its *relation type* — its composition rule and its condition of
  use (K5). Graded, untyped association strengths cannot decide whether `A—B—C` licenses `A—C`. (Round 12 below
  makes the same point quantitatively: a matrix of per-link belief strengths is insufficient, §2 (5) R2/R3.)
- **Requirement.** Report recall, recombination, new inference and acquisition speed-up as four separate
  measurements (K7); imagination ("new thought") counts only as *predictions recorded before the outcome*
  (obligation 3).
- **Requirement / test template.** The seven obligations are a ready acceptance protocol for "the way of
  constructing transfers": in particular the ablation with an equal-size unrelated control (4) and the
  third-task reuse (7). No study in this group passes (4) or (7). By my reading [ANALYST] round 12 meets (1)
  and (3), and parts of (2) (a target with the same objects and bijectivity but another relation), (5) (source
  and renaming costs are recorded) and (6) (the source tables survive a failed transfer, but the scope found is
  crude: the whole target action is dropped) — all inside a supplied language (§2).
- **Warning.** Salience (emotion, surface similarity) may steer *access*; it must never be read as evidence of
  validity (K6).
- **Open question.** The strong form K8 — a formed bridge that later bridges build on — is untested anywhere in
  this group.

---

## 2. Study `living_learning_relations_2026_09_22` (round 12) — separately acquired relations, new composition, wrong transfer

### (1) Question
Can relations learned from experience in two separate environments that share no object identities be matched
to a newly named target (mapping not given), predict two-step results that were never taught, and reduce the
target experience needed — and what does the same transfer cost when the target's relation family has changed
or has one local exception? (`PROTOCOL.md:9-16`, hypotheses H8–H11.)

### (2) Mechanism / state representation actually used
Symbolic, deterministic, enumerative; beliefs are prior masses times a {0,1} survival mask. File:
`research/living_learning_relations_2026_09_22/relation_learning.py`.

- **World language.** Six states, each action a function `f: S → S` (`:22`).
- **Source acquisition.** `learn_source` (`:47-57`) fills a 6-entry table from observed `(x, y)` pairs; raises on
  contradictory observations and on an incomplete table ("incomplete source; no teacher completion allowed",
  `:55-56`).
- **The transportable object.** `observed_class(table)` (`:60-61`) = the de-duplicated set
  `{π∘f∘π⁻¹ : π ∈ Sym(6)}`, built by trying all 720 renamings with `conjugate` (`:40-44`,
  `result[rename[old]] = rename[function[old]]`). In the audit's notation
  `C(f) = {π∘f∘π⁻¹ : π ∈ Sym(S)}` (`MATHEMATICAL_AUDIT_TR.md:13-15`). Class sizes follow
  `N = 6!/∏_j j^{m_j} m_j!` (`:23-29`): 120 (one 6-cycle), 15 (three 2-cycles), 40 (two 3-cycles), 144 (fixed
  point + 5-cycle). What experience contributes is therefore the *cycle type*; that renaming by bijections is the
  right notion of "same relation elsewhere" is supplied ("Bir kaynağın hedefe nesne bijeksiyonuyla taşınabileceği
  hipotez dili araştırmacının önkabulüdür", `PROTOCOL.md:22`).
- **Persistent learner state** (`Learner.__init__` `:89-115`, `state()` `:173-181`): the learned source tables
  (raw source episodes are *not* kept); `candidates` (one row per candidate function over all groups);
  `group_ids`; fixed `prior_weights[f] = 1/(G·|group(f)|)` (`:105`: classes equal, members equal within a class);
  a boolean survival mask `active[2, n]` per target action (`:106`); `fallback[2]`, `fallback_at[2]`
  (`:107-108`); the fact table `observed[2, 6]` (`:109`); cost counters.
- **Running = reading the belief** (`probabilities`, `:117-133`): for an action not in fallback,
  `P(x→y | D) = Σ_{f active} w_f·[f(x)=y]` with `w` = renormalised prior weights of the survivors (`:127-132`); for
  an action in fallback, a seen cell answers with certainty and an unseen cell is uniform `1/6` (`:120-126`). The
  prediction is the posterior mean; no single candidate is picked ("ilk aday keyfî seçilmez", `PROTOCOL.md:47`).
- **Learning = one filter step on the same state** (`update`, `:135-156`): record the fact (`:141`);
  `new = old & (candidates[:, x] == y)` (`:146`). If survivors remain, keep them. If none remain: the control
  `frozen_source_both` counts a rejection and keeps the last non-empty pool (`:150-151`); every other method
  switches **that action only** permanently to the fact table (`:152-156`). The class posterior is
  `Pr(k | D) ∝ Pr(k)·n_k(D)/N_k` (`MATHEMATICAL_AUDIT_TR.md:33-39`; code `posterior_summary`, `:158-171`).
- **Two-step inference** (`evaluate`, `:265`): `p[0] @ p[1]` and `p[1] @ p[0]`, the product of the two
  posterior-mean transition matrices. The composed labels are never given to learning.
- **Life loop** (`run_trajectory`, `:273-287`): for each target fact, the prediction is recorded *before* the
  outcome (`:278-280`), then `update` (`:281`), then a full evaluation (`:282`); the final state is hashed (`:287`).
- **Scores** (`score`, `:247-259`): Brier = mean over queries of `Σ_y (p_y − 1[y = truth])²` (0 exact, 2 certain
  and wrong); `tie_accuracy` = expected accuracy of a uniform random choice among tied maxima (no identity-based
  tie-break); `confident_wrong` = `max p ≥ 1−1e−12` and `p[truth] < 1e−12`.

*Given vs learned.* Given: the six object identities, action identities, determinism, the renaming hypothesis
language, the common target state space that lets one action's output be the other's input, and the composition
interpreter (`PROTOCOL.md:22-26`; `MATHEMATICAL_AUDIT_TR.md:19`). Learned: the source tables (hence the class),
which candidates survive in the target, the class posterior of each target action. The learner is *not* told
that one target action is A-like and the other B-like, nor any source–target object mapping.

### (3) Experimental design
- **Sources** (`make_sources`, `:210-218`; `CANONICAL`, `:28-29`): per life four independently renamed source
  environments — A = 6-cycle, B = three 2-cycles, C = two 3-cycles, D = fixed point + 5-cycle; each stream = the
  six transitions twice, shuffled (12 observations; the second copies add no constraint in this noise-free model
  but are counted as experience).
- **Target worlds** (`make_world`, `:221-244`): two actions on six common states with independent renamings and
  shuffled action order; the 12 `(action, state)` cells are revealed once each in random order. Three regimes:
  `compatible` (renamed A and B); `cycle_shift` (B replaced by two 3-cycles — same objects, same bijectivity,
  different relational structure; A still valid); `local_exception` (one transition of B redirected, so that
  action is no longer bijective; A still valid).
- **Units:** 24 lives × 3 regimes × 8 methods = **576 trajectories**; 6,912 pre-outcome predictions; 7,488
  evaluation snapshots (not independent samples). Methods and regimes of one life are paired.
- **Eight methods** (`PROTOCOL.md:38-45`): `table` (no source); `permutation_prior` (all 720 bijections uniform —
  bijectivity given, no source knowledge); `source_a`; `source_b`; `source_both` (A and B classes, 1/2 each);
  `unrelated_sources` (classes of C and D: also 24 *correct* source observations, different structure);
  `frozen_source_both` (refuses updates that would empty the pool; still writes the fact into its audit table but
  does not use it to predict); `cycle_catalog` (all 11 cycle types of Sym(6) equal, members equal within a type).
- **Queries:** the 12 direct cells; the not-yet-seen cells; 12 two-step queries (six starts × two orders of the
  two *different* actions). Same-action repetition is deliberately excluded (law R2 below).
- **Statistics:** descriptive paired counts over 24 lives; "daha geniş dünya ailelerine ilişkin anlamlılık veya
  güven aralığı iddiası yoktur" (`REPORT_TR.md:33`). Protocol and the mathematical audit were written before the
  performance run; the structural witness and `NEXT_HYPOTHESES_TR.md` after it.
- **Verification** (`verify_relations.py`): class sizes against an independent cycle classification and 24
  missing-edge rejections (`:43-68`); renaming of sources, target objects and target actions leaves transformed
  predictions equal within 1e−12 at 312 checkpoints (`:71-96`); the composition probability by direct enumeration
  of 12 candidate pairs, the same-action counterexample, the predicted equality of the initial marginals of
  `source_a/b/both` (`:99-155`); frozen/fallback semantics, score boundaries, and — by `inspect.signature` — that
  the learner's constructor and `update` take no truth, regime or seed (`:158-179`); 24 fresh-process
  continuations whose later predictions and final state equal the *uninterrupted live object* exactly
  (`:191-218`); a full recheck of all 576 recorded runs (`:221-260`); an exact full rerun.

### (4) Results (the study's numbers) and decision labels

**Compatible target** (`REPORT_TR.md:37-46`; all re-read from `results/paired_summary.json`):

| Method | mean pre-outcome Brier over the 12 target facts | composition Brier after 6 facts | after 8 facts | expected accuracy after 8 |
|---|---:|---:|---:|---:|
| table | .833333 | .666667 | .497106 | 51.25 % |
| permutation_prior | .591667 | .562838 | .342014 | 66.32 % |
| source_a only | .675000 | .631526 | .428530 | 62.67 % |
| source_b only | .611343 | .532022 | .273148 | 74.77 % |
| **source_both** | **.398479** | **.343091** | **.092231** | **92.71 %** |
| unrelated_sources (24 correct observations, other structures) | .920371 | .750482 | .582262 | 50.68 % |
| frozen_source_both | .398479 | .343091 | .092231 | 92.71 % |
| cycle_catalog | .531733 | .483121 | .240733 | 81.42 % |

- `source_both` has the lower pre-outcome Brier than table, both single-source controls, the bijection prior and
  the all-cycle-types catalog in **24/24 lives** [EMP n=24]. Not at every instant: at fact 8 its composition
  error is lower than the bijection control in 23 lives and higher in one (`REPORT_TR.md:48`).
- Number of target facts until all 12 two-step queries are exact (descriptive extra): **8.58** (7–10) for
  `source_both`; 10.54 (10–11) for the bijection prior and the cycle catalog; 12 for the table
  (`REPORT_TR.md:50`); `source_a` 11.38, `source_b` 10.83, `unrelated_sources` 12 (my aggregation of
  `paired_summary.json › first_exact`).
- The price: 24 source observations and 1,440 renaming attempts first. "Bu sonuç **hedef veri verimliliği**,
  toplam yaşam deneyiminden tasarruf garantisi değil" (`REPORT_TR.md:50`).

**Changed targets** (`REPORT_TR.md:54-64`), mean pre-outcome Brier:

| Target | source_both | permutation_prior | source_a only | frozen_source_both |
|---|---:|---:|---:|---:|
| compatible | .398479 | .591667 | .675000 | .398479 |
| relation family changed (`cycle_shift`) | .745994 | .591667 | .683681 | .832221 |
| one local exception | .692425 | .746701 | .676736 | .623336 |

- **Negative transfer [EMP n=24].** With the family changed, `source_both` is worse than the bijection prior and
  than `source_a` alone in **24/24 lives** — a second, *correct* piece of past knowledge slows learning when
  carried to an incompatible target. (Also 24/24 worse than `cycle_catalog` and than `unrelated_sources`; in this
  regime one of the "unrelated" sources is the one that now fits, `REPORT_TR.md:29`.)
- **False certainty [EMP].** Certain-and-wrong pre-outcome predictions of the correctable two-source learner:
  **0/288** compatible, **15/288** family changed, **13/288** local exception (`REPORT_TR.md:60`).
- **Correction scope.** When all candidates of an action are eliminated, only that action falls back to the fact
  table; the other action keeps its learned structure. With complete target evidence every correctable method
  ends exact. The frozen control ends with composition Brier **.666667** (family changed) and **.430556** (local
  exception), worse than the correctable learner at the end in 24/24 lives in both regimes (48 and 31 rejected
  updates in total).
- **Counter-result the report insists on** (`REPORT_TR.md:64`): under one local exception, *refusing* the
  contradiction gives the better mean pre-outcome error — **.623336** vs **.692425**; the correctable learner is
  better in 7 lives, equal in 2, worse in 15. The offered reason — one exception makes the learner drop the whole
  structural transfer for that action although most of it is still valid — is [SUGGESTED]: "ayrı nedensel
  ayrıştırma deneyi yapılmadı". "Mevcut geri dönüşün en iyi düzeltme yöntemi olduğu gösterilmedi."

**Hypothesis table** (`REPORT_TR.md:83-89`):

| Hypothesis | Status in the report | Scope the report attaches |
|---|---|---|
| H8 separate sources together ease target acquisition | **destek** in the compatible regime | not generalised to all regimes; source cost differs |
| H9 correct old knowledge yields false certainty in an incompatible target | **destek** | under the given finite candidate language and closed prior |
| H10 knowing the source structure ≠ knowing the role mapping | no verdict word in the report: "Gizli eşleme girişi yok; ad değiştirme denetimi geçti" → **inconclusive** (my label) | no separate oracle-mapping comparison; sizes of the two advantages not separated |
| H11 separate relations support untrained compositions | **destek** | composition rule and common target interface are given |
| H12 equal current output ≠ equal future learning | **post-result exact witness** | for a state reduced to mean predictions; no impossibility claim for all representations |

**Cost** (`REPORT_TR.md:93-101`): source observations / renaming attempts / distinct candidate tables / mean
logical filter checks (compatible) / mean final JSON bytes — table 0 / 0 / 0 / 0 / 311; bijection prior
0 / 0 / 720 / 1,746 / 35,613; A only 12 / 720 / 120 / 304 / 6,215.08; B only 12 / 720 / 15 / 40.92 / 1,052; both
24 / 1,440 / 135 / 344.92 / 6,959; unrelated 24 / 1,440 / 184 / 446.42 / 8,815.25; cycle catalog
0 / 0 / 720 / 1,746 / 35,998. The counters are logical checks, not FLOPs or CPU time.

### (5) Laws / invariants claimed (none presented as new; `LITERATURE_TR.md:13,33,39`)

- **R1 — when mean beliefs compose exactly [EXACT, conditional].** For two *different* actions with independent
  posteriors, `q(P₀,P₁ | D₀,D₁) = q₀(P₀ | D₀)·q₁(P₁ | D₁)` gives `E_q[P₀P₁] = E_{q₀}[P₀]·E_{q₁}[P₁]`
  (`MATHEMATICAL_AUDIT_TR.md:45-62`): the product of posterior-mean transition matrices *is* the exact two-step
  prediction "tanımlanan bağımsız posterior altında". Confirmed by direct enumeration (`verify_relations.py:99-120`).
  The generator of the compatible world actually has a dependency ("one action is A, the other B") that the
  learner is not given, so the product is exact for the learner's model, not for the generator's full knowledge
  (`:64`).
- **R2 — when they do not [EXACT, counterexample].** For the *same* unknown action used twice,
  `E[P²] ≠ (E[P])²` in general. The protocol's class B is the counterexample: each of the 15 functions is an
  involution, so `E[P²] = I`, while the one-step mean `K` (0 on the diagonal, 1/5 elsewhere) gives `K²` with 1/5
  on the diagonal and 4/25 off it (`MATHEMATICAL_AUDIT_TR.md:68-81`). "Bu, sayısal hassasiyet sorunu değil, aynı
  gizli fonksiyona ait bağımlılığın silinmesidir." The experiment therefore contains no `0→0`, `1→1`, `0→1→0`
  queries and "sonuç uzun zincirlere genişletilmez" (`PROTOCOL.md:57`).
- **R3 — same current answers, different future learning [EXACT, post-result witness]**
  (`structural_witness.py:17-41`, `results/structural_witness.json`, `REPORT_TR.md:68-79`). Before any target
  evidence the A class (120 six-cycles) and the B class (15 triple-two-cycles) give *identical* one-step
  predictions: 0 for staying, 1/5 for each of the five others. Give both the same new fact `0→1`. Then
  `P(1→0) = 0` under A (a six-cycle contains no closed two-cycle; 24 candidates remain) and `P(1→0) = 1` under B
  (3 candidates remain). Hence: "A deterministic state/update using only the shared current marginal matrix and
  this same evidence cannot reproduce both correct class-conditioned updates"; in the report's words "**Yalnız
  mevcut ortalama tahminleri korumak, bu modelin gelecekteki öğrenme davranışını korumaya yetmez.** İlişkiler
  arasındaki ortak kısıtların veya eşdeğer ayırt edici bilginin de korunması gerekir." *Scope:* "Bu, embeddinglerin
  yetersizliğine ilişkin genel bir teorem değildir; uygun bir temsil bu yapıyı saklayabilir."
- **R4 — a prior is not neutral [EXACT, arithmetic].** Equal class priors give each six-cycle weight 1/240 and
  each triple-two-cycle 1/30; giving the 135 pooled functions equal weight would instead mean class priors 8/9
  and 1/9 (`MATHEMATICAL_AUDIT_TR.md:31`). `permutation_prior` and `cycle_catalog` cover the same 720 functions
  and differ only in prior; the difference between them "ifade kapasitesi artışı değildir" (`:43`). A change of
  class posterior is re-weighting among given alternatives, not a new class being learned (`:41`).
- **R5 — contradiction leaves the model [EXACT, definitional].** When every candidate is eliminated the
  normalising constant is zero and Bayesian conditioning is undefined inside that model; the fall-back to the
  table is a *supplied model-switch rule*, not conditioning under one fixed prior (`MATHEMATICAL_AUDIT_TR.md:87`).
- **R6 — transfer needs structural fit [EMP n=24].** Past structure helps exactly when it fits the target
  (24/24 better than every control in the compatible regime) and hurts when it does not (24/24 worse than a
  structure-free bijection prior after the family changed), with false certainty appearing only in the unfit
  regimes (0 → 15 and 13 per 288). Equal amounts of correct source experience with the wrong structure do not
  help (`unrelated_sources` .920371 vs table .833333 in the compatible regime).
- **R7 — immediate prediction and eventual fit are different measures [EMP n=24].** "Sonradan bilinen gerçeğe
  uyum ile hemen sonraki tahmin başarımı aynı ölçü değildir" (`REPORT_TR.md:64`): the correctable learner ends
  exact and the frozen one never does, yet under a single exception the frozen one predicts the *next* facts
  better (.623 vs .692).
- **R8 — invariance checked, not assumed [EXACT check].** Predictions transform correctly under independent
  renaming of source objects, source record order, target objects and target actions (312 checkpoints, 1e−12).

### (6) Limits
*Admitted by the study.* Objects are given as recognised; the relation language, the renaming family, the common
interface and the composition procedure are supplied — calling the result a self-acquired web of meaning "kapsam
aşımı olur" (`LITERATURE_TR.md:41`). Positioning: relational transfer of acquired finite models + Bayesian
candidate weighting + a supplied composition procedure, each with clear precedents (Gentner 1983; Tenenbaum,
Kemp, Griffiths & Goodman 2011; Whittington et al. 2020, TEM), "Birlikte kullanılmış olmaları … yeni ve genel
bir matematiksel ilkesinin bulunduğunu göstermez" (`:39`). Two-source vs one-source comparisons differ in the
amount of source information; the equal-24 unrelated control shows only that the count alone is not enough.
Total-life saving is not shown. No oracle-mapping comparison (H10). The fall-back was not shown to be the best
correction. Results are descriptive; 576 trajectories are not 576 independent worlds. Local checks only, no
external replication.

*Limits I see [ANALYST].*
1. The belief is not graded: likelihoods are {0,1}. Whenever the truth lies outside the candidate language the
   observed fact had model probability exactly 0 — one such event per affected action in every life (24/24 lives
   for `source_both` in both changed regimes; §6 check A). The reported "kesin yanlış" count (15, 13) only counts
   the subset in which some *other* outcome had probability 1; the events in which the truth was excluded with
   certainty are 24 and 24. Under a log score every one of them is an infinite loss.
2. The fall-back is irreversible and all-or-nothing per action: after one exception the learner never returns to
   the structure although 5 of 6 transitions still obey it (`:152-156`; there is no path back from
   `fallback[action] = True`). This is the cause the report *suggests* for the counter-result R7; it was not
   isolated by an experiment.
3. The stored object is far larger than the invariant: the class is carried as an explicit list (120/15 rows;
   6,959 bytes vs 311 for the table) although the cycle type plus the observed facts determine it.
4. Enumeration over 6! renamings does not scale; nothing is said about approximate matching.
5. Structure was *identified* quickly — each action's class is certain after 2.96 of its own facts on average
   (range 2–4; §6 check C) — because the two given classes are easy to tell apart. A richer class catalogue costs
   more facts (`cycle_catalog`: 10.54 vs 8.58 to exactness).
6. Wrong structure is worse than no structure on the cells not yet seen: after 8 facts the unseen-cell Brier is
   .986697 for `unrelated_sources` against .833333 for the table (compatible regime), and 1.140046 for
   `source_b` after 6 facts when the family has changed (`paired_summary.json`, my aggregation).

### (7) Implications for a core where learning and running are one process
- **Mechanism to reuse.** The predict-before-outcome life loop on one state object (`run_trajectory`), with a
  state hash, restart from serialised state compared against the *uninterrupted* object, renaming-equivariance
  checks, a privileged-free interface proved by signature inspection, and the frozen / unrelated-equal-count /
  strong-prior controls. This is the group's cleanest template for testing "learning and running are one
  process".
- **Mechanism to reuse.** Transfer as a *prior over target structures*: a class abstracted from one experience
  becomes a set of candidate structures for a newly named situation, weighted class-then-member, filtered by
  each new fact, with a per-action scoped model switch when nothing fits.
- **Requirement (from R3).** Equality of two states must be tested on their *response to the same next
  evidence*, never on current outputs alone. Any consolidation, compression or "sleep" step of the core has to
  pass that test.
- **Requirement (from R2, R3).** "Weights as graded beliefs" cannot mean per-link marginal strengths: a matrix of
  such strengths (a) composes wrongly with itself and (b) does not determine how to learn from the next fact.
  Beliefs have to be held over structures that keep the joint constraints, or over an equivalent summary (here:
  class identity + facts).
- **Requirement (from R4).** Whatever counts hypotheses (per function, per class, per representation) is a
  modelling decision with measurable consequences; it must be declared and varied as a control.
- **Warning (from R6).** Acquired structure produces certainty that is wrong outside its scope, and correct
  knowledge can be worse than none. A measure of "how real a construction is" must be taken against a
  structure-free but informed prior on not-yet-seen cases, not against a memorising table only ("Sadece zayıf
  bir ezber tablosunu yenme sonucuna dayanılmıyor", `REPORT_TR.md:48`).
- **Warning (from R5, R7).** Hard elimination forces an out-of-model switch and that switch discards valid
  structure. Neither "freeze" nor "abandon the whole relation" is right.
- **Open question (H15, below).** How to tell, from evidence alone, "the source is wrong" from "the role mapping
  is wrong" from "the relation family changed" from "a local exception".
- **Open question (H16).** The interface through which two relations compose was given here; round 22
  (crossmap, another group) reports that its attempt to remove this limit "tam karşılanmadı"
  (`docs/research/living_learning_crossmap_2026_09_23/REPORT_TR.md:11`).

### 2B. `NEXT_HYPOTHESES_TR.md` — after round 12: continuity while changing

Written after the results; "Yeni hipotezler sonuç öncesi kayıt gibi sunulmaz" (`:5`). The owner's new
observation, verbatim (`:9`):

> bilinç esnek bir yapıdır. mesela doğru kabul görmüş bir konu bilinçte doğru referanstır örnek verecek olursak matematik modellemeleri gibi somut şeyler her zaman evrenseldir. olayların değerlendirilme şekli bilginin bağlam uzayındaki yapısı her an değişkenlik görsetebilir ancak ipucu yani börel yollar takip edildiğinde deneyim ile geçmiş öğretilerin bir araya gelmesi zihindeki yapının kopukluk yaşamadan varlığını sürdürmesine olanak sağlar.

Researchable core kept (`:11`): "Bir bilgi yapısı, yeni deneyimle bağlantılarını ve değerlendirmelerini
değiştirirken geçmiş deneyimlerin kimliğini, ilişkilerini ve geleceğe etkisini tümüyle kaybetmeden
sürdürülebilir." No criterion for subjective experience is set up. The mathematics example is split into two
levels (`:13`): validity of a theorem under stated axioms and conditions (stable), and the fit of that structure
to a particular physical or lived event (revisable by evidence); group acceptance alone does not make a
proposition universal.

| # | Hypothesis | Status | Test demanded |
|---|---|---|---|
| H12 (`:15-19`) | behind equal current answers there can be different learning structures | **proved by the exact witness** of round 12; "yeni bir evrensel yasa iddiası değildir" | evaluate "how will it change when the same new fact arrives", not only "what does it answer now" |
| H13 (`:21-27`) | a persistent state that separates the *event record*, the *interpretation* drawn from it and the *assumptions* that made the interpretation possible supports selective re-learning when context changes | not tested | changed context, corrected event, changed value as separate interventions in one stream; no supervisor telling which changed; "Bir 'gerekçe listesi' yazabilmek yeterli başarı ölçütü değil" — effect on future decisions and new learning required |
| H14 (`:29-35`) | growth of the concept order is not limited to tree growth: overlapping, compatible memberships of the same event can need less re-learning than one fixed tree — if the gain exceeds the cost of maintaining them | not tested ("tur 12'de sınanmadı") | fixed tree vs overlapping membership vs re-coded continuous representation vs fact memory under equal information and budget; split/merge counts are not success |
| H15 (`:37-43`) | when an old relation fails in the target, distinguishing "source wrong" / "role mapping wrong" / "relation family changed" / "local exception" by evidence reduces both stubbornness and wholesale abandonment | need made visible by round 12; no general method found | worlds whose observation prefix is compatible with all four explanations and diverge later; measure false certainty, explanations invented before discriminating experience, and damage to the old source competence |
| H16 (`:45-51`) | the joining interface itself may be learnable when the separately acquired structures share neither object identities nor a ready common state space; to be more than name-matching, object count, abstraction level and relation arity must be able to differ | open; "güçlü sürümü hâlâ açık" | no bridge type or mapping given as label; counter-examples: similar surface but opposite relation direction, shared word but incompatible inference rule, many-to-one mappings; benefit tested on untrained queries and on acquiring a third environment. "Birkaç hazır seçenekten köprü seçmek, seçenekleri de deneyimden üretmekle aynı değildir" |
| H17 (`:53-57`) | internal state as a source of new experience and of concept change | not tested | same external input and same recorded past with different current inner state: only a different answer, or a new competence that persists and transfers? random perturbation vs specific changed state vs fixed state under equal budget. "Farklı davranış tek başına öğrenme sayılmamalı" |

The note's own generality table (`:61-67`) is worth keeping as a guard: a given summary being insufficient does
not invalidate all compact representations; round 12 shows that fit of learned relations to a new target governs
data efficiency and new compositions, and that "Hazır temsil dili, ortak arayüz ve sonlu yeniden adlandırma
uzayı açık uçlu anlam gelişimi değildir". Closing question (`:69`): "deneyim, gelecekte kullanılabilir hesaplama
kapasitesini nasıl kalıcı, kontrollü ve genellenebilir şekilde değiştirir?"

**Implication for the core.** H13 is a direct state-design requirement: three separately revisable layers —
what was recorded, what was concluded, which assumptions license the conclusion — so that a later fact
("gelemediği için hastanedeydi", `:25`) changes expectations without deleting the event, and so that a corrected
*record* (measurement error) is a different operation from a corrected *interpretation*. H15 is the same
requirement seen from the transfer side. Neither has evidence yet in this group.

---

## 3. Study `living_learning_context_2026_09_22` (round 14) — inferring context, confounded cues, self-selected experience

### (1) Question
Without being given the context identity, can a system learn from experience *which observable history relation*
selects between two task rules — and can that be told apart from following a visible surface cue that coincides
with it during acquisition? (H42–H45, `PROTOCOL.md:9-17`.) Extension written before any result: if the learner
chooses its own next experience from its current uncertainty, what is gained per external observation and what
does the choosing cost? (`ACTIVE_EXPERIENCE_PROTOCOL.md`.)

### (2) Mechanism / state representation actually used
Exhaustive soft Bayesian mixture over a supplied finite structure language; the persistent state is a vector of
integer error counters. File: `research/living_learning_context_2026_09_22/context_learning.py`.

- **Input** `(h, s, x)`: four "history" bits, one surface bit, four task bits; 512 inputs (`:27`). `h` is
  *supplied as a ready summary*: "h, gerçek akışın önceki dört olayından çıkarılmıyor; örnekler arasında kayan
  zaman penceresi yok" (`REPORT_TR.md:13`).
- **Structure language (supplied).** Sixteen gates: the 15 non-zero parities `g_a(h) = aᵀh mod 2` plus the
  surface gate `s` (`:31-32`). Thirty-two experts: affine parities `f_{u,b}(x) = uᵀx ⊕ b` (`:28-30`). A structure
  `θ = (g, f₀, f₁)` computes `F_θ(h,s,x) = f_{g(h,s)}(x)`. All 16·32² = 16,384 structures are tabulated on all 512
  inputs in one bank (`:34-37`; 67,108,864 bytes of float64). They are 15,904 distinct functions:
  `16(32² − 32) + 32` (`MATHEMATICAL_AUDIT_TR.md:25-31`), because the 32 structures with `f₀ = f₁` are the same
  function under every gate.
- **Persistent state** (`Learner._initialize`, `:74-86`): `errors` — **one integer per structure: how many observed
  labels it contradicted** (`:76`); a fixed `prior` (`:81-84`: uniform over the allowed structures, or for
  `joint_balanced_family` `.5/(15·1024)` per history-gated structure and `.5/1024` per surface-gated one); two
  counters. No raw event rows (`raw_event_rows=0`, `:146`). Table baselines keep `counts[key, label]` (`:79`).
- **Running = reading the belief** (`:88-105`): `log q(θ | D) = log prior(θ) + errors(θ)·log(ε/(1−ε))` up to a
  constant, ε = .05 (`:26`, `:89-91`); prediction `p(z) = Σ_θ q(θ | D)·F_θ(z)` (`weights() @ bank`, `:100-101`) —
  the posterior mean of the *clean* candidate outputs; no label noise is added to the prediction.
- **Learning = one increment on the same state** (`observe`, `:107-116`):
  `errors += (FUNCTIONS[candidate_ids, index] != label)` (`:112`). Nothing else changes.
- **Why soft.** Labels are clean; ε = .05 is "öğrenicinin önceden seçilmiş hata varsayımı" (`REPORT_TR.md:19`),
  fixed before the first run, chosen so that one contradiction does not delete a candidate irreversibly
  (`MATHEMATICAL_AUDIT_TR.md:90`) and so that no "Boş aday kümesi ve keyfî geri dönüş yordamı" is needed
  (`PROTOCOL.md:55`) — the deliberate contrast to round 12's hard elimination and fall-back.
- **Privilege isolated.** The true-gate control is a separate subclass (`OracleLearner`, `:149-155`); the
  ordinary constructor refuses the oracle method (`:65-66`) and takes only the method name
  (`verify_context.py:85-86`).
- **Life loop** (`run_one`, `:203-219`): predict before the label (`:209-210`), then `observe` (`:211`); snapshots
  after 0, 12, 24, 48, 49, 52, 56, 64 observations.
- **Self-selected experience** (`active_experience.py`): a `Controller` holds the learner, a 512-bit `seen` mask
  and an RNG (`:21-29`). `choose` (`:37-55`): `uniform_query` takes a random unseen input; `disagreement_query`
  predicts all 512 inputs and takes the unseen input maximising `p(1−p)` (`scores = predictions*(1-predictions)`,
  `:47`; ties within 1e−12 broken by the RNG, `:48-50`). The environment discloses the label only after the choice
  (`:90-94`): "Sistem kendi cevabını doğru etiket saymaz" (`REPORT_TR.md:71`). The scoring cost is counted in
  candidate × input cells (`:42`, `:46`).

*Given vs learned.* Given: the four-bit history code, the two-expert decomposition, the parity gate and expert
families, the full candidate bank, and — in the extension — an interface that returns the correct label for any
of the 512 inputs on request. Learned: the weights inside that family. "Geçmişten ilgili değişkenleri çıkarmanın,
sınırsız kavram yaratmanın veya bağlam kavramının kendisinin keşfedildiği söylenemez"
(`MATHEMATICAL_AUDIT_TR.md:21`).

### (3) Experimental design
- **World** (`make_world`, `:158-183`): true gate = parity of two or three history bits; two experts = one- or
  two-bit affine parities with different masks (different, not complements, disagreeing on exactly 8 of the 16
  `x`); labels always produced by the *history* gate; no noise. The first 48 distinct inputs are all *aligned*
  (`s = g(h)`); then 16 new inputs in one of three continuations: `aligned` (all `s = g(h)`), `balanced` (8 aligned
  + 8 reversed, shuffled), `reversed` (all `s ≠ g(h)`). The true function never changes — only which inputs are
  observed. The learner gets neither the boundary nor the regime label.
- **Units:** 24 lives × 3 continuations × 7 methods = **504 trajectories**, 4,032 snapshots, 32,256 pre-label
  predictions; unit of independence = 24 lives; paired.
- **Seven methods** (`PROTOCOL.md:47-53`): `full_table` (counts per full input; 1/2 on unseen); `pooled_x` (counts
  per `x` only); `surface_only` (gate fixed to `s`, 32² expert pairs weighted); `history_only` (15 history gates ×
  expert pairs — the true gate is not given, but *the prior knowledge that the surface is irrelevant is*);
  `joint_uniform_gate` (16 gates equal: families 15/16 vs 1/16); `joint_balanced_family` (families 1/2 vs 1/2; same
  16,384 structures, same update — only the prior differs); `oracle_gate` (true gate given; privileged yardstick).
- **Measures:** `(p−y)²` on the full 512 cube, the 256 aligned inputs, the 256 reversed inputs, and the
  not-yet-seen inputs; total posterior mass of the true gate and of the surface gate; pre-label error.
- **Extension:** 24 lives × 2 priors × 2 policies = **96 trajectories**, 16 queries each = 1,536 external labels,
  starting from the common 48 aligned records; snapshots after 0/1/2/4/8/16 queries.
- **Statistics:** descriptive paired counts over 24 lives; no intervals.
- **Verification:** all 8,388,608 bank values against an independent scalar bit computation and the count
  15,904 (`verify_context.py:38-43`); for all 72 worlds: equal prefix across continuations, 0/8/16 reversed
  records in the suffix, experts disagreeing on 8 `x`, the true structure and its *matched surface structure*
  agreeing on the whole prefix and differing on exactly 128 inputs, all of them in the reversed half (`:44-60`);
  the two priors (`:61-64`); the exact odds law `prior_ratio·19^k` step by step (`:65-77`); state isolation of
  exported/restored learners (`:78-84`); 21 fresh-process continuations equal to the uninterrupted object
  (`:101-124`); recheck of all 504 runs (`:127-153`); full exact rerun. For the extension
  (`verify_active.py`): 1,536 choices replayed and their labels scalar-checked, 768 greedy choices re-scored
  independently (`:30-68`); 4 fresh-process continuations restoring learner + seen mask + RNG state, which then
  **select the same next queries** (`:71-101`).

### (4) Results (the study's numbers) and decision labels

**Fitting the training stream is not finding the right explanation** (`REPORT_TR.md:21-39`). MSE on the 256
reversed-cue inputs:

| Method | after the first 48 (all aligned) | after 8 aligned + 8 reversed more | after 16 reversed more |
|---|---:|---:|---:|
| full_table | .250000 | .242188 | .234375 |
| pooled_x | .179817 | .163687 | .165193 |
| surface_only | .500000 | .500000 | .499905 |
| history_only (surface excluded by fiat) | ≈0 (6.3e−13) | ≈0 | ≈0 |
| joint_uniform_gate | **.125000** | .000053 | ≈0 (5.8e−17) |
| joint_balanced_family | **.439453** | .004155 | ≈0 (1.3e−14) |
| oracle_gate | ≈0 | ≈0 | ≈0 |

- After the 48 aligned records both joint learners are essentially exact on aligned inputs (1.6e−13 and
  4.3e−15) and still carry .125 / .439453 on reversed-cue inputs. **More data of the same kind changes nothing:**
  with all 64 records aligned the two values are again .125000 and .439453 — "Bu, yalnız az örnek sorunu
  değildir" (`REPORT_TR.md:39`) [EMP n=24, and exact by C1/C3 below].
- Posterior mass of the true history gate after the 48 aligned records: **.5** (uniform gates) vs **.0625**
  (balanced families); surface gate .5 vs .9375 (`paired_summary.json`).
- **Separating experience transfers to inputs never seen** (`REPORT_TR.md:49-58`), balanced continuation, MSE on
  unseen inputs: uniform gates **.068966 → .0000298**; balanced families **.242457 → .0023553** (464 unseen inputs
  before, 448 after). Both end at 100 % classification accuracy on unseen inputs while their probability errors
  differ by a factor ≈ 79: "Yalnız doğruluk ölçmek bu farkı gizler." Reversed continuation: 3.15e−17 and
  6.95e−15.
- **Self-selected experience** (`REPORT_TR.md:69-86`), MSE on the fixed 512 cube after n new queries:

  | Prior, policy | 0 | 1 | 2 | 4 |
  |---|---:|---:|---:|---:|
  | uniform gates, random query | .062500 | .047031 | .0289584 | .00809975 |
  | uniform gates, disagreement query | .062500 | .000625 | .00000191 | 1.47e−11 |
  | balanced families, random query | .219727 | .176960 | .1250542 | .04993372 |
  | balanced families, disagreement query | .219727 | .048659 | .00039788 | 3.31e−9 |

  After four queries the disagreement policy is better than random in **24/24 lives** for both priors, on the
  full cube and on unseen inputs. It is *not* better at every step: at the first query 3 lives (uniform) and 6
  lives (balanced) go the other way, and one life at the second query (balanced).
- **Cost** (`REPORT_TR.md:88-102`): candidate–label checks for 64 labels / final learner JSON bytes —
  full_table 0 / 3,186; pooled_x 0 / 208; surface_only 65,536 / 22,644.5; history_only 983,040 / 465,166.25;
  joint (uniform) 1,048,576 / 398,611.75; joint (balanced) 1,048,576 / 488,726.75; oracle 65,536 / 22,023.5; plus
  the shared 67,108,864-byte bank. In the extension both policies make the same 1,048,576 candidate–label checks,
  but selection costs 262,144 candidate–input cells for random and **134,217,728** for disagreement — **512×**.
  "Sonuç dış gözlem verimliliğidir; toplam hesaplama veya enerji üstünlüğü gösterilmedi."

**Verdicts** (`REPORT_TR.md:124`): H42 (context relation and context-dependent rules can be acquired jointly
without the context label) — **destek, yalnız hazır geçmiş/dil içinde**; H43 (fully confounded acquisition cannot
determine the explanation) — **destek** ("aynı bilgiyle ayırt edilemezlik bakımından"); H44 (separating
experience corrects the use relation persistently and beyond the training items) — **destek**; H45 (two models
covering the same functions generalise differently in the ambiguous period because of their initial weights) —
**destek**. "Hiçbiri temsil dilinin kendiliğinden doğuşunu veya genel yapay zekâyı kanıtlamaz."

### (5) Laws / invariants claimed (all labelled known: "Bu olgu bilinen ön olasılık/ayırt edilebilirlik sorunudur", `REPORT_TR.md:47`)

- **C1 — same experience, different worlds [EXACT].** Let `H(h,s,x) = f_{g(h)}(x)` and `S(h,s,x) = f_s(x)`. On every
  aligned input they give the same label; "hizalı altkümenin 256 girdisinin tamamı bilinse de iki açıklama ayırt
  edilemez", and among the 16,384 structures exactly these two fit the whole noise-free aligned subset
  (`MATHEMATICAL_AUDIT_TR.md:39-49`; I re-derived both statements independently, §6 check D). With the two
  worlds weighted equally, any single prediction `p` on an input where they disagree has
  `((p−H)² + (p−S)²)/2 = (p − ½)² + ¼ ≥ ¼`, hence mean MSE **≥ 1/16** if the surface bit is independent and
  uniform at test time (disagreement rate 1/4) and **≥ 1/8** if it is always reversed (rate 1/2) (`:55-77`).
  *Scope:* "tek bilinen gerçek dünya için zorunlu hata değildir" — a learner that always assumes history reaches
  zero in the H world and pays in the S world; the proof only forbids a guarantee in both from the same data.
- **C2 — what counts as evidence [EXACT].** An observation changes the odds between the true history structure
  and its matched surface structure only if **both** `s ≠ g(h)` **and** `f₀(x) ≠ f₁(x)`; each such observation
  multiplies the odds by `(1−ε)/ε = 19`; after `k` of them the odds are `19^k` (uniform gates) or `19^k/15`
  (balanced families); every other observation leaves them unchanged (`MATHEMATICAL_AUDIT_TR.md:92-108`).
  Flipping the surface cue alone is not enough. "Kanıt sağlayan deneyim, rakip açıklamaların sonuçlarını
  birbirinden ayıran deneyimdir." *Scope:* proved for that pair of structures, "bütün geçmiş ailesinin toplam
  ağırlığına doğrudan uygulanmaz" (`REPORT_TR.md:67`).
- **C3 — the prior decides inside the ambiguity [EXACT + EMP n=24].** With identical structures, data and update,
  the odds of the true history explanation against the surface explanation are 1 (gates equal) or 1/15
  (families equal), and aligned records never move them ("bu oranlar öğrenimle değişmez"); hence 50 % vs 6.25 %
  and reversed-cue MSE .125 vs .439453. "'İki aileye eşit yaklaşma', onların içindeki belirli açıklamalara eşit
  yaklaşma değildir" (`REPORT_TR.md:45`). Also: the number of representations is not the number of functions; a
  gate-independent affine function receives 16 times the prior mass of one context-dependent function
  (`MATHEMATICAL_AUDIT_TR.md:33`). Consequence stated by the report: "aynı gözlem başarısından aynı anlam
  örgüsünün öğrenildiği çıkarılamaz" (`REPORT_TR.md:47`).
- **C4 — the structure is identifiable from the whole function, inside the family [EXACT].** Writing
  `f₀ = b₀ ⊕ u₀ᵀx`, `f₀ ⊕ f₁ = δb ⊕ δuᵀx`:
  `F(h,x) = b₀ ⊕ u₀ᵀx ⊕ δb·(aᵀh) ⊕ Σ_{ij} a_i δu_j h_i x_j`. The matrix of `h_i x_j` interaction coefficients is
  `M = a δuᵀ`, non-zero and of rank 1 over F₂, whose factorisation is unique (the only non-zero scalar is 1); the
  slice `h = 0` gives `f₀`, `M` gives `a` and `δu`, the remaining linear history coefficients give `δb`
  (`MATHEMATICAL_AUDIT_TR.md:110-122`). *Scope:* "verilen aile içinde"; requires different, non-complementary
  experts; says nothing about finite training already knowing the cube.
- **C5 — processing vs acquiring [conceptual, with C1 as its proof]** (`REPORT_TR.md:110`): the bound C1 holds
  under "the same information". When the system obtains a genuinely new, separating external label, that
  condition no longer applies. "Mevcut bilginin işlenmesi ile yeni dış bilginin edinilmesi ayrı süreçlerdir."
- **C6 — choosing the experience [EMP n=24].** Querying where the current belief disagrees with itself most
  reaches after 4 external labels what random querying has not reached after 16 (1.47e−11 vs 2.4e−7 full-cube
  MSE, uniform gates; `paired_summary.json`), 24/24 lives at 4 queries, not at every single step.
- **C7 — observation efficiency is not computation efficiency [EMP]:** 512× more selection cells for the same
  16 labels.

### (6) Limits
*Admitted by the study.* The history vector is supplied; the structure language and bank are supplied;
`history_only` receives the privileged exclusion of the surface gate and "bağlam açıklamasını veriden tek başına
bulmakla aynı değildir" (`REPORT_TR.md:37`); priors are over representations, not functions; ε = .05 is an
assumption about a noise-free world; the full-cube measure includes seen inputs; full-table counters do retain
event information; the query interface ("512 girdiden istenen herhangi birini doğru etiketle sorgulamaya izin
verir") is a strong supplied privilege; "Yanlış aday dili içindeki yüksek belirsizlik, doğru deneyimi seçmenin
evrensel ölçütü değildir" (`ACTIVE_EXPERIENCE_PROTOCOL.md:22`); this is not a new active-learning method
(`REPORT_TR.md:86`). And the decisive one (`REPORT_TR.md:120`): "Tur 14'te R/P/U ve seçim kuralının biçimi hazır;
yalnız bunların içindeki inanç durumu değişiyor. Aktif sorgu kararı değişen inançtan etkileniyor, fakat yeni bir
sorgu dili veya güncelleme kuralı oluşmuyor."

*Limits I see [ANALYST].*
1. The setting is realizable and tiny: the truth is one of 16,384 enumerated structures. Nothing is known about
   the behaviour of the ε-mixture when the truth is outside the language.
2. After the 48 aligned records the learner is in effect a *two-hypothesis* learner. Over all 48 balanced-regime
   runs the recorded true-gate mass after 64 records equals the closed form `r·19^k/(1 + r·19^k)` (`r = 1` or
   `1/15`, `k` = number of discriminating records in the suffix) within 1.7e−9 (§6 check E). So from then on the
   entire learning process is **one number** — the log-odds between two rival structures — and the measured
   table entries follow in closed form: `.125 = (1/2)²·(1/2)`, `.439453 = (15/16)²·(1/2)`.
3. Only a fraction of "cue-reversed" experience is evidence: 3.92 of the 16 suffix records are discriminating on
   average in the balanced continuation (range 1–8) and 8.17 in the reversed one (range 6–13) (§6 check E), as C2
   predicts (experts disagree on half the `x`).
4. `p(1−p)` is the variance of the mean prediction — a one-step, myopic score — and the policy needs a pool it can
   enumerate and an oracle for arbitrary `(h, s, x)`. A stream cannot be asked for an arbitrary history.
5. The state that buys the generalisation is two orders of magnitude larger than the table baseline (398–489 KB
   vs 3.2 KB JSON, plus the 67 MB bank).
6. No baseline of a different kind (e.g. a learned continuous representation) was run; "strong" here means the
   oracle and the prior-privileged `history_only`.

### (7) Implications for a core where learning and running are one process
- **Mechanism to reuse.** The error-counter belief: `log-weight = log prior + (#contradictions)·log(ε/(1−ε))`,
  prediction = belief-weighted mean of structure outputs, update = integer increments. It is graded, never
  zero, additive, order-free, mergeable, exactly restartable, and it needs no out-of-model fall-back. This is the
  group's concrete instance of "a weight is a graded belief" — with the caveat that the weight belongs to a
  whole *structure*, not to a link.
- **Mechanism to reuse.** Seek the experience on which held rival structures disagree; reveal the outcome only
  after the choice; persist learner + seen mask + RNG so that a restart selects the same next experience.
- **Requirement (from C1, C3).** Fit on experienced inputs is non-diagnostic. A construction must be scored on
  *separating* cases — inputs where a named rival predicts differently — and the rival must be named. The core
  needs an explicit record of "which other structure explains everything I have seen equally well".
- **Requirement (from C3).** The initial weighting of structures is part of the model and decides behaviour in
  every ambiguous period; it must be stated (per representation / per function / per family) and varied.
- **Requirement (from the 100 % vs 79× result).** Do not judge beliefs by the arg-max decision; keep and score
  the probability. (The study's own measure choice: MSE/Brier with expected accuracy under ties, never a point
  pick.)
- **Requirement (from C5).** Separate in the design, and in the accounting, internal processing of what is
  already known from acquisition of new external information. Imagination can order hypotheses and choose where
  to look; it cannot remove an ambiguity that the data leave.
- **Warning (from C7).** Self-selected experience bought a ≥ 4× saving in labels at 512× selection compute in a
  512-input world. The cost scales with the pool the system imagines over.
- **Warning.** "Uncertainty" computed inside a wrong or incomplete structure language is not a trustworthy guide
  to which experience is informative — stated by the protocol, untested.
- **Open question.** Everything the study lists as not done: extracting the history summary from a stream (taken
  up by round 15, temporal — another group), forming the gate/expert language, forming a query language or a
  new update rule, real access costs for experience.

### 3B. `CONTINUOUS_SYSTEM_TARGET_TR.md` — the target, stated as a computation frame

Not an experiment, not an architecture; written independently of the round-14 results (`:5`). It is the
programme's clearest statement of "learning and running are one process" and should be read as the definition
the new core is held to.

- **Target** (`:11`): "Hizmet veren, deneyim edinen, deneyimini işleyen ve sonraki işleyişini bu deneyim nedeniyle
  değiştiren süreç aynı programın yaşamı içinde süreklilik taşımalıdır." Dependence on a researcher building a
  new training set and restarting the system for every acquisition is to be reduced.
- **Training and inference are two functions in one life** (`:13`): "Inference mevcut durumdan bir çıktı üretir;
  öğrenme sonraki çıktıları veya sonraki öğrenmenin biçimini etkileyen durumu değiştirir." They may alternate,
  run on different time scales or concurrently; gradients at run time are compatible in principle but not
  required.
- **"Continuous" is not "change a weight at every token"** (`:15`): no obligation to make every experience
  permanent at once or to keep all records forever; consolidation may come later. Required: learning is part of
  the running system's life cycle and an accepted change is usable afterwards.
- **Frame** (`:21-29`), roles not modules:
  `(a_t, r_t) = A(S_t, o_t; B_t)`, `e_t = (o_t, a_t, o_{t+1}, f_t)`, `S̃_{t+1} = U(S_t, e_t)`,
  `S_{t+1} = C(S_t, S̃_{t+1}; V_t, B_t)` — `S` learning-relevant state, `o` observation, `a` action or
  information-seeking choice, `r` output to the current job, `f` external feedback if any, `B` resource budget,
  `V` accessible evidence for judging a change, `U` proposes the change, `C` decides how much of it enters the
  working state and under which condition.
- **Reload test** (`:33`): must check not only the same answer but that learning from the same next experience is
  preserved, "Çünkü aynı bugünkü tahmini üreten iki iç durum, yarın aynı deneyimden farklı şeyler öğrenebilir."
- **Own output is not evidence** (`:37`): "Sistemin kendi ürettiği cevabı dış kanıt yerine koyması yeterli
  değildir."
- **Nine goals with their current evidence and open criteria** (table `:41-51`): learning while running; acquiring
  its own experience; forming its own contexts; acquiring new knowledge and skills ("Sadece verilen adaylar
  arasında seçim yapmanın ötesinde, yeni kullanılabilir yapılar oluşturmak"); developing new learning goals;
  using its uncertainty ("Eksik veya yanlış aday dili altında belirsizliğin güvenilirliğini sınamak"); changing
  while keeping working capacity; survival of what was learned when raw experiences are removed; self-driven
  progress ("ölçütü değiştirerek veya kolay örnekleri seçerek görünür başarı üretmesini ayırmak").
- **"Kendi kendine" is not "from nothing"** (`:53`): what is given at the start must be listed; what arises from
  experience must be shown separately.
- **Continuous learning is not continuous improvement** (`:55-61`), two counter-examples: if the next label is an
  independent fair bit, no past experience determines it; a fixed finite memory cannot keep all details of
  unboundedly many arbitrary histories. Hence the right goal is a principle that explains "ilerlemenin ortaya
  çıktığı koşulları, kalıcılığını, bedelini ve başarısızlık sınırlarını".

**Implication for the core.** The `U`/`C` split is a design requirement worth adopting literally: a proposed
change and its *acceptance into working state given evidence `V` and budget `B`* are different operations. In
round 18's vocabulary (§4) the proposal is the imagined result and the acceptance is the verified one.

### 3C. `THEORY_NOVELTY_BOUNDARY_TR.md` — what a new theory would have to explain

Written in answer to the owner's Q6 ("… sinir ağlarının çalışma prensiblerini yamalamak gibi bir niyetim yok
günün sonunda bambaşka bir buluş yapmak niyetimiz var", context `REPORT_TR.md:116`). "Yeni bir buluş istemek
araştırmanın yönüdür; buluşun gerçekleşmiş olması bir sonuç iddiasıdır" (`:5`).

- **Scope-keeping notation** (`:9-21`): `z_t = (R_t, P_t, U_t, Q_t, m_t)` — `R` the structure used to parse and
  represent experience; `P` the executable operations and their composition rules; `U` the procedure by which
  experience changes persistent state; `Q` the procedure choosing which experience to seek or which internal
  computation to do; `m` other persistent information. `a_t = Q_t(z_t, o_t; b_t)`,
  `e_t = (o_t, a_t, o_{t+1}, f_t)`, `z_{t+1} = U_t(z_t, e_t; b_t)`. "Bu eşitlikler henüz açıklayıcı bir kuram
  değildir" (`:25`).
- **Two guards against empty novelty** (`:27`): "Aynı cevapları üreten iki durum, sonraki deneyime farklı tepki
  veriyorsa öğrenme bakımından eşdeğer değildir. Ancak her öğrenmenin yeni düğüm veya yeni işlem gerektirdiği de
  varsayılmamalıdır." and "Bütün z bir parametre dizisine kodlanabiliyorsa 'parametre değişimi' ile 'yapısal
  değişim' sözcüklerinden tek başına temel bir ayrılık çıkarılamaz; ayrılık erişilen işlev, öğrenme davranışı ve
  kaynak maliyetinde gösterilmelidir." The report adds (`REPORT_TR.md:118`): "aynı işlevsel değişimin farklı veri
  yapılarında ifade edilmesi tek başına yeni hesaplama ilkesi değildir."
- **Nearest prior work and what it leaves open** (`:31-37`): *DreamCoder* (Ellis et al., PLDI 2021) — library
  components extracted from solved tasks and reused inside later components, a neural search policy trained
  alongside; given: base language, task specifications, the library-building procedure; it changes
  *reachability under bounded search*, not computability. So "uyku sırasında deneyimden kütüphane öğrenme" alone
  cannot be the new claim. *Gödel machines* (Schmidhuber 2006) — self-rewrites applied when proved useful under
  an axiomatised utility, hardware and environment; no guarantee of finding all improvements or of unending
  progress. So "sistem kendi öğrenme kodunu değiştirsin" alone is not novelty. *Learned optimisation*
  (Andrychowicz et al., NeurIPS 2016) — an LSTM update rule trained across tasks, frozen at evaluation, failing
  to transfer from sigmoid to ReLU networks. So parametrising the learning rule alone does not answer the
  question either. Three full texts were read at the cited sections; no claim of a complete literature search.
- **Five criteria any claimed new principle must meet** (`:41-45`): (1) *explicit assumptions and a
  counter-example* — environment class, accessible information, feedback, budget; non-identifiability of two
  worlds with the same past is a limit on the guarantee, not a success of the theory; (2) *a discriminating
  prediction* — what changes when the principle is removed, stated in advance; (3) *a strong comparison with
  the same information* — a fixed but sufficiently wide representation, a program library, a learned updater,
  experience selection, with preparation, search, verification and storage costs included ("Eşit etiket sayısı,
  eşit toplam kaynak demek değildir"); (4) *continuity while changing* — old competence, new gain and
  persistence after restart verified separately, including the updates in between; preservation must not turn
  into never correcting wrong knowledge; (5) *transfer to a third task* — task 1 builds the structure, task 2
  changes it, an unseen task 3 measures the effect, with useful and misleading commonalities both present;
  renamed instances of one family do not count as broad competence.
- **Where round 14 stands** (`REPORT_TR.md:120-122`): only the belief inside fixed `R/P/U/Q` changes. The next
  theoretical obligation: "**verilen adayların içinden doğruyu seçmeyi aşan, deneyimden kullanılabilir bir
  hesaplama yapısı oluşturmanın ne olduğudur**".

**Implication for the core.** This note is the bar the new initiative has already set for itself. Leaving the
Transformer / embedding / softmax data structures is, by the owner's own earlier standard, not a result; the
result has to be a difference in reachable function, learning behaviour or resource cost against the three
named neighbours, under the five criteria.

---

## 4. Study `living_learning_interactions_2026_09_22` (round 18) — changes that only work together, and the price of unconditional persistence

### (1) Question
Given a repair schema acquired earlier and eight places in the system's own program where it could be applied:
do the measured effects of *single* edits suffice to find which edits must be applied *together*? And once a
task has taught the learner that a richer (pairwise) effect model was needed, what does carrying that preference
permanently into later tasks cost? (`PROTOCOL.md:5-9`.) The protocol restricts it: "ana yaşayan öğrenme
probleminin yalnız iç değişiklik atama yönüdür" (`PROTOCOL.md:9`).

### (2) Mechanism / state representation actually used
Symbolic program edits over F17; measured finite-difference effect vectors; one persistent integer. File:
`research/living_learning_interactions_2026_09_22/interaction_learning.py` (helpers imported from round 17,
`research/living_learning_repair_2026_09_22/repair_learning.py`).

- **Program language.** ASTs with `add/sub/mul` over F17, variables `x`, `y`, integer constants
  (`repair_learning.py:15,43-51`); the learner sees its own candidate AST, 16 distinct `(x, y, output)` examples
  and the acquired schema; it does not see the target AST, the right mask, the task family or the 289-point
  truth table (`PROTOCOL.md:15`).
- **The object inherited from experience.** Round 17's three-hole rule `(A+B)*C → A*C+B`, read from that round's
  verified result (`source_states`, `:170-177`); its acquisition cost of 6,158 work units is charged to every
  method (`:176,181`). It is *not* an algebraic identity — "kanıta göre önerilen bir değişikliktir"
  (`REPORT_TR.md:13`).
- **Edit sites.** `locate` (`:43-50`) matches the rule's left side at every AST path; this world yields exactly 8
  disjoint, non-nested sites (asserted, `:57-58`). A candidate is a bitmask over the 8 sites.
- **Real effect of a change.** `query(mask)` builds the edited AST (`candidate`, `:62-65`), executes it on the 16
  evidence inputs, caches the output vector, and raises `Solved` when it equals the labels (`:67-74`).
- **Measured effect model** (`build_effects`, `:76-88`), computed on the system's *own* program and only on the
  accessible inputs: `d_i = f_{i} − f_∅ (mod 17)` (`:81`); `d_ij = f_{ij} − f_i − f_j + f_∅ (mod 17)` (`:87`).
  Order 1 needs 8 real executions, order 2 needs 28 more.
- **Imagined effect of a change** (`modeled`, `:90-96`): `f̂(mask) = Σ_{S ⊆ mask, S measured} coefficient[S] (mod 17)`
  — a prediction of what the edited program *would* output, with no execution.
- **Search = imagine, then verify** (`effect_search`, `:98-103`): scan all 256 masks in order (number of edits,
  then value; `:21`); whenever the imagined output equals the labels, **execute the real edited program**
  (`:102`); it is accepted only if the real output matches; otherwise `prediction_mismatches += 1` (`:103`).
- **Methods** (`solve`, `:106-136`): `bulk_all` (apply all eight, `:111`); `exact_subset` (execute all 256 masks,
  `:112-113`); `greedy_single` (flip the single bit that strictly lowers the count of mismatching outputs, up to
  8 rounds, `:114-122`); `first_order`; `second_order`; `adaptive_reset` and `adaptive_persistent` (`:123-129`):
  `order = 2 if second_order else start_order if adaptive_persistent else 1`; when an adaptive method finds no
  verified candidate at order 1 it sets `expanded=True`, the persistent one sets `final_order=2`, and both
  continue with `effect_search(2)` re-using the task's measurements.
- **What persists across tasks** (`run_life`, `:180-199`):
  `state = dict(order=1, rule, library=[], total_work=acquisition_cost, expanded_tasks=[])` (`:181`);
  `state['order'] = result['order_after']` (`:187`); every accepted program with its examples is appended to
  `library` (`:195`) and **never consulted again**. Effect coefficients are re-measured in every task. So the
  only thing experience changes for the future is one integer in {1, 2}, and it can only go up.
- **Work accounting** (`Work`, `:26-30`; per-task limit 1,000,000, `:20`; atomic refusal): nodes of every built
  candidate (`:64`), nodes × 16 per execution (`:70`), 16 per single-effect and 48 per pair-effect subtraction
  (`:80,86`), 16 × (number of sub-mask coefficients) per imagined mask (`:92`), plus traversal and matching
  charges. "Sanal iş, gerçek CPU veya RAM değildir" (`PROTOCOL.md:49`).

### (3) Experimental design
- **A life** (`make_life`, `:153-167`): 16 tasks. Every local site starts as `(u+v)*w` with `u, v, w` affine in
  `x` or `y` (`:150,158-161`); the rule proposes `u*w+v`. `combine(parts, degree)` multiplies consecutive groups
  of `degree` sites and adds the groups (`:139-147`). Order (`:156`): tasks 0–3 **additive** (sum of the 8
  locals), 4–7 **pairs** (sum of four pair products), 8–11 **additive again**, 12–15 **triples** (two triple
  products + one pair product). The target is the program with 4 of the 8 sites edited, chosen at random
  (`:162`); 16 evidence inputs drawn without repetition from the 289 points of F17² (`:164`).
- **A planted negative witness**: task 4 has every `w = 0` (`:160`) and edits both sites of the first two pairs
  (`:162`) — the start program and every single edit output zero everywhere; only joint edits of a pair produce
  anything. "Bu sıra ve tanık deneyci tarafından kurulmuştur; doğal yaşam dağılımı olarak yorumlanmaz"
  (`REPORT_TR.md:15`).
- **Units:** 24 lives × 7 methods = **168 paths**, 2,688 target attempts; 384 targets per method, 96 per phase.
  Independence is at the level of the 24 lives; methods are paired.
- **Outside the learner:** every accepted program is tested on all 289 inputs (`:190-194`); the result never
  flows back.
- **Protocol** written before the performance run and unchanged; the coverage diagnosis
  (`diagnose_coverage.py`) is explicitly post-run.
- **Verification** (`verify_interactions.py`): an independent scalar interpreter of edited programs (`:23-38`);
  for three representative tasks 3,072 intervention outputs and 768 complete Möbius reconstructions, all
  coefficients above the task's degree zero, a non-zero third-order coefficient in the triple task (`:41-62`);
  the witness — start and all eight single edits are zero on all 289 inputs while the target is not (`:63-66`);
  atomic budget (`:67-71`); recheck of all 168 runs: **94,110** real probes re-evaluated, **1,744** acceptances
  re-tested on the full field, cost categories summed (`:95-127`); 14 fresh-process continuations (7 methods ×
  2 checkpoints) reproducing the rest of the life exactly (`:130-143`); exact full rerun.

### (4) Results (the study's numbers) and decision labels

**Full-field correctness** (`REPORT_TR.md:21-29`; `results/summary.json`), correct targets out of 96 per phase:

| Method | first additive | pairs | additive again | triples | total /384 | mean work per life |
|---|---:|---:|---:|---:|---:|---:|
| bulk_all | 0 | 0 | 0 | 0 | 0 | 88,186.00 |
| greedy_single | 2 | 8 | 3 | 21 | 34 | 667,716.71 |
| first_order | 96 | 18 | 96 | 0 | 210 | 598,782.58 |
| second_order | 96 | 96 | 96 | 84 | 372 | 1,733,919.83 |
| adaptive_reset | 96 | 96 | 96 | 84 | 372 | 1,217,804.83 |
| adaptive_persistent | 96 | 96 | 96 | 84 | 372 | 1,450,551.83 |
| exact_subset | 96 | 96 | 96 | 96 | 384 | 4,820,016.29 |

- No task hit the 1,000,000 limit (maximum 396,593), so the differences are differences of *proposal coverage
  and its price*, not of budget cut-off (`REPORT_TR.md:33`).
- **All 1,744 acceptances pass the full field; no false acceptance.** Four acceptance records carry a mask
  different from the planted one — one task (life 14, task 7) solved by four methods with a program equal on
  all 289 inputs: "Doğru davranış ile hedefin iç yapısını birebir yeniden bulmak ayrıldı" (`REPORT_TR.md:35`).
- **The witness** (task 4, 24 lives): `first_order` and `greedy_single` solve **0/24**; `second_order`, both
  adaptive methods and `exact_subset` solve **24/24** (`REPORT_TR.md:39`; `summary.json › groups › points[4]`).
- **Required order depends on the target mask, not on the family** [post-run diagnosis, exact on the recorded
  data; `diagnose_coverage.py:15-41`]: `first_order` solves 18 pair-phase targets — exactly those whose mask
  contains no complete pair; `second_order` misses 12 triple-phase targets — exactly those whose mask contains a
  complete triple, for which the second-order expansion has a non-zero residual on the 16 examples;
  `exact_subset` solves those 12.
- **Persistence** (`REPORT_TR.md:47-61`): the persistent method switches 1→2 once per life, at the witness (24
  switches); the resetting method expands 174 times. Accuracy and the sequences of accepted masks are identical
  for `second_order`, `adaptive_reset` and `adaptive_persistent` in every life. The persistent method is **more
  expensive in 24/24 lives**: mean +232,747 work units, ≈ +19.1 %.

  | Phase | persistent − resetting, mean work per life | lives in which persistent is cheaper |
  |---|---:|---:|
  | first additive tasks | 0 | 0/24 (all equal) |
  | pair tasks | +10,939 | 9/24 |
  | additive tasks again | +303,728 | 0/24 |
  | triple tasks | −81,920 | 24/24 |
  | whole life | +232,747 | 0/24 |

- **Recorded wrong model predictions: 0.** Order expansion came from *not finding* a verified candidate, never
  from an imagined result being refuted by execution (`REPORT_TR.md:67`).

**Decision labels in the report.** "'ikinci derece her yerde yeterlidir' sonucu **reddedilir**; 'üçlü bir yapı
varsa bütün hedefler üçüncü derece gerektirir' sonucu da **reddedilir**" (`:43`); "'Aynı tür görevlerde her zaman
kazandırır' iddiası da **desteklenmedi**" (`:59`); "'Kendi yanlış tahminini doğrudan çürüttü' anlatımı bu sonuç
için **kullanılmamalıdır**" (`:67`); "Geniş bir 'ilk kez' veya yeni temel computational ilke iddiası yoktur"
(`:71`). The protocol's expected discriminators (`PROTOCOL.md:53-55`) — single effects exact for additive
structure, cross terms under products, fixed second order not universal, a persistent second-order preference
saving on same-kind tasks and wasting on return to simple ones — were all observed.

### (5) Laws / invariants claimed ("klasik sonlu fark / Möbius açılımıdır. Yeni kuram olarak sahiplenilmez", `MATHEMATICS_AND_PRIOR_ART_TR.md:5`)

- **I1 — effect expansion [EXACT, classical].** For edit set `A ⊆ {1..n}` and input `x`,
  `m_x(S) = Σ_{T⊆S} (−1)^{|S|−|T|} f_x(T)` and `f_x(A) = Σ_{S⊆A} m_x(S)`; proof by
  `Σ_{T⊆S⊆A} (−1)^{|S|−|T|} = (1−1)^{|A|−|T|}`; uses only addition and subtraction, so it holds in F17
  (`:7-30`). The degree-≤k truncation is correct **on all masks** iff every coefficient above `k` vanishes;
  being correct **at the one mask that is needed** is a weaker condition (`:32-40`).
- **I2 — a change that is invisible alone can be necessary [EXACT, constructive witness].**
  `(a_i + z_i d_i)(a_j + z_j d_j) = a_i a_j + z_i d_i a_j + z_j d_j a_i + z_i z_j d_i d_j`; with `a_i = a_j = 0` no
  single edit changes the output and both together give `d_i d_j`. The zero function and `z_i z_j` agree on the
  empty mask and on every singleton, so "Yalnız bu ölçümlere erişen bir yordam ikisini genel olarak ayıramaz"
  (`:42-51`). "Bu karşı örnek 'tek tek işe yaramıyor, o hâlde birlikte de gereksiz' çıkarımını çürütür." For
  three factors the term `z_i z_j z_k d_i d_j d_k` is invisible to every smaller intervention: "Sabit düşük
  derece, keyfî yüksek dereceli değişim etkileşimlerinin evrensel açıklaması değildir" (`:55`). *Scope:* a known
  observability limit of finite differences; with knowledge of the internal structure, or other evidence, the
  limit does not apply in the same way; no claim that every learner needs such a model.
- **I3 — "the program has an order-3 interaction" is not "this target needs order 3" [EXACT argument + EMP
  n=24]** (`:40`; `REPORT_TR.md:41-43`): 84 of 96 triple-phase targets are found with order 2; 18 of 96
  pair-phase targets with order 1.
- **I4 — effect, correctness and origin are three things** (`:59-65`): `f_x(A)` is the result of the system's
  own candidate; agreement with the labels is the accessible success measure; two different masks can produce
  the same output everywhere, so finding a correct program is not diagnosing which internal places were
  changed. Expansion on "no candidate found" may be due to an insufficient model, an insufficient edit family or
  a restricted search — "'öğrenici etkileşim derecesinin doğru olduğunu kanıtladı' denmez".
- **I5 — persistence must be accounted over the life [identity + EMP n=24]** (`:67-79`):
  `ΔC = Σ_t (C_persistent,t − C_reset,t)`; "Bu özdeşlik bir yeni optimizasyon teoremi değildir", it forces one to
  say in which future tasks work was skipped and in which it was done needlessly. Measured: +232,747 per life,
  24/24.
- **I6 — the supported narrow proposition** (`:93`): "**İç değişikliklerin tekil etkileri, bazı birlikte
  düzeltmeleri açıklamaya yetmez; geçmişte yararlı bulunan daha zengin bir değişim modelini kalıcı kılmak da
  bütün yeni görevlerde yarar sağlamaz.** İlk bölüm klasik matematiksel karşı örnek, ikinci bölüm bu protokolde
  deneysel sonuçtur."
- **I7 — two scopes** (`REPORT_TR.md:86`): the contribution to the general problem is separating "öğrenilecek iç
  değişimin kapsamı" from "o değişikliği geleceğe taşımanın kapsamı". "Geçmişten kalıcı bir nesne taşımanın
  varlığı ile kalıcı olarak yararlı gelişme farklı ölçülmelidir."

### (6) Limits
*Admitted by the study* (`REPORT_TR.md:63-71`). `bulk_all` is structurally mismatched to 4-of-8 targets, so its
zero is no evidence of novelty; the greedy loss on exact output equality over a finite field is a weak
heuristic. The meaningful comparison is first/second order, reset, persistence and the representation-matched
exhaustive search; **there is no symbolic analysis/factoring control that reads the interactions directly off
the visible AST**, so "en güçlü bütün olası rakiplerin aşıldığı söylenemez". Edit sites are disjoint, eight in
number, operations are known arithmetic, feedback is correct and task-bound; noise, delay, nested
transformations, an unknown observation language and unbounded program families were not tested. The persistent
object is "bir bitlik inceleme tercihi"; no interaction language is invented; coefficients are not transferred;
the library is not used. Not a continuously interacting system; preservation of old skills in a new common
representation was not measured. Virtual work is not CPU, energy or RAM. Two helper-script failures during the
post-run diagnosis are recorded and were caught before any success record was produced (`:82`).

*Limits I see [ANALYST]* (numbers from §6 check F).
1. The negative-persistence result is a product of **cost asymmetry × task mix**, both fixed by the design.
   After the switch (tasks 5–15: 264 task instances) 114 are solvable at order 1 and 150 need order 2 or more.
   On the former the persistent preference pays the 28 unneeded pair measurements: +75,947 work on average
   (72,644–79,268). On the latter it saves exactly the failed first-order scan, 20,480 work
   (`16·Σ_mask (1 + |mask|)`), every time. The 114 over-modelled tasks cost 8,657,928 in total and the 150
   saved scans return 3,072,000; the difference, 5,585,928 = 24 × 232,747, is the reported mean to the unit.
   The break-even share of order-2 tasks is 75,947/(75,947 + 20,480) = **78.8 %**; the life contained
   **56.8 %**. With a different mix, or cheaper measurements, the sign flips.
2. Over-modelling is expensive because of *real executions*: 84 % of the extra cost on the returning additive
   tasks is `evaluation_nodes`; the larger imagined-model scan is 8.6 %.
3. The ceiling for any learned scope: a per-task oracle choosing the cheaper of "order 1 first" and "order 2
   directly" would cost 1,069,325 per life — 12.2 % below resetting (1,217,805), 26.3 % below the persistent
   preference (1,450,552), 38.3 % below always-order-2 (1,733,920). That is all that "learning when the richer
   model applies" (H18-A below) can gain in this world.
4. The persistent preference is the weakest possible form of "experience changes future computation": one bit,
   global, irreversible, unconditioned. The result refutes *unconditional* persistence, not persistence.
5. The imagine-then-verify gate was never exercised in the negative direction (0 mismatches, 0 false
   acceptances), so its protective value is not demonstrated here; the family (4-of-8 masks, 16 examples in
   F17²) is benign.
6. Scalar feedback vs effect knowledge: on the 192 additive targets the greedy single-bit descent on the
   mismatch count solves 5, the first-order effect model solves 192, from the same kind of single-edit
   executions. In this world a partial fix gives no partial credit; what carries information is *what each
   change does to the output vector*, not how good the result was.

### (7) Implications for a core where learning and running are one process
- **Mechanism to reuse.** The imagine-then-verify loop: measure a few real effects of changes to one's own
  computation, predict the outcome of unexecuted combinations from them, execute only the predicted hits, accept
  only on the real result, count the mispredictions. This is the group's cleanest small instance of "the system
  imagines and measures how real its constructions are" — with real numbers for what imagination saves (order-2
  model: 372/384 at 1.73 M work vs exhaustive 384/384 at 4.82 M).
- **Mechanism to reuse.** Model-order escalation on evidence of insufficiency, with measurements re-used inside
  the task; virtual work budgets with atomic charging and per-category counters; whole-life cost with the
  acquisition cost of inherited knowledge charged to everyone.
- **Requirement (from I2).** Credit assignment for internal changes cannot be restricted to single-change
  effects; the core needs a way to represent and test joint effects, and must not prune a change because it
  does nothing alone.
- **Requirement (from I3, and limit 1).** The order of the effect model is a per-situation decision; under an
  asymmetry like the one measured (failed cheap attempt 20,480 vs unneeded rich measurement ≈ 75,947) "cheap
  first, escalate on failure" is the right default, and a remembered preference must be conditional.
- **Requirement (from I5, I7).** Every persisted change to *how the system constructs* carries two scopes — where
  it applies now and where it should be carried — and is evaluated over the whole life against (a) not carrying
  it, (b) always using it, (c) the per-task oracle.
- **Requirement (from I4).** Behavioural correctness and identification of the internal cause are scored
  separately; equal behaviour does not license a claim about which internal change "was the real one".
- **Warning.** Same accuracy can hide a 19 % cost difference; accuracy-only reporting would have shown the
  persistent learner as equal to the resetting one.
- **Warning (limit 6).** A scalar success signal can be nearly uninformative where structural effect
  measurements are sufficient.
- **Open question.** H18-A/B/C below; and the missing symbolic competitor.

### 4B. `WHOLE_PROBLEM_AND_NEXT_TEST_TR.md` — what is in hand after round 18, and the next discriminators

Not an experiment (`:5`). Main question restated (`:7`): "**Bir sistem, yaşadığı deneyimler nedeniyle
gelecekteki hesaplama ve davranış kapasitesini kalıcı, kontrollü ve genellenebilir biçimde nasıl
değiştirebilir?**" The note warns that the interaction order of round 18 "kullanıcı tarafından anlatılan anlam
uzayının veya zihinsel gelişimin karşılığı olarak kabul edilemez" (`:9`).

**What remains in hand, with its status** (table `:13-20`):

| Kept | Status | What does not generalise |
|---|---|---|
| If experience affects future behaviour or learning, accessible persistent state mediates that effect | general causal distinction from earlier work | what the state is and how it changes usefully does not follow |
| Same answer today is not same learning tomorrow | concrete witnesses in the earlier mathematics and in round 12 | storing more data or a bigger representation is not automatically the solution |
| Correctness of a new representation and transporting the running old state into it are separate | round 16 and earlier transport conditions | the procedure for small given state families is not a general transport algorithm |
| Acquiring a repair schema from past solutions can change later search | round 17's finite program family | a strong bulk-edit control gave the same gain; no new principle |
| Joint effect cannot always be derived from single-change effects | classical mathematics + round 18's constructive witness | no conclusion that every learner must use an effect polynomial or a separate intervention module |
| A richer search preference carried from experience can create a lasting extra cost | experimental negative result, 24 paired lives | not "all persistence is harmful", not "every context change requires forgetting", not "the right scope can always be learned" |

"Birleştirildiklerinde hazır bir mimari veya yeterli genel kuram oluşturmazlar" (`:22`).

**The gap named** (`:28-30`): rounds 12, 13 and 14 tested composition, a durable rule and context separation *in
supplied languages*; repeating them and saying "kavram oluşumunu ilk kez gösterdik" would be wrong. Still
missing: "**işe yarayan ayrımın kendisinin geçmiş başarısızlık ve aktarım deneyiminden oluşturulması**". Giving
the right concept name, which task goes to which branch, all ready gates, or the new correct operation pattern
in advance does not close it. A base execution language is still needed; "hedef, hesaplanabilirlik sınıfını
büyütmek değil, geçmiş deneyimin aynı kaynakla hangi yeni yapıların kurulmasını mümkün kıldığını ayırmaktır."

**Three hypotheses to test** (`:34-38`), related but not equivalent:

- **H18-A — the scope of a change is itself an object to be learned.** Instead of keeping an internal procedure
  globally, acquire from experience *under which accessible structural conditions* it is useful. Round 18 showed
  the insufficiency of the global preference, not this positive hypothesis. It must go beyond choosing a ready
  context gate (round 14), and the competitors must include a symbolic analyser that can read the AST.
- **H18-B — reorganisation must not be judged by today's output alone.** Experiences held together may need to
  be separated when they serve different later acquisitions; structures held apart may be re-used in a common
  computation. Today's accuracy is to be measured together with acquisition, correction and transport cost on
  the same future evidence. "Ağaç veya grafik zorunluluğu varsayılmamalı: aynı işlemleri sağlayan düz bir
  kodlama eşit sonuç veriyorsa biçimsel yenilik iddiası geri çekilmeli."
- **H18-C — development over a long life may not be the sum of local gains.** With acquisition, storage,
  transport of old abilities and removal of wrong transfer priced in, some short-task gains vanish. Needed: a
  whole-life budget instead of a per-task one, a separate memory measure, and future tasks set aside in
  advance.

**Obligations for the next experiment** (`:42-50`): situations answered correctly by one common explanation at
first must later need different future operations in the same visible context; no branch name given; if the
information for the distinction is truly absent, that condition is a *negative control*, not a task to be
solved; later tasks test transfer beyond memorisation and preservation of old computations. Comparisons with
the same past, the same observation rights and whole-life cost: recalling events; strong inference in a ready
rich language; an analyser computing directly from accessible structure; the candidate that uses the part it
built from experience; and the counterfactual that removes only the use of that part. **Falsification criteria
written in advance:** the gain comes only from more stored data or from a ready correct family; the same
procedure as a strong competitor under another name; old skills break; harm in the new context exceeds the
benefit; future acquisition does not change although a representation was added.

**Implication for the core.** This note is the programme's own acceptance test for "concepts formed when
needed": the trigger is a *failure of an explanation that used to suffice*, the product is a distinction not
supplied in advance, and the proof is an ablation of exactly that product on later tasks under whole-life
accounting. Rounds 19 (distinctions) and 22 (crossmap) are, by their own opening lines, the attempts that follow
— see the other groups' digests for what they did and did not achieve.

---

## 5. Cross-study synthesis for the new core

**The group in one paragraph.** The owner described a mind in which concepts split, merge, cross-link and are
re-valued by experience, and in which separately grown bodies of knowledge join through a shared relation to
give new thoughts and fast learning (§1). Three experiments then tested the parts of that picture that could be
isolated *inside supplied languages*: carrying an acquired relation to a renamed situation and composing it
(§2); telling a true context relation from a cue that always accompanied it, and choosing the experience that
separates them (§3); finding internal changes that only work jointly, then carrying the lesson forward (§4).
Each produced one exact limit — a state reduced to its current answers loses the learner (R3); explanations that
always coincide cannot be separated by more of the same experience (C1); single-change effects do not determine
joint effects (I2) — and one measured asymmetry — structure helps only where it fits (24/24 lives in both
directions); the prior decides inside an ambiguity (.125 vs .439); unconditional persistence costs (+19.1 %).
None of them formed a concept, a language or a new way of learning, and all three say so.

### 5.1 One table: which future operation needs which state

This group's instances of the programme's recurring statement that *the family of future operations determines
the state that must be kept*. Each row is backed by an exact witness or a measured result above.

| Future operation the system must still be able to perform | State that suffices | State that is insufficient, with the witness | Source |
|---|---|---|---|
| answer a one-step question about a relation now | the mean (marginal) transition belief | — | §2 (2) |
| chain two *different* relations held with independent beliefs | the two mean matrices — their product is exact | — | R1 [EXACT] |
| use the *same* uncertain relation twice | the joint belief over whole functions | the mean matrix: `E[P²] = I` but `K²` has 1/5 on the diagonal | R2 [EXACT] |
| learn correctly from the next fact about a relation | the surviving structure set, or class identity + facts | the mean matrix: after the same fact `0→1`, `P(1→0)` must be 0 for one history and 1 for the other | R3 [EXACT] |
| carry a relation to newly named objects | the renaming-invariant part (here the cycle type) + a way to match renamings | the source table with its names | §2 (2) |
| keep learning after a fact that no held structure explains | the fact record kept alongside + an explicit model switch | the structure set alone — conditioning is undefined | R5 [EXACT] |
| predict when a correlation that always held in experience breaks | every structure that fits the past equally well, with its weight | the best-fitting single structure; training accuracy (both joint learners ≤ 2e−13 on aligned inputs, .125 / .439 on reversed) | C1, C3 |
| be moved by the next separating observation at the right rate | per-structure contradiction counts (or the log-odds between named rivals) | current predictions only ("Yalnız son tahminleri saklamak, öğrenme durumunun korunduğu anlamına gelmez", context `MATHEMATICAL_AUDIT_TR.md:132`) | C2 |
| choose its own next experience | rival structures' predictions on inputs *not yet experienced* | beliefs about experienced inputs | C6 |
| apply several internal changes together | effect coefficients up to the order present inside that set | single-change effects — all zero in the witness while the pair is required | I1, I2 [EXACT] |
| regroup old events by a distinction not used before | a kept trace of that distinction for the old events | the old grouping | K2 [EXACT] |
| resume after a restart with the same *future* | all of the above + seen-mask + RNG state, checked against the uninterrupted object | equality of current outputs | 63 fresh-process continuations (24 + 21 + 4 + 14), all exact |

### 5.2 What the group says about each design pillar

**(P1) Learning and running are one process.**
*Established as a working pattern:* in rounds 12 and 14 one state object is read by the prediction call and
written by the update call, the prediction is recorded before every outcome, and there is no phase switch
(`relation_learning.py:273-287`, `context_learning.py:203-219`, `active_experience.py:82-98`); in round 18 one
life state is carried through sixteen tasks and changed by the outcome of each
(`interaction_learning.py:180-199`). The definition the programme adopts is in §3B: inference produces an output
from the current state; learning changes the state that affects later outputs *or the form of later learning*.
*Tested property:* continuity of future learning across a process restart, 63 times, always against the
uninterrupted live object. *Not covered anywhere in this group:* an update arriving while a computation is in
flight, long streams, or any change of the update procedure itself.

**(P2) Weights are graded beliefs.**
*The concrete instance* is round 14's contradiction counter per structure with
`log-weight = log prior + n·log(ε/(1−ε))`: graded, additive, order-free, restartable, never zero. *Three
constraints the group proves on any such design:* (i) the belief has to be over structures that keep joint
constraints — per-link marginals fail for self-composition (R2) and for the next update (R3); (ii) the way
hypotheses are counted is itself a substantive choice (R4, C3: 1 vs 1/15 → .125 vs .439); (iii) hard 0/1 belief
leaves the model at the first out-of-language fact (one probability-zero event per affected action in 24/24
lives, §6 check A) and needs a fall-back that discards valid structure (R7). *A reporting constraint:* arg-max
accuracy hid a 79× difference in probability error (§3 (4)).

**(P3) Concepts are formed when needed.**
*No positive evidence in this group.* Every experiment selects inside a supplied language and says so (relations
`LITERATURE_TR.md:41`; context `MATHEMATICAL_AUDIT_TR.md:21`; interactions `REPORT_TR.md:61`). What the group
does provide: (a) the description of the target phenomenon and its acceptance obligations (§1); (b) three
operational *signals of need*, each already computed by existing code — a fact that every held structure gives
probability zero (`relation_learning.py:146-156`), exhaustion of the current model order without a verified
candidate (`interaction_learning.py:126-129`), and two structures that agree on everything experienced yet
disagree elsewhere (`active_experience.py:45-50`); (c) the fact that in each case the response was a pre-written
step (fact table / order + 1 / a query), never a new distinction. The gap is named exactly: "işe yarayan ayrımın
kendisinin geçmiş başarısızlık ve aktarım deneyiminden oluşturulması" (§4B).

**(P4) The system imagines and measures how real its constructions are.**
*Imagination, as it actually occurs here* = computing the consequences of held structures for situations not
experienced: two-step results never taught (round 12), predictions over the whole input cube in order to choose
a query (round 14), outputs of unexecuted edit combinations from a few measured effects (round 18). *Measures of
reality used:* loss on predictions recorded before the outcome; counts of certain-and-wrong predictions; counts
of imagined results refuted by execution; acceptance only after a real run, then an outside full-domain test;
performance on separating cases. *What imagination bought:* all 12 compositions exact after 8.58 target facts
instead of 10.54–12; four chosen queries doing more than sixteen random ones; 372/384 targets at 25–36 % of
the exhaustive cost. *What it cost or risked:* 512× selection compute; certainty that is wrong outside the
structure's scope (15 and 13 per 288; unseen-cell error above the uniform level); and C5 — imagination cannot
remove an ambiguity the data leave, only new external information can. *Rules to keep:* the system's own
output is never a label (§3B); proposal and acceptance are separate operations (`U` and `C`, §3B).

**(P5) The way of constructing improves and transfers.**
*The only experiment in which a way of working is carried forward* is round 18's one-bit model-order preference,
and carried unconditionally it was a net loss (+19.1 % work, 24/24 lives, equal accuracy). *The only experiment
in which a structure is carried to a new situation* is round 12: a gain in 24/24 lives when it fits, a loss in
24/24 when it does not. Neither changes the update procedure `U` or the representation language `R`: "yalnız
bunların içindeki inanç durumu değişiyor" (context `REPORT_TR.md:120`). The programme's own conditions for a
claim here are the five criteria of §3C and obligations (4) and (7) of §1B (ablation with an equal-size
unrelated control; reuse in a third task) — none met yet in this group.

### 5.3 Requirements extracted (each traceable to a result above)

| # | Requirement for the core | Rests on |
|---|---|---|
| REQ-1 | Two states are "the same" only if they respond identically to the same next evidence; every persistence, compression or consolidation step is tested that way, against the uninterrupted state. | R3; §3B reload test; 63 continuations |
| REQ-2 | A belief is held over a structure with its joint constraints (or an equivalent summary such as class identity + facts), never only as per-link strengths. | R2, R3 |
| REQ-3 | No belief is exactly 0 or 1; the record of what happened is kept beside the structure that explains it; a switch of model is an explicit, logged operation; an exception should cost one fact, not the structure. | R5, R7; context ε; §6 check A |
| REQ-4 | How hypotheses are counted (per representation, per function, per family, per class) is declared and varied as a control. | R4, C3 |
| REQ-5 | A construction is scored on cases where a *named* rival predicts differently and on not-yet-seen cases against a structure-free but informed prior; probabilities are scored, not the arg-max. | C1, C3, R6, 79× result |
| REQ-6 | A link carries its relation type: composition rule and condition of use. | K5; R1 vs R2 |
| REQ-7 | The concept store admits several simultaneous groupings over the same leaves and keeps fact content, grouping, evaluation and transient access as separately revisable. | K1; §1A table; H13 |
| REQ-8 | Imagined results propose; real outcomes dispose. Acceptance requires a real result; mispredictions are counted; the system's own output is never used as evidence. | round 18 loop; §3B |
| REQ-9 | Experience is sought where held rivals disagree; the cost of choosing is accounted; the chooser's state (seen-mask, RNG) is part of the persistent state. | C2, C6, C7; `verify_active.py:71-101` |
| REQ-10 | Credit for internal changes covers joint effects; a change that does nothing alone is not pruned; the model order is decided per situation, cheap first with escalation. | I1–I3; §6 check F |
| REQ-11 | Anything persisted about *how to construct* carries its conditions of applicability, is revisable, and is evaluated over the whole life against not carrying it, always using it, and a per-situation oracle. | I5, I7; H18-A, H18-C |
| REQ-12 | Every experiment lists what is given (language, interface, composition rule, query rights, life order), isolates privileged controls in separate code, checks renaming invariance, and fixes its protocol before results. | method of all three studies |
| REQ-13 | A novelty claim meets the five criteria of §3C; a change of data structure, or a tree/graph form that a flat encoding matches, is not a result. | §3C; H18-B |
| REQ-14 | Recall, recombination, new inference and acquisition speed-up are measured separately; a bridge between bodies of knowledge is accepted only with an ablation against an equal-size unrelated link and a third-task reuse. | K7; §1B obligations |

### 5.4 Open questions the group leaves (named by the studies unless marked)

1. Forming the distinction itself from failure and transfer experience, without supplied categories, gates or
   patterns (H1, H6, H14; §4B).
2. Learning the interface through which separately acquired structures compose, including changes of object
   count, abstraction level and arity (H16).
3. Attributing a failed transfer to "source wrong / role mapping wrong / family changed / local exception" from
   evidence, and learning the *scope* of a change (H15, H18-A).
4. Separating event record, interpretation and licensing assumptions in persistent state (H2, H13).
5. Concept vs evaluation; internal state as a source of new experience (H4, H17).
6. Lasting effect without the raw record; meaning of event order (H3, H7).
7. Reliability of uncertainty computed inside a wrong or incomplete language (§3 (6)).
8. Whole-life economics: life-level budget, memory measure, future tasks set aside (H18-C).
9. Change of the update procedure and of the representation language themselves (§3C).
10. [ANALYST] Which compact belief sits between "per-link marginals" (cheap, proved insufficient) and "one
    counter per enumerated structure" (sufficient, exponential) while keeping exactly the joint constraints the
    declared future operations need — the constructive side of R3's scope sentence "uygun bir temsil bu yapıyı
    saklayabilir".
11. [ANALYST] Whether a soft-likelihood version of round 12 (ε instead of hard elimination, the fact table kept
    in parallel) removes the local-exception penalty (R7) without losing the compatible-regime gain. The code
    for both halves exists (`relation_learning.py`, `context_learning.py:88-116`); the combination was not run.

---

## 6. Analyst tiny checks (not part of the owner's studies)

All read-only; scripts and outputs in
`/tmp/claude-0/-workspace-cevahir-ai/79909244-c51d-5307-ad00-1a1e5ef77203/scratchpad/newcore/prior/_checks_g4/`
(`g4_tiny_checks.py` → `g4_tiny_checks.json`; `g4_extra_checks.py` → `g4_extra_checks.json`; `g4_integrity.py` →
`g4_integrity.json`). They read the stored result JSONs and do small independent enumerations in pure Python;
they never import the repository's modules and never rerun an experiment. Total run time ≈ 12 s.

**A. Round 12 — events in which the truth had model probability exactly zero** (from the 6,912 recorded
pre-outcome predictions). By construction a fall-back happens exactly when the observed fact had probability 0.

| Regime, method | pre-outcome predictions | truth had probability 0 | counted by the study as "certain and wrong" |
|---|---:|---:|---:|
| compatible, source_both | 288 | 0 | 0 |
| compatible, source_a only | 288 | 24 | 0 |
| compatible, source_b only | 288 | 24 | 19 |
| compatible, unrelated_sources | 288 | 48 | 14 |
| family changed, source_both | 288 | 24 | 15 |
| family changed, frozen_source_both | 288 | 48 | 39 |
| local exception, source_both | 288 | 24 | 13 |
| local exception, frozen_source_both | 288 | 31 | 17 |
| local exception, permutation_prior / cycle_catalog | 288 | 24 | 4 |

So the study's false-certainty counter sees only part of the events (those where another outcome had
probability 1): `source_a` in the compatible regime excludes the truth with certainty once in every life and
is recorded with 0. Under a log score each such event is infinite.

**B. Round 12 — independent re-derivation of the structural witness and of the composition counterexample**
(fractions, 720 permutations): class sizes 120 and 15; one-step marginals identical (row 0 = 0, 1/5, 1/5, 1/5,
1/5, 1/5); after the fact `0→1` 24 and 3 candidates remain; `P(1→0)` = 0 and 1; for the 15 involutions
`E[P²] = I` while `(E[P])²` has 1/5 on the diagonal and 4/25 elsewhere. Agrees with
`results/structural_witness.json` and `MATHEMATICAL_AUDIT_TR.md:72-79`.

**C. Round 12 — how fast the transferred structure is identified** (compatible regime, `source_both`): the class
of an action is certain after **2.96** of its own facts on average (2–4); both classes are certain after 7.17
target facts (4–10); total surviving candidates over both actions: 270 → 162 → 101.5 → 55.0 → 21.7 → 13.3 → 8.8 →
4.75 → 3.29 → 2.29 → 2 (t = 0…10). At t = 0 `source_a`, `source_b` and `source_both` have identical direct Brier
.800000 (the audit's symmetry prediction, `MATHEMATICAL_AUDIT_TR.md:83`), the table .833333, and the
type-equal `cycle_catalog` **.899449** — a prior that weights the 11 cycle types equally is *worse than uniform*
before any evidence (it puts 1/11 on the identity), a concrete instance of R4.

**D. Round 14 — independent two-world witness** (`g(h) = h₀⊕h₁`, `f₀ = x₀`, `f₁ = x₁⊕1`, own enumeration of all
16,384 structures on all 512 inputs): history-gated and surface-gated worlds agree on all 256 aligned inputs
and differ on 128 inputs, all in the reversed half; exactly **two** structures reproduce all 256 noise-free
aligned labels — the true one and its surface twin; 15,904 distinct functions; bounds 1/16 and 1/8; closed
forms `(1/2)²·(1/2) = .125` and `(15/16)²·(1/2) = .439453125` equal the measured reversed-cue errors after 48
aligned records.

**E. Round 14 — the learner after 48 aligned records is a two-hypothesis learner.** For all 48 balanced-regime
runs of the two joint learners, the recorded total mass of the true gate after 64 records equals
`r·19^k/(1 + r·19^k)` (`r = 1` or `1/15`, `k` = discriminating records in the suffix) with maximum absolute
difference **1.7e−9**. Discriminating records among the 16 suffix records: aligned 0; balanced 3.92 on average
(1–8); reversed 8.17 (6–13). The study proves the 19× law for the matched pair only; empirically the whole
family mass follows it because nothing else retains weight.

**F. Round 18 — anatomy of the persistence cost** (per-task records of `adaptive_persistent` and
`adaptive_reset`, tasks 5–15 of the 24 lives = 264 task instances): 114 solvable at order 1, 150 needing more.
Persistent − resetting = **+75,947** on average (72,644–79,268) on the former and **−20,480** exactly on the
latter (`16·Σ_{mask}(1 + |mask|) = 20,480`, the failed first-order scan). Totals 8,657,928 − 3,072,000 =
5,585,928 = 24 × 232,747. Break-even share of order-2 tasks **78.8 %**, observed **56.8 %**. Of the extra cost on
the returning additive tasks 84.4 % is real execution (`evaluation_nodes`), 8.6 % the larger imagined-model
scan, 5.3 % construction, 1.8 % coefficient arithmetic. Mean life work: first_order 598,783; adaptive_reset
1,217,805; adaptive_persistent 1,450,552; second_order 1,733,920; exact_subset 4,820,016; a per-task oracle
between "order 1 first" and "order 2 directly" **1,069,325** (−12.2 % vs resetting).

**G. Integrity.** 53/53 stored artifact hashes equal the current files; `git status` clean before and after.

---

## 7. Established / Refuted / Open — across the group

Grades as in §0. "Established" means: shown by this group's own evidence *within the stated scope* — in every
case a supplied finite language, 24 paired lives where empirical, descriptive statistics, no external
replication. Nothing below is claimed by the studies as a new principle.

### Established

| # | Statement | Grade | Scope / what is given | Where |
|---|---|---|---|---|
| E1 | The same events admit groupings that cut across each other; an order that only appends to categories fixed once, read one way, cannot be assumed sufficient. | [EXACT] thought witness | two accessible binary features; not an impossibility result for hierarchies | concepts `OBSERVATION…:60-68` |
| E2 | Equal link diagrams do not have equal valid compositions (`before` composes, `knows` does not). | [EXACT] logic | — | concepts `CROSS…:35-40` |
| E3 | Structure acquired from two separate experiences, matched to a renamed target without a given mapping, yields untaught two-step results with less target evidence: pre-outcome Brier .398 vs .592 (bijection prior) / .532 (all cycle types) / .833 (table); all 12 compositions exact after 8.58 vs 10.54 / 10.54 / 12 facts; better than every control in 24/24 lives. | [EMP n=24] | compatible target; relation language, common interface and composition rule given; 24 source observations spent first | relations `REPORT_TR.md:37-50` |
| E4 | Correct past structure carried to an unfit target is worse than no structure: 24/24 lives worse than a structure-free bijection prior after a family change; certain-and-wrong predictions 0 → 15 and 13 per 288. | [EMP n=24] | closed candidate language, hard elimination | relations `REPORT_TR.md:54-60` |
| E5 | A learner that refuses contradiction never recovers (final composition Brier .667 / .431; worse at the end in 24/24). | [EMP n=24] | — | relations `REPORT_TR.md:62` |
| E6 | Mean beliefs of two different, independently held relations compose exactly; mean beliefs of the same relation used twice do not (`E[P²] = I` vs `K²`). | [EXACT] | independent posteriors | relations `MATHEMATICAL_AUDIT_TR.md:45-81` |
| E7 | Identical current predictions, opposite correct response to the same new fact (0 vs 1): keeping current mean predictions does not keep the learner. | [EXACT] post-result witness | for a state reduced to mean predictions | `structural_witness.json`; relations `REPORT_TR.md:68-79` |
| E8 | Learner state survives a process restart with the same future learning (63 fresh-process continuations equal to the uninterrupted object); predictions are equivariant under renaming of sources, objects and actions (312 checkpoints). | [EXACT] checks | the three experiments' learners | verify scripts |
| E9 | Experience in which two explanations always coincide cannot identify which is right, at any sample size; any single predictor has mean MSE ≥ 1/16 (independent cue) or ≥ 1/8 (reversed cue) over the two worlds. Measured: 48 or 64 aligned records leave .125 / .439453 unchanged. | [EXACT] + [EMP n=24] | two equally weighted worlds; parity gate/expert family | context `MATHEMATICAL_AUDIT_TR.md:39-77`; `REPORT_TR.md:39` |
| E10 | With identical structures, data and update rule, the initial weighting alone fixes behaviour in the ambiguous period: odds 1 vs 1/15 → true-gate mass .5 vs .0625 → reversed-cue MSE .125 vs .439453. | [EXACT] + [EMP n=24] | priors over 16,384 representations of 15,904 functions | context `REPORT_TR.md:41-47` |
| E11 | Only observations on which the rival structures predict differently move their odds, by exactly `(1−ε)/ε = 19` each; such evidence generalises to never-seen inputs (.0690 → .0000298; .2425 → .00236). | [EXACT] + [EMP n=24] | matched pair of structures; ε = .05 assumed | context `MATHEMATICAL_AUDIT_TR.md:92-108`; `REPORT_TR.md:49-58` |
| E12 | A learner that chooses its next experience by its own predictive disagreement is better than random choice after four queries in 24/24 lives (full cube and unseen), at 512× selection compute; not better at every step. | [EMP n=24] | noiseless label oracle for any of 512 inputs | context `REPORT_TR.md:69-102` |
| E13 | The structure (gate, experts) is uniquely recoverable from the whole function inside the family (rank-1 interaction matrix over F₂). | [EXACT] | experts different and not complementary | context `MATHEMATICAL_AUDIT_TR.md:110-122` |
| E14 | Finite-difference (Möbius) expansion of the effects of edits is exact in F17; all single-edit effects can be zero while a pair of edits is required. First-order model and greedy descent 0/24 on the witness; second-order and exhaustive 24/24. | [EXACT] + [EMP n=24] | 8 disjoint sites, known arithmetic | interactions `MATHEMATICS…:7-57`; `REPORT_TR.md:39` |
| E15 | The model order a target needs depends on the target's own edit set, not on the family: order 1 solves exactly the 18/96 pair-phase targets with no complete pair; order 2 solves exactly the 84/96 triple-phase targets with no complete triple. | [EMP n=24] + post-run exact diagnosis | — | interactions `REPORT_TR.md:41-43`; `coverage_diagnostics.json` |
| E16 | Carrying a richer search preference forward unconditionally: identical accuracy and accepted programs, +232,747 work per life (+19.1 %), more expensive in 24/24 lives (−81,920 in the triple phase, +303,728 on return to additive tasks). | [EMP n=24] | designed task order; virtual work model; one irreversible bit | interactions `REPORT_TR.md:47-61` |
| E17 | Imagine-then-verify accepted 1,744 programs with 0 false acceptances on the full domain; one task was solved by a program equal in behaviour but different in edit set from the planted one. | [EMP] | 16 examples in F17², 4-of-8 masks | interactions `REPORT_TR.md:35` |
| E18 | Effect vectors beat scalar feedback: on 192 additive targets the first-order effect model solves 192, greedy descent on the mismatch count solves 5. | [EMP n=24]; the reading is [ANALYST] | finite-field outputs give no partial credit | interactions `REPORT_TR.md:21-29` |
| E19 | Whenever the truth is outside a hard-elimination learner's language, the fact that reveals it had model probability exactly 0 (24/24 lives per affected action); the study's false-certainty counter records only 13–15 of those 24. | [ANALYST] from recorded data | round 12 | §6 check A |
| E20 | After the ambiguity is set up, round 14's 16,384-counter learner behaves as a two-hypothesis learner: one log-odds number reproduces the recorded masses within 1.7e−9. | [ANALYST] from recorded data | balanced continuation, 48 runs | §6 check E |

### Refuted or not supported

| # | Idea | Verdict and evidence | Where |
|---|---|---|---|
| X1 | Good prediction on the experienced stream means the right explanation was found. | **Refuted**: ≈ 1e−13 on aligned inputs, .125 / .439453 on reversed-cue inputs; "Eğitimde doğru görünmek, doğru bağlam açıklamasını bulmak değildir". | context `REPORT_TR.md:21-39` |
| X2 | More data of the same kind resolves the ambiguity. | **Refuted**: 64 aligned records give the same two values. | context `REPORT_TR.md:39` |
| X3 | Treating two explanation families equally treats the explanations equally. | **Refuted**: 1/15 odds for the specific true explanation. | context `REPORT_TR.md:45` |
| X4 | Classification accuracy is a sufficient measure of what was learned. | **Refuted**: both 100 %, probability error differs ≈ 79×. | context `REPORT_TR.md:56` |
| X5 | Uncertainty-driven choice of experience is better at every step / is cheaper overall. | **Not supported**: worse in 3 and 6 of 24 lives at the first query; 512× selection compute. | context `REPORT_TR.md:84,102` |
| X6 | More correct prior knowledge can only help. | **Refuted**: 24/24 worse after a family change; equal-count correct sources with the wrong structure .920 vs table .833. | relations `REPORT_TR.md:44,60` |
| X7 | Preserving current (mean) predictions preserves future learning. | **Refuted** by exact witness. | relations `REPORT_TR.md:77` |
| X8 | Mean transition beliefs can be multiplied for repeated use of one relation / long chains. | **Refuted** by exact counterexample; the experiment excludes such queries. | relations `MATHEMATICAL_AUDIT_TR.md:68-81` |
| X9 | Accepting contradiction (fall back for the whole relation) is always better than refusing it. | **Refuted for immediate prediction** under one local exception: .692 vs .623 (7 better / 2 equal / 15 worse); the cause is only suggested; "Mevcut geri dönüşün en iyi düzeltme yöntemi olduğu gösterilmedi". | relations `REPORT_TR.md:64` |
| X10 | A change that has no effect alone is unnecessary. | **Refuted** by constructive witness and 0/24 vs 24/24. | interactions `MATHEMATICS…:42-53` |
| X11 | A fixed second-order effect model suffices everywhere. / If triple structure is present every target needs third order. | **Both rejected**: 84/96; exactly the 12 targets containing a complete triple fail. | interactions `REPORT_TR.md:43` |
| X12 | What helped once should be kept for good; returning to the same kind of task always pays. | **Not supported**: +19.1 % work at equal accuracy; in the pair phase +10,939, cheaper in only 9/24 lives. | interactions `REPORT_TR.md:49-59` |
| X13 | The learner refuted its own wrong prediction and therefore expanded its model. | **Did not happen**: 0 recorded mispredictions; expansion followed absence of a verified candidate. | interactions `REPORT_TR.md:67` |
| X14 | Linking two bodies of knowledge through a common node yields the inference. | **Refuted** logically (E2); also: adding links cannot regenerate a missing distinction. | concepts `CROSS…:35-40,75` |
| X15 | More (or fewer) branches means better (or worse) understanding. | **Rejected as a measure.** | concepts `OBSERVATION…:43` |
| X16 | These results amount to a new principle / concept formation / a self-acquired web of meaning. | **Explicitly disclaimed** by every report; weak baselines (`bulk_all` 0/384, greedy) are stated not to be evidence. | relations `REPORT_TR.md:9`, `LITERATURE_TR.md:41`; context `REPORT_TR.md:9,86`; interactions `REPORT_TR.md:5,65,71` |
| X17 | Expressing the same functional change in another data structure is a new computational principle. | **Rejected in advance** by the programme's own standard. | context `REPORT_TR.md:118`; `THEORY_NOVELTY_BOUNDARY_TR.md:27` |

### Open

| # | Question | Named in | Status in this group |
|---|---|---|---|
| O1 | Forming the useful distinction itself from failure and transfer experience (no categories, gates or patterns supplied). | H1, H6, H14; interactions `WHOLE_PROBLEM…:28-30` | no experiment; all three select within supplied languages |
| O2 | Learning the interface through which separately acquired structures compose (object count, abstraction level, arity may differ). | H16 | interface supplied in round 12 |
| O3 | Telling "source wrong" from "role mapping wrong" from "family changed" from "local exception"; learning the scope of a change. | H15, H18-A | need shown (X9, E16); no method |
| O4 | Persistent state that separates event record, interpretation and licensing assumptions. | H2, H13 | untested |
| O5 | Concept vs evaluation; internal state as a source of new, persistent competence. | H4, H17 | untested |
| O6 | Lasting effect without the raw record; meaning carried by the order of events. | H3, H7 | H3 only instantiated in passing (round 12 keeps no raw source episodes) |
| O7 | Whether uncertainty computed in a wrong or incomplete language can guide experience. | context `ACTIVE_EXPERIENCE_PROTOCOL.md:22` | untested |
| O8 | Whole-life economics of development (life budget, memory measure, future tasks set aside). | H18-C | one negative data point (E16) |
| O9 | Transfer to a third task; a formed bridge that later bridges build on; ablation with an equal-size unrelated link. | concepts `CROSS…:55,62,65`; `THEORY_NOVELTY_BOUNDARY_TR.md:45` | not attempted |
| O10 | Change of the update procedure `U` and of the representation `R` by experience. | context `REPORT_TR.md:120-122` | only beliefs inside fixed `R/P/U/Q` change; one bit of `Q` in round 18 |
| O11 | Extraction of the history summary from a stream; real access costs for experience. | context `REPORT_TR.md:13,73` | `h` and the query oracle supplied |
| O12 | A symbolic competitor that reads joint effects off the visible program. | interactions `REPORT_TR.md:66` | absent |
| O13 | Statistical strength: intervals, more than 24 lives, more than one configuration, independent replication. | every report's closing lines | descriptive only |
| O14 | [ANALYST] A compact belief that keeps the joint constraints the declared future operations need (between marginals and full enumeration); a soft-likelihood re-run of round 12's local-exception regime. | §5.4 items 10–11 | not in the studies |
