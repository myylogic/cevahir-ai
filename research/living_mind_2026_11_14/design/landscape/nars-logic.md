# Landscape: non-axiomatic reasoning, evidential truth values, belief revision, and their neural forms — mapped against P1–P6

Date of survey: 2026-11-14. Scope: NARS / OpenNARS / ONA, Probabilistic Logic Networks (OpenCog, Hyperon), evidential truth values (frequency/confidence, subjective logic), belief revision (AGM, ranking theory), truth maintenance, and neural or differentiable implementations of any of these (neural theorem provers, logic tensor networks, differentiable ILP, logical neural networks, neuro-symbolic concept learners, evidential deep learning).

## 0. How to read this file

**Status tags on every statement**

| Tag | Meaning |
|---|---|
| **[V]** VERIFIED | I read the primary source in this session: the paper's full text from arxiv.org, or the project's own source code / README from raw.githubusercontent.com or a clone. |
| **[A]** ABSTRACT | Only an abstract, search snippet or catalogue entry was seen. |
| **[M]** MEMORY | From my own knowledge; not checked in this session. |
| **[D]** DERIVED | My own arithmetic or inference from statements tagged [V]. Not a claim found in a source. |

**The principles** (short names used below): **P1** learning = running, every experience changes the weights at once; **P2** weights are graded beliefs (weak first hearing, contradiction refines instead of deleting, a category rule is induced and feeds support back to the instance); **P3** concepts are formed when needed, no fixed embedding table; **P4** imagination: the system observes its own constructions, measures how real they are and advances by that measure; **P5** what is acquired is the way of constructing, which improves and transfers; **P6** an engine of tensors and equations, with language in and out eventually.

**Ratings**: N = satisfied natively, P = partially, – = not at all. P† = only prediction-versus-observation checking (a narrow part of P4).

**What was done**: 37 web searches (34 standard, 3 extended) and 13 arXiv API queries; 47 papers downloaded from arxiv.org, of which the 42 listed in section 8 were read in part or in full; source files of OpenNARS, OpenNARS-for-Applications (ONA), and READMEs of ANSNA, NARS-GPT, PLN (MeTTa), MeTTa-NARS, NACE read directly. Local copies are under `/tmp/claude-0/-workspace-cevahir-ai/79909244-c51d-5307-ad00-1a1e5ef77203/scratchpad/newcore/landscape/_src_nars/` (text extracts) and helper scripts under `/tmp/claude-0/-workspace-cevahir-ai/79909244-c51d-5307-ad00-1a1e5ef77203/scratchpad/newcore/landscape/_tools_nars/`.

**What could not be done**
- Hosts that refused connections in this session: cis.temple.edu (most NARS papers, including the ONA paper and the NAL specification), frontiersin.org and pmc.ncbi.nlm.nih.gov ("Self in NARS"), proceedings.mlr.press, openreview.net, jair.org, and the GitHub repository API. Other publisher sites (AAAI, Springer, Cambridge) were not tried. Works available only on such sites are tagged [A] or [M].
- ONA was **not executed**. I cloned it and read it, but building and running third-party code was not permitted in this session. All numbers for the "red" example are computed by hand from the truth functions read in ONA's source (script: `_tools_nars/red_example_nal.py`). Which derivation ONA's attention picks in which cycle was therefore not observed.

---

## 1. Verdict in ten lines

1. The closest existing family is **NARS** (Non-Axiomatic Logic). It satisfies P1 natively and most of P2 natively, P3 partially, P4 and P5 only on paper, P6 not at all. [D]
2. **No work in this field satisfies P3, P4, P5 or P6 natively.** [D]
3. Every neural or differentiable version gives up P1: all are trained offline by gradient descent on a dataset (Differentiable PLN, NTP, LTN, ∂ILP, LNN, NS-CL/VCML, evidential deep learning). [V for each, see section 3]
4. Most neural versions also give up evidence accounting. Differentiable PLN drops the confidence number [V]; evidential deep learning keeps the two-number format but its "evidence" is predicted by a trained network, and is shown not to be a faithful measure [V].
5. The only attempts to put NARS-style evidence on vectors or on a neuron-like network are **ANSNA** (2019, discontinued in favour of a return to symbols) and **Xu's mini-column model** (2023, small, no language). [V]
6. NARS reproduces the first half of the owner's worked example exactly (0.23 → 0.57 is NAL revision with one more unit of evidence) and **cannot** produce the second half (rule at 0.77 from one instance; instance lifted to 0.88 by its own rule). [D]
7. The reason is a deliberate design rule: a conclusion may not be merged into a belief that shares evidence with it ("stamp" overlap). NARS forbids exactly the self-supporting loop the owner's example describes. [V]
8. In NARS a question never arises by itself from a new generalisation. Questions are typed in, or spawned from goals when a switch that is off by default is turned on, or produced by a `^wonder` operator that the system does not trigger for this purpose. [V]
9. Language is the weakest side of the whole field: NARS reads English through a regex parser or through GPT-4 [V]; I found nothing in the field that acquires or generates language with its own mechanism at scale. [D]
10. The belief-revision literature (AGM, TMS) and the evidential-logic literature (NARS, PLN) barely know each other: a 2026 survey of belief change with an implementation roadmap does not mention NARS once. [V]

---

## 2. The 25 most relevant works

| # | Title | First author | Year | Venue / arXiv id | Status |
|---|---|---|---|---|---|
| 1 | Non-Axiomatic Logic: A Model of Intelligent Reasoning (implemented as OpenNARS) | Wang | 2013 | World Scientific (book); code github.com/opennars/opennars | book MEMORY; truth functions, revision conditions, stamps, mental operators VERIFIED in source |
| 2 | OpenNARS for Applications: Architecture and Control (ONA) | Hammer | 2020 | AGI-20, LNCS 12177 | paper ABSTRACT (host blocked); README and source at commit dc4efd0 (2025-05-16) VERIFIED |
| 3 | Self in NARS, an AGI System | Wang | 2018 | Frontiers in Robotics and AI 5:20 | ABSTRACT; the operators it describes VERIFIED in OpenNARS source |
| 4 | Adaptive Neuro-Symbolic Network Agent (ANSNA) | Hammer | 2019 | AGI-19, LNCS 11654, pp. 80–90 | paper ABSTRACT; project README VERIFIED |
| 5 | A Brain-Inspired Sequence Learning Model based on a Logic | Xu | 2023 | arXiv:2308.12486 | VERIFIED |
| 6 | Non-Axiomatic Term Logic: A Computational Theory of Cognitive Symbolic Reasoning (NATL) | Funakoshi | 2022 | arXiv:2210.06316; Trans. JSAI 37(6) | VERIFIED |
| 7 | NARS-GPT (GPT as language channel for ONA) | Hammer | 2023 | software, github.com/opennars/NARS-GPT | README VERIFIED (author and year from MEMORY) |
| 8 | Probabilistic Logic Networks (PLN) | Goertzel | 2008 | Springer (book); MeTTa code github.com/trueagi-io/PLN | book MEMORY; formulas VERIFIED via arXiv:2412.19524; MeTTa README VERIFIED |
| 9 | Differentiable Probabilistic Logic Networks | Potapov | 2019 | arXiv:1907.04592 | VERIFIED (full text) |
| 10 | OpenCog Hyperon: A Framework for AGI at the Human Level and Beyond | Goertzel | 2023 | arXiv:2310.18318 | VERIFIED (abstract, MeTTa and cognitive-process sections) |
| 11 | Bounded Recursive Self-Improvement (AERA) | Nivel | 2013 | arXiv:1312.6764 (RUTR-SCS13006) | VERIFIED (abstract, §4.1, reliability equations, §6) |
| 12 | On the Logic of Theory Change: Partial Meet Contraction and Revision Functions (AGM) | Alchourrón | 1985 | J. Symbolic Logic 50(2) | MEMORY |
| 13 | Ordinal Conditional Functions: A Dynamic Theory of Epistemic States (ranking theory) | Spohn | 1988 | in Harper & Skyrms (eds.), Causation in Decision, Belief Change, and Statistics II | MEMORY |
| 14 | Belief Revision in Probability Theory | Wang | 1993 | UAI-93; arXiv:1303.1517 | VERIFIED (abstract, §1–2) |
| 15 | Machine Learning as Iterated Belief Change à la Darwiche and Pearl | Aravanis | 2025 | arXiv:2506.13157 | VERIFIED (abstract) |
| 16 | What Do Hebbian Learners Learn? Reduction Axioms for Iterated Hebbian Learning | Kisby | 2024 | AAAI-24 | ABSTRACT |
| 17 | Fundamental Problems With Model Editing: How Should Rational Belief Revision Work in LLMs? | Hase | 2024 | TMLR; arXiv:2406.19354 | VERIFIED (abstract, introduction, Table 1) |
| 18 | A Truth Maintenance System | Doyle | 1979 | Artificial Intelligence 12(3) | MEMORY (abstract seen as snippet) |
| 19 | BeliefBank: Adding Memory to a Pre-Trained Language Model for a Systematic Notion of Belief | Kassner | 2021 | EMNLP 2021; arXiv:2109.14723 | VERIFIED (abstract) |
| 20 | End-to-End Differentiable Proving (Neural Theorem Prover) | Rocktäschel | 2017 | NeurIPS 2017; arXiv:1705.11040 | VERIFIED |
| 21 | Logic Tensor Networks | Badreddine | 2022 | Artificial Intelligence 303; arXiv:2012.13635 | VERIFIED (abstract, §2) |
| 22 | Learning Explanatory Rules from Noisy Data (∂ILP) | Evans | 2018 | JAIR 61; arXiv:1711.04574 | VERIFIED |
| 23 | Logical Neural Networks | Riegel | 2020 | arXiv:2006.13155 | VERIFIED (abstract, conclusion, appendix G.3) |
| 24 | Visual Concept-Metaconcept Learning (VCML, built on NS-CL) | Han | 2019 | NeurIPS 2019; arXiv:2002.01464 | VERIFIED (abstract, model figures) |
| 25 | Evidential Deep Learning to Quantify Classification Uncertainty | Sensoy | 2018 | NeurIPS 2018; arXiv:1806.01768 | VERIFIED (abstract, eq. 1) |

---

## 3. Mechanism and P1–P6 for each work

### A. Symbolic evidential reasoners

**1. Non-Axiomatic Logic / OpenNARS (Wang 2013)**
- Mechanism: Every statement (for example `S --> P`, "S is a kind of P") carries a truth value ⟨f, c⟩ — f is the share of positive evidence, c = w/(w+k) for total evidence w and horizon k = 1 — and a stamp listing the input evidence it rests on; about a dozen syllogistic rules (deduction, induction, abduction, comparison, analogy, revision) each have a fixed truth function. [V: OpenNARS `TruthFunctions.java`, `UtilityFunctions.java`; ONA `Truth.c`] Each term has a concept that holds its beliefs and tasks; priority "bags" choose what is processed next and drop the lowest-priority items when full, so the system runs in real time with bounded memory. [V for bag sizes and forgetting rates in `Parameters.java`; M for the description]
- P1 **N** — no training phase; each input and each derivation immediately revises beliefs and priorities. [M; V for the revision path in `LocalRules.java`]
- P2 **N, with two exceptions** — the default confidence of a first hearing is 0.9, not weak (a parameter) [V]; a conclusion deduced from a rule cannot be merged into a belief that is part of that rule's own evidence (`revisible` requires `!Stamp.baseOverlap`). [V]
- P3 **P** — concepts are created at run time for every term, including compound terms the system builds itself, and a term's meaning is its experienced relations; but atomic terms are identifiers supplied by the input channel. [M]
- P4 **P** — mental operators `^believe`, `^want`, `^wonder`, `^evaluate`, `^doubt`, `^hesitate`, `^consider`, `^remind`, `^feel`, `^anticipate` and an `InternalExperience` plugin exist in the source [V]; the plugin starts disabled (`enabled=false`), full reflection is off by default, and when on it fires a random mental operator with probability 2.5·10⁻⁵ per reasoning event. [V]
- P5 **P** — the logic and the control heuristics are fixed code; what grows is knowledge, including procedural knowledge. [M]
- P6 **–** — symbolic term rewriting in Java; language through an external parser. [M]

**2. OpenNARS for Applications, ONA (Hammer & Lofthouse 2020)**
- Mechanism: An event-driven C implementation: selected belief events (attention buffer of 40) are matched against related concepts (at most 4096), and the rule table is applied only if the two stamps do not overlap; the conclusion's stamp is the zip of both, cut at 10 entries. [V: `Cycle.c`, `Stamp.c`, `Config.h`] Each concept keeps one eternal belief, the latest event and a predicted event, merged by revision when the evidence is disjoint and otherwise by "choice" (the more confident one stays); temporal and procedural implications `(a &/ ^op) =/> b` are learned from event sequences and drive decisions with subgoaling. [V: `Inference.c`, `Memory.c`, README]
- P1 **N** [V README; M]. P2 **N, same two exceptions** [V]. P3 **P** [M]. P4 **P†** — a predicted event that does not arrive counts as negative evidence with confidence 0.01. [V `Config.h`] P5 **–** — control is fixed; exploration is random "motor babbling" with chance 0.2. [V] P6 **–** — C structures; English via a regex/part-of-speech parser or GPT. [V]
- Limits of the default build: 255 atomic terms (`Atom` is one byte), 4096 concepts, 120 implications per table. [V `Config.h`]

**3. Self in NARS (Wang, Li, Hammer 2018)**
- Mechanism: "Self" is an ordinary concept whose content is acquired from experience: selected internal events are entered into the system's own experience, and mental operations "supplement and adjust the automatic inference control routine". [A] Self-knowledge is therefore "mostly acquired and constructive, while being incomplete and subjective". [A]
- P1 **N**, P2 **N** (inherits NARS) [M]. P3 **P** [M]. P4 **P** — the NARS account of watching one's own activity; it has no measure of how real a construction is. [A; D] P5 **P** — the one NARS work aimed at improving the way of reasoning itself; realised only as in #1. [A; V] P6 **–**.

**4. ANSNA (Hammer 2019)**
- Mechanism: NARS with vectors instead of terms: each event holds a sparse distributed representation (SDR), is stored in the concept with the closest SDR, and new events are composed by union, intersection and tuple operations; "inheritance and similarity naturally arise from symmetric and asymmetric SDR overlap evaluations". [V README] Predictive links between concepts carry NAL evidence values and are learned online ("lifelong online learning", "NAL-based evidence measurement") under a spiking-style priority control. [V README]
- P1 **N** [V]. P2 **P** — evidence values on predictive links only; "restricted to NAL7/8 plus a SDR-based realization of NAL6". [V] P3 **P** — the nearest thing in this field to concepts without a symbol table: a concept is whatever container has the closest SDR; the input encoders are hand-made. [V; D] P4 **P†** — it derives predicted events [V]; that they are checked against what arrives is assumed from the NARS design. [M] P5 **–**. P6 **P** — bit-vector operations plus truth equations; no language. [V]
- Fate: discontinued. ONA "is a completely new platform" that took ANSNA's event handling and went back to terms. [V ONA README]

**5. A Brain-Inspired Sequence Learning Model based on a Logic (Xu 2023)**
- Mechanism: One mini-column per concept, whose neurons are the concept in different contexts; links between neurons are temporal implication or equivalence statements with NAL truth values ⟨f; c⟩. [V] Learning runs online on an endless event stream in three steps — hypothesise new links, revise their truth values by temporal induction, recycle weak links under a fixed budget — and event types are created at run time. [V]
- P1 **N** [V]. P2 **P** — graded links that are revised; no induction of category rules. [V; D] P3 **P** — "the types of events are not predetermined … but dynamically constructed". [V] P4 **P†** [V]. P5 **–**. P6 **P** — a network with truth-value equations, one unit per concept rather than distributed; the author says it is "not powerful enough" and not yet sensible for natural language. [V]

**6. Non-Axiomatic Term Logic, NATL (Funakoshi 2022)**
- Mechanism: Keeps term-logic syntax but attaches to every term, statement and copula a semantic vector computed by neural networks, so that unification works by similarity in a latent space instead of symbol identity. [V] The reasoner is a function R(E, Q) → (T₁, T₂, T_c, t), to be trained by supervised or reinforcement learning, with one real confidence t in place of NAL's two numbers; the paper is theory only ("no quantitative evaluation is presented"). [V]
- P1 **–**, P2 **–** [V; D]. P3 **P** [V]. P4 **–**. P5 **P** — the reasoning skill itself is to be learned. [V] P6 **P** — a design; pretrained language models are the separate "language system". [V]

**7. NARS-GPT (Hammer et al. 2023)**
- Mechanism: GPT-4 turns each English sentence into statements stored in ONA's memory, merging new terms with existing ones by embedding similarity; ONA does revision and inference with NAL truth values. [V README] To answer, up to k memory items chosen by recency and embedding similarity are placed in the prompt, and the certainty values it reports are NAL confidences. [V README]
- P1 **P** — the memory learns continuously, the language model is frozen. [V; D] P2 **N** (inside ONA) [V]. P3 **P** [V]. P4 **–**. P5 **–**. P6 **P** — language in and out, but through a frozen external Transformer. [V]

**8. Probabilistic Logic Networks, PLN (Goertzel et al. 2008)**
- Mechanism: Term-logic and higher-order rules over truth values made of a probability-like strength and a confidence (or a count, or an interval "indefinite probability"); deduction uses the probabilities of the terms themselves, and induction and abduction are Bayes-rule inversions followed by deduction. [V via arXiv:2412.19524; M for indefinite probabilities] A revision rule merges evidence about the same statement; the current MeTTa implementation attaches EvidenceIDs "to avoid double-counting or to make a choice" and bounds inference with a confidence-ordered priority queue. [V README of trueagi-io/PLN]
- P1 **P** — incremental in principle, run as a chainer over a knowledge base (`PLN.Derive`, `PLN.Query`). [V; D] P2 **N in the same sense as NARS**, but induction needs term probabilities (base rates). [V] P3 **P** — concept formation is left to separate processes (blending, pattern mining). [V Hyperon paper] P4 **–**. P5 **P** [M]. P6 **–**.
- Goertzel's own comparison: when term probabilities are highly uncertain, PLN and NARS give similar strength × confidence. [V]

**9. Differentiable Probabilistic Logic Networks (Potapov et al. 2019)**
- Mechanism: Truth values become PyTorch tensors and each rule formula a differentiable operation, so a chain of rule applications is a computation graph through which errors flow back to the truth values of premises and to parameters inside the formulas. [V] In the single experiment (modus ponens on fruit colours) gradient descent recovers P(B|A) = n₁/(n₀+n₁), the plain frequency count; "we don't consider the confidence component of truth values here". [V]
- P1 **–** (batch) [V]. P2 **P** [V]. P3 **–**. P4 **–**. P5 **P** — rule formulas are trainable: the one place in this field where the way of inferring is itself learned by the tensor engine. [V; D] P6 **P** [V].
- Reported failure: when formulas and truth values are learned together, "truth values can be learned as non-linearly transformed desirable values"; "a more principled solution that forces preserving the semantics of truth values might be desirable". [V]

**10. OpenCog Hyperon (Goertzel et al. 2023)**
- Mechanism: A redesign of OpenCog around the MeTTa language and an Atomspace metagraph on which many processes — PLN, ECAN attention allocation, pattern mining, concept blending, evolutionary learning, neural modules — work on one shared store. [V] It is a framework and roadmap: "the precise mix of cognitive processes is subject to ongoing experimentation". [V]
- P1–P5 **P** (stated aims) [V; D]. P6 **P** — neural networks and language models are modules beside the symbolic store. [V]
- A later proposal from the same group (ActPC-Geom, arXiv:2501.04832) for "real-time online learning" joining neural and symbolic networks calls itself "a novel, speculative approach". [V abstract]

**11. AERA — Bounded Recursive Self-Improvement (Nivel, Thórisson et al. 2013)**
- Mechanism: Knowledge is a growing set of executable causal models learned on-line from observation, starting from a small seed; each model has reliability e⁺/(e+1), which the report writes as success rate × experience e/(e+1), and the likelihood of a prediction or goal is the product of the reliabilities along its chain. [V] A value-driven scheduler runs many forward (prediction) and backward (goal) chains in parallel under resource bounds, and the prototype learned a multimodal interview dialogue by watching two humans. [V]
- P1 **N** ("on the job") [V]. P2 **P** — graded reliability from evidence counts; poorly performing models are discarded rather than refined. [V; A] P3 **P** [V]. P4 **P** — every model is tested continuously by its predictions, and the system models itself. [V] P5 **P** — self-programming within designer bounds. [V] P6 **–**.
- Its knowledge representation "has its roots in a non-axiomatic term logic" [V]; its reliability is NARS's f × c with k = 1. [D]

### B. Belief revision and truth maintenance, and their neural readings

**12. AGM (Alchourrón, Gärdenfors, Makinson 1985)**
- Mechanism: A belief state is a logically closed set of sentences, and revision by a new sentence must satisfy postulates of success, consistency and minimal change, realised by partial-meet contraction, epistemic entrenchment or systems of spheres. [M] Darwiche and Pearl (1997) add postulates for repeated revision, because AGM does not say how the ordering itself changes. [M; A]
- P1 **–** (a norm, not a mechanism). P2 **P** — minimal change, but beliefs are all-or-nothing and a contradicted belief is removed. [M] P3–P6 **–**.

**13. Ranking theory (Spohn 1988)**
- Mechanism: Every possible world gets a rank of disbelief (0 = not disbelieved), a proposition is believed with firmness equal to the rank of its negation, and evidence arrives with a firmness of its own that shifts ranks by conditionalisation. [M] The result is again a ranking function, so revision can be repeated indefinitely and weak evidence moves ranks without flipping beliefs. [M]
- P2 **P** — graded, weak first evidence, shift instead of deletion; it does not induce rules, and the firmness numbers must be supplied from outside. [M] All others **–**.

**14. Belief Revision in Probability Theory (Wang 1993)**
- Mechanism: Separates updating (conditioning within one probability space) from revision (changing an earlier judgment on the same statement when more evidence arrives) and argues that neither Bayes' theorem nor Jeffrey's rule is a general revision rule. [V] The reason given: "a probability distribution function alone does not contain the information needed by the operation" — the amount of evidence behind a value must be stored as well. [V; last clause D]
- P2 **P** — it is the argument for beliefs with two numbers. All others **–**. [D]

**15. Machine Learning as Iterated Belief Change (Aravanis 2025)**
- Mechanism: The input–output behaviour of a binary feed-forward network is written as a propositional belief set, and training is read as a sequence of belief-change steps. [V] Dalal's operator gives a gradual evolution of belief states [V]; training dynamics are modelled by lexicographic revision and moderate contraction, which fit the Darwiche–Pearl framework. [A]
- P1 **P**, P2 **P**, P6 **P** (binary networks only) [D]. Others **–**.

**16. What Do Hebbian Learners Learn? (Kisby, Blanco, Moss 2024)**
- Mechanism: A dynamic logic whose models are neural networks: "typically φ" is forward propagation, and a dynamic operator stands for repeated Hebbian update until the net reaches a fixed point. [A] Reduction axioms rewrite any statement about the net after learning into one about the net before, which gives a complete logic of that learning policy and reads it as a plausibility upgrade. [A]
- P1 **P**, P2 **P**, P6 **P** [D]. Others **–**.

**17. Fundamental Problems With Model Editing (Hase et al. 2024)**
- Mechanism: Treats model editing as belief revision and lists twelve open problems, among them background beliefs, many possible worlds, complete corrigibility, missing context, coherence at all cost, entailments that are hard to annotate, vague claims, and whether language models have editable beliefs and credences at all. [V] In a semi-synthetic Wikidata setting where an exact Bayesian agent supplies the target posteriors, edited Transformers "generalize poorly with respect to other relevant beliefs" and their probabilities "consistently diverge from Bayesian posterior probabilities". [V]
- Satisfies none; it is the measurement that P1 and P2 do not hold when one edits Transformer weights. [D] (P6 **P**: the object of study is a tensor engine with language.)

**18. Truth maintenance (Doyle 1979; de Kleer 1986)**
- Mechanism: A subsystem records for every belief the reasons that support it and labels it in or out; on a contradiction, dependency-directed backtracking finds the responsible assumptions and withdraws one. [M; A] De Kleer's assumption-based version instead labels each datum with the sets of assumptions under which it holds, keeping many contexts at once. [M]
- P1 **P** (incremental) [M]. P2 **–** (binary beliefs, withdrawal) [M]. P4 **P** — it keeps and inspects the reasons for its own beliefs. [D] Others **–**.

**19. BeliefBank (Kassner et al. 2021) and REFLEX (Kassner et al. 2023)**
- Mechanism: The yes/no answers of a frozen language model are stored in a persistent symbolic memory; a weighted MaxSAT solver flips beliefs that clash with others under given constraints, and stored beliefs are fed back as context for later queries. [V] REFLEX builds, per question, a belief graph by backward-chaining queries to the model and minimises contradictions with a constraint reasoner, improving consistency by 8–11 points absolute. [V]
- P1 **P** — the memory changes while running, the weights do not. [V] P2 **P** — weighted beliefs that are flipped, not graded. [V] P4 **P** — self-querying and measuring its own inconsistency. [V; D] P6 **P** [V]. P3, P5 **–**.

### C. Neural and differentiable logic

**20. Neural Theorem Prover (Rocktäschel & Riedel 2017)**
- Mechanism: Prolog-style backward chaining is unrolled into a network where symbol unification is an RBF kernel on symbol embeddings, a proof scores the minimum similarity along it, and a query takes the maximum over proofs. [V] Trained by gradient descent on a knowledge base it learns embeddings and induces rules, but it enumerates all proof paths and scaling is "an open problem" (GNTP later keeps only the most promising paths; CTP learns which rules to select). [V]
- P1 **–**. P2 **P**. P3 **P** — a learned vector per symbol of a fixed vocabulary. P4 **–**. P5 **–** (CTP: P, a learned selection strategy). P6 **P**. [V; D]

**21. Logic Tensor Networks (Badreddine et al. 2022)**
- Mechanism: "Real Logic": the constants, functions and predicates of a first-order language are grounded as tensors and neural networks, connectives are fuzzy operators and quantifiers are aggregators, so every formula has a differentiable truth degree. [V] Learning maximises the satisfiability of a knowledge base by gradient descent on the grounding parameters. [V]
- P1 **–**. P2 **P**. P3 **P**. P4 **–**. P5 **–**. P6 **P**. [V; D]
- "Continual Reasoning" (Kyriakopoulos & Garcez 2023, arXiv:2305.02171) adds a curriculum with recall to obtain exception handling on the penguin task. [V abstract]

**22. ∂ILP (Evans & Grefenstette 2018)**
- Mechanism: All clauses allowed by rule templates get a weight; differentiable forward chaining makes the valuation of every ground atom a function of those weights, which are trained by backpropagation on positive and negative examples. [V] It tolerates mislabelled data, but "requires significant memory resources", which limits predicates to arity two. [V]
- P1 **–**. P2 **P**. P3 **P** (invented auxiliary predicates [M]). P4 **–**. P5 **–**. P6 **P**. [D]

**23. Logical Neural Networks (Riegel et al. 2020)**
- Mechanism: One neuron per subformula of a weighted real-valued logic, each holding a lower and an upper truth bound that upward and downward passes tighten, so "unknown" and "contradictory" are distinct states and inference runs in every direction. [V] Training minimises a contradiction loss Σ max(0, L − U); the bound-tightening algorithm is "provably convergent in finite steps". [V]
- P1 **–**. P2 **P** — graded belief with explicit ignorance, tolerant of inconsistent knowledge. P3 **–**. P4 **–** (it does compute its own degree of contradiction). P5 **–**. P6 **P**. [V; D]

**24. Visual Concept-Metaconcept Learning, VCML (Han, Mao et al. 2019)**
- Mechanism: Objects and concepts are vectors; a concept is a half-space whose Gaussian volume is its probability, so P(b|a) = Vol(V_a ∩ V_b)/Vol(V_a), and questions are parsed into programs executed on these embeddings. [V] Metaconcepts such as `same_kind` are small networks over pairs of concept embeddings, trained jointly from images and text-only questions like "Do red and green describe the same property of objects?", and they help learn a new colour from a few biased examples. [V]
- P1 **–** (batch) [V]. P2 **P** [D]. P3 **P** — the best neural example of category-level knowledge ("red and green are both colours") shaping instance-level concepts; each word still has a slot in a table. [V; D] P4 **–**. P5 **P** — what is learned on colours carries over to shapes. [V] P6 **P** [V].

**25. Evidential Deep Learning (Sensoy, Kaplan, Kandemir 2018)**
- Mechanism: The network outputs non-negative evidence e_k per class, read as a subjective-logic opinion: belief b_k = e_k/S and uncertainty u = K/S with S = Σ(e_i + 1), which is a Dirichlet with α_k = e_k + 1. [V] The evidence function is learned by loss minimisation; Jürgens et al. (ICML 2024, arXiv:2402.09056) show that such methods "do not represent epistemic uncertainty in a faithful manner". [V]
- P2 **P** — the two-number belief format inside a tensor engine, but the evidence is predicted, not accumulated. [V; D] P6 **P**. Others **–**.
- For two classes, 1 − u = (e₁+e₂)/(e₁+e₂+2): the NARS confidence formula with k = 2. [D]

### Matrix

| # | Work | P1 | P2 | P3 | P4 | P5 | P6 |
|---|---|---|---|---|---|---|---|
| 1 | NAL / OpenNARS | N | N* | P | P | P | – |
| 2 | ONA | N | N* | P | P† | – | – |
| 3 | Self in NARS | N | N* | P | P | P | – |
| 4 | ANSNA | N | P | P | P† | – | P |
| 5 | Xu 2023 mini-column model | N | P | P | P† | – | P |
| 6 | NATL | – | – | P | – | P | P |
| 7 | NARS-GPT | P | N* | P | – | – | P |
| 8 | PLN | P | N* | P | – | P | – |
| 9 | Differentiable PLN | – | P | – | – | P | P |
| 10 | Hyperon | P | P | P | P | P | P |
| 11 | AERA | N | P | P | P | P | – |
| 12 | AGM | – | P | – | – | – | – |
| 13 | Ranking theory | – | P | – | – | – | – |
| 14 | Wang 1993 | – | P | – | – | – | – |
| 15 | Aravanis 2025 | P | P | – | – | – | P |
| 16 | Kisby 2024 | P | P | – | – | – | P |
| 17 | Hase 2024 | – | – | – | – | – | P |
| 18 | Truth maintenance | P | – | – | P | – | – |
| 19 | BeliefBank / REFLEX | P | P | – | P | – | P |
| 20 | Neural Theorem Prover | – | P | P | – | – | P |
| 21 | Logic Tensor Networks | – | P | P | – | – | P |
| 22 | ∂ILP | – | P | P | – | – | P |
| 23 | Logical Neural Networks | – | P | – | – | – | P |
| 24 | VCML | – | P | P | – | P | P |
| 25 | Evidential Deep Learning | – | P | – | – | – | P |

N* = native for graded evidence, revision, and induction followed by deduction; not for a weak first hearing by default, and not for a rule supporting its own source instance.

What the matrix shows: [D]
- P1 native appears only in rows 1–5 and 11. Tensor-like engines (P6 = P) appear mostly in rows 15–25. The two sets meet only in rows 4 and 5 — both small, both without language, one discontinued.
- No row has N in P3, P4, P5 or P6.
- No row combines N in P1 with P or better in every other column.

### Other works read (not in the top 25)

| Work | What it adds | Status |
|---|---|---|
| Goertzel 2024, "PLN and NARS Often Yield Similar strength × confidence…", arXiv:2412.19524 | Side-by-side NARS and PLN formulas for deduction, induction, abduction | VERIFIED |
| Johansson, Hammer, Lofthouse 2024/2025, arXiv:2405.03340, arXiv:2505.07079; Johansson 2024, arXiv:2405.19498 | ONA extended with "acquired relations" to show functional equivalence and same/opposite relational responding from minimal training | VERIFIED (abstracts, conclusion) |
| Beikmohammadi & Magnússon 2023, arXiv:2304.03291 | ONA versus Q-learning on seven Gym tasks: "comparable performance on average" | VERIFIED |
| Gabriel & Wang 2026, arXiv:2604.18873 | NARS-Reasoning-v0.1: natural-language problems paired with executable Narsese; a language model is trained to emit the symbolic structure | VERIFIED (abstract, §6–7) |
| Li, Wang et al. 2025, arXiv:2511.08392 (PCRLLM) | Language-model reasoning constrained to single NAL steps so each step can be checked | VERIFIED (abstract) |
| van der Sluis 2024, arXiv:2405.17874 (NUTS) | Random dimensionality reduction + NARS as a few-shot word recogniser | VERIFIED (abstract) |
| Xu 2026, arXiv:2607.20902 | Anti-goals in NAL: avoidance is not pursuit of a negated event | VERIFIED (abstract) |
| Goertzel et al. 2020, arXiv:2005.12535 | DeepWalk embedding differences roughly align with PLN "intensional difference" | VERIFIED (abstract) |
| Almeida & Casals 2026, "From Doyle to AGM", arXiv:2608.14567 | Survey of belief change from TMS to AGM with an implementation roadmap; §8.4 lists what neural systems add | VERIFIED (abstract, §8.4) |
| Darwiche & Pearl 1997; Jin & Thielscher 2005; Kern-Isberner et al. 2024 | Postulates for iterated revision and why they are not enough | MEMORY / ABSTRACT |
| Olsson & Westlund 2006, Erkenntnis 65 | An epistemic state must include a research agenda (open questions) besides beliefs and entrenchment | ABSTRACT |
| Thagard 1989, "Explanatory coherence" (ECHO) | Connectionist network: propositions are units, explanation is an excitatory link, contradiction an inhibitory one; acceptance = settled activation | ABSTRACT |
| Leitgeb 2001; Alviano, Giordano et al. 2023, arXiv:2305.00304 | Neural networks read as nonmonotonic / conditional knowledge bases with typicality | ABSTRACT; VERIFIED (abstract) |
| Wilie et al. 2024, arXiv:2406.19764 (Belief-R) | About 30 language models "generally struggle to appropriately revise their beliefs" | VERIFIED (abstract) |
| Minervini et al. 2020, arXiv:1912.10824 (GNTP), arXiv:2007.06477 (CTP) | Scaling fixes for neural theorem provers | VERIFIED (abstracts) |
| Maene & De Raedt 2023 (DeepSoftLog), NeurIPS | NTP's soft unification lacks non-redundancy, well-defined proof scores, non-sparse gradients | ABSTRACT |
| Manhaeve et al. 2018, arXiv:1805.10872 (DeepProbLog) | Neural predicates inside probabilistic logic programs | VERIFIED (abstract) |
| van Krieken, Acar, van Harmelen 2022, arXiv:2002.06100 | Which fuzzy operators can be trained by gradient descent | VERIFIED (abstract) |
| Marconato et al. 2023, arXiv:2302.01242, arXiv:2305.19951 | Neuro-symbolic continual learning; reasoning shortcuts | VERIFIED (abstracts) |
| Mao et al. 2019, arXiv:1904.12584 (NS-CL); Mei et al. 2022, arXiv:2203.16639 (FALCON) | Concept learners; FALCON learns a new concept from one or a few examples as a box embedding | VERIFIED (abstracts) |
| Chen et al. 2019, arXiv:1811.10667 (UKGE); Chen et al. 2021, arXiv:2104.04597 (BEUrRE) | Embeddings trained to reproduce the confidence scores of uncertain facts | VERIFIED (abstracts) |
| Richardson & Domingos 2006 (Markov logic); Jøsang 2016 (Subjective Logic) | Weighted first-order formulas; opinions (belief, disbelief, uncertainty) with fusion operators | MEMORY |
| Kemp, Perfors, Tenenbaum 2007, "Learning overhypotheses…" (neighbouring field) | Hierarchical Bayes: knowledge about a category level learned from instances and used on new ones | ABSTRACT |

---

## 4. Known failure modes of the field

### Stability

- **Inference explosion and hand-tuned attention (NARS, PLN).** Control rests on budget heuristics with many constants: priority, durability, quality, forgetting rates per bag, derivation "leak" factors. [V `Parameters.java`] An outside researcher working from NAL writes that the NARS algorithm "is very naive, and it is difficult to eliminate useless inference and obtain significant inference results within a valid time". [V NATL §3.4.2] The ONA paper itself is reported to call the allocation of resources between input and derived events a difficult balance. [A]
- **Evidence only accumulates.** Revision adds evidence weights, so confidence never falls (ONA: `max(w2c(w1+w2), c1, c2)`), capped at 0.99, which is 99 evidence units. [V] A well-supported belief therefore moves slowly when the world changes: at the cap, one contrary observation of unit weight moves frequency from 1.00 to 0.99. [D] ONA discounts only events over time (projection decay 0.8 per step), not eternal beliefs. [V] This is a slowdown, not an inability: in a small operant task ONA "successfully adjusted its behavior when task conditions were reversed". [V arXiv:2405.19498 abstract]
- **Double counting leaks in.** ONA stamps hold 10 evidence ids; merged stamps are cut to that length, after which overlap with dropped ids is no longer detected. [V `Stamp.c`; D] Every input sentence also gets a fresh evidence id, so the same sentence repeated by the same source counts as new evidence each time. [V `Stamp.h` comment; D]
- **Learned truth values drift.** When Differentiable PLN learns rule formulas together with truth values, the values stop meaning probabilities. [V]
- **Logic and gradients disagree.** Many standard fuzzy operators are "highly unsuitable" for gradient learning; implications give "a strong imbalance between gradients driven by the antecedent and the consequent"; the best-performing combinations "no longer satisfy the usual logical laws". [V van Krieken et al.]
- **Learned evidence is not evidence.** Second-order loss minimisation has identifiability and convergence problems; results depend on the regulariser. [A; V for the conclusion]
- **Concepts with the wrong meaning.** Neuro-symbolic predictors "can attain high accuracy but by leveraging concepts with unintended semantics" (reasoning shortcuts). [V Marconato et al.]
- **Edited weights do not stay coherent.** Edits "generalize poorly with respect to other relevant beliefs" [V Hase]; language models that do update on new evidence "often underperformed in scenarios without necessary updates". [V Wilie]

### Forgetting

- **NARS forgets by eviction, not by interference.** Beliefs are separate items; the lowest-priority concepts and links are dropped when memory is full (ONA 4096 concepts; OpenNARS 10000 concepts, 100 task links and 100 term links each). [V] Xu's model "theoretically does not suffer from catastrophic forgetting, and the practical results also support this". [V] The price is that evicted knowledge is simply gone; there is no consolidation into a smaller form. [D]
- **Neural-symbolic systems forget catastrophically.** "Stock neuro-symbolic architectures suffer from catastrophic forgetting"; adding continual-learning strategies helps but produces reasoning shortcuts. [V Marconato et al.]
- **AGM and truth maintenance have no notion of forgetting**; contraction is a deliberate removal. [M]

### Scale

- **NARS**: the default ONA build holds 255 atomic terms and 4096 concepts [V]. Published demonstrations are Pong and Space Invaders variants, cart-pole, grid worlds, small robots, a first-responder assistant [V ONA README], and seven Gym tasks where ONA is on a par with Q-learning. [V] I found no demonstration at the scale of open-domain language or vision. [D from the searches]
- **PLN / Hyperon**: "in-progress software system" [V]; the MeTTa interpreter is slow enough that the MeTTa-NARS port recommends a compiler instead. [V metta-nars README]
- **Neural theorem provers**: "can only be used with small-scale symbolic KBs". [V GNTP abstract]
- **∂ILP**: memory limits predicates to arity two. [V]
- **AERA**: one laboratory demonstration (the interview task); no independent evaluation found. [V; A]

### Language

- NARS speaks Narsese. English enters either through a shallow parser — "noun is adjective" becomes `<noun --> [adjective]>`, with a flag for "not"/"no" and nothing for frequency words such as "sometimes" [V `english_to_narsese.py`] — or through GPT-4. [V]
- In 2026 the NARS group still treats translation from natural language to Narsese as a benchmark task for a trained language model. [V arXiv:2604.18873]
- The 2023 NAL sequence model is judged by its author as not yet sensible for natural language. [V]
- Neuro-symbolic concept learners parse questions into a small domain-specific program language. [V]
- Where language is native (language model plus belief layer), the beliefs live outside the weights. [V]

### Meaning of the numbers

- NARS frequency/confidence and PLN strength/confidence are different quantities with different formulas; PLN's own author describes confidence handling by "a simple independence assumption on confidences". [V arXiv:2412.19524]
- In NAL any conclusion of induction, abduction or comparison has confidence at most 1/(1+k) = 0.5, because its evidence weight is at most one unit. [D from V formulas]
- A probability alone cannot support revision; a second quantity is needed. [V Wang 1993]

---

## 5. The closest existing system on the owner's "red" example

**System chosen: NARS, in its maintained implementation ONA, with OpenNARS consulted for the self-monitoring operators.** Reasons: it is the only running system in the field that stores a graded belief from one sentence, refines it on contrary evidence without deleting it, induces a category statement from instances, and deduces from the category to instances — all online. [D] PLN does the same in principle but needs base rates for induction and its implementation is in transition. [V] VCML has the category level ("same kind") but learns it by batch training on a dataset. [V]

### 5.1 Input

In Narsese (ONA syntax):

```
<red --> [dark]>.            // "red is dark"                      default truth {1.0 0.9}
<red --> [dark]>. {0.0 0.9}  // "...but sometimes it is not dark"  negative evidence on the same statement
<red --> colour>.            // red is a colour
<yellow --> colour>.         // another colour, nothing known about its lightness
<yellow --> [dark]>?         // a question the user must type
<?1 --> colour>?             // "which colours are there?" - also typed by the user
```

- ONA's English channel would turn "red is dark" into `<red --> [dark]>.` [V rule "noun BE adjective"]. It has no rule for "sometimes"; "red is sometimes light" would not become a lower frequency on the first statement. [V; D] The step from the sentence to "contrary evidence about the same statement" has to be done by whoever writes the Narsese. [D]

### 5.2 Truth functions used

Read in ONA `src/Truth.c` [V]:

- confidence from evidence: c = w/(w+k), w = k·c/(1−c), k = 1
- revision: w = w₁+w₂, f = (w₁f₁+w₂f₂)/w, c = w/(w+k)
- deduction `(S-->M),(M-->P) |- (S-->P)`: f = f₁f₂, c = c₁c₂f
- induction `(A-->B),(A-->C) |- (C-->B)`: f = f(A-->B), c = w/(w+k) with w = f(A-->C)·c₁·c₂
- expectation (used for decisions): e = c·(f − 0.5) + 0.5

Cross-check without running: ONA's own example `examples/nal/penguin.nal` lists the expected answers f = 1.0, c = 0.729 and f = 0.0, c = 0.81, which are 0.9³ and 0.9² as the deduction formula gives. [V]

### 5.3 What happens, with ONA's defaults (every input f = 1.0, c = 0.9, i.e. 9 units of evidence)

All values below are [D], computed from the [V] formulas.

| Step | Statement | f | c | Comment |
|---|---|---|---|---|
| 1 first hearing | `<red --> [dark]>` | 1.00 | 0.900 | Stored strongly, not weakly. |
| 2 contrary evidence | same statement after revision | 0.50 | 0.947 | Not deleted: frequency moves to the evidence ratio, confidence rises. "Sometimes" is the frequency. |
| 3 induction with `<red --> colour>` | `<colour --> [dark]>` | 0.50 | 0.460 | The category statement is weaker than the instance it came from. |
| 3b two, three, four colours with the same history | `<colour --> [dark]>` | 0.50 | 0.630, 0.719, 0.773 | Confidence grows only with further instances. |
| 4 deduction back to red | `<red --> [dark]>` (derived) | 0.50 | 0.348 | Its stamp contains red's own evidence, so it is **not** merged; "choice" keeps the direct belief (0.50, 0.947). |
| 4b if the rule came from three *other* colours | red after revision | 0.50 | 0.949 | Merging is allowed; the gain is small (0.947 → 0.949). |
| 5 unseen colour | `<yellow --> [dark]>` | 0.50 | 0.348 | Expectation 0.5: the rule "sometimes dark" predicts nothing in either direction. |
| 6 question | — | — | — | None is produced. |

### 5.4 The same steps with weights chosen to match where the owner's example starts

The owner's numbers are single values that rise with support, so they are compared with NAL confidence. All values [D].

| Owner's example | NAL with matching weights | Result |
|---|---|---|
| "red is dark" 0.23 | c = 0.23 means w = 0.30 evidence units (ONA's default is 9.0). Alternatively one unit with horizon k = 3.35. | Possible by setting a parameter; not the default behaviour. |
| "dark but sometimes light" 0.57 | Adding one unit of evidence: c = 1.30/2.30 = **0.565**. Read as contrary evidence, f falls to 0.23; read as support for the refined statement, f stays 1.0. | **Reproduced exactly.** |
| "all colours are sometimes dark, sometimes light" 0.77 | Induction from red alone gives c = **0.337**; the ceiling for any single induction is 0.5. Reaching 0.77 takes seven such instances (0.781). | **Not reproduced.** In NAL a rule cannot be more certain than the one instance behind it. |
| red rises to 0.88 | Blocked when the rule rests on red (stamp overlap). With a rule at 0.77 built from other evidence: deduction gives c = 0.693 and red rises 0.565 → **0.781**. Reaching 0.88 needs a rule at c ≈ 0.95, which takes about 40 independent instances at these weights. | **Not reproduced**, and the self-support version is forbidden by design. |
| "which other colours exist?" arises by itself | Not generated. | **Not reproduced.** |

### 5.5 Why the last three steps fail, in NARS's own terms

- **Rule weaker than instance.** Induction passes on at most f·c₁·c₂ ≤ 1 unit of evidence per instance. [V] This is intended: one observed case is weak evidence for a general statement. [M]
- **No support from one's own rule.** `Inference_RevisionAndChoice` revises only if `Stamp_checkOverlap` is false; otherwise it keeps whichever belief is more confident. [V] The purpose, stated in the source, is "that evidence is only counted once in the conclusions the system makes". [V `Stamp.h`] The owner's step "the rule feeds support back to the instance" is, in this vocabulary, counting red's evidence twice. OpenNARS differs only in bookkeeping: a concept may hold up to 28 beliefs with different stamps [V `Parameters.java`], so the direct and the rule-derived belief about red can sit side by side, still unmerged. [M]
- **No spontaneous question.** In ONA, questions exist only as input; the code answers them and raises the priority of the answering concept. [V `NAR.c`] ONA's "active question" example obtains information only because the user supplied a goal that needs it. [V `examples/nal/activequestion.nal`] In OpenNARS a question can be derived from a *goal* ("how can I achieve it?"), behind two switches that are both off by default [V `ProcessGoal.java`, `Parameters.java`], and `^wonder` creates a question when it is executed as an operation [V `Wonder.java`]. Nothing links a newly induced, weakly supported generalisation to a question about its other instances. [D]

### 5.6 What NARS does that the example needs and neural systems lack

- Contradiction refines a belief in one step, without retraining [V] and without touching unrelated beliefs. [D]
- Order-independence: revision is a sum of evidence weights, so the result does not depend on the order in which evidence arrives (up to the 0.99 cap). [D from V formula]
- The rule transfers to **unseen** instances at once (step 5). [D]
- If "light" is entered as a separate term with no stated relation to "dark", NARS holds both beliefs and derives `<red --> ([dark] & [light])>` (1.0, 0.81) and also `<[dark] <-> [light]>` (1.0, 0.448): two properties of the same thing become "similar". [D] This is how properties get grouped into a dimension in NARS, and also a source of unwanted conclusions. [D]

### 5.7 What the neighbours would do

- **PLN**: revision behaves like NARS. Induction from red to "colour" needs P(red), P(colour), P(dark); with one instance the result is dominated by the base rate of "dark", not by red. [V formulas; D]
- **VCML**: after joint training it can answer "do red and green describe the same property?" and use that to learn a new colour from a few examples. [V] It does not hold a belief such as "colours are sometimes dark", and one sentence changes nothing in it. [D]
- **Hierarchical Bayes** (neighbouring field): the standard account of a category-level regularity learned from instances and applied to new ones. [A] It is computed by posterior inference over all data, not by incremental weights. [M]
- **Explanatory-coherence networks** (ECHO): the one classical model in which a hypothesis and what it explains raise each other's activation. [A] It has no evidence counts: activation measures fit with the rest of the network, not the amount of evidence. [M]

---

## 6. What is still unsolved, in the field's own terms

1. **Inference control.** What to think about next is decided by hand-set budget arithmetic in NARS and by attention-economy heuristics in OpenCog. [V; M] Learning the control has been proposed — reinforcement learning of the reasoner (NATL), learned rule selection (CTP) — but not inside an online evidential system. [V; D]
2. **Self-monitoring that does work.** NAL-9 mental operators exist as code that starts disabled and is triggered at random. [V] There is no account of how the system should measure the quality of its own constructions and steer by it. [D]
3. **Concepts below the term level.** NARS concepts are built from identifiers given by the input channel. [M] The SDR line (ANSNA) was abandoned for a return to terms. [V] Neural concept learners form concept vectors only by batch training and are prone to learning the wrong meaning. [V]
4. **Language.** No system in the field acquires or produces language through its own inference; all delegate to a parser or a language model. [V]
5. **Truth-value meaning under learning.** As soon as truth values or rule formulas are trained by gradient descent, their meaning is not guaranteed (Differentiable PLN), learned "evidence" is not faithful (evidential deep learning), and operators that train well break logical laws. [V] No differentiable system keeps additive evidence and learns online. [D]
6. **Provenance of evidence in a distributed store.** Stamps are lists of discrete evidence ids, truncated in practice. [V] There is no known equivalent for weights shared by many beliefs, so independence of evidence cannot be checked there. [D]
7. **Support between a category and its instances.** NAL forbids a rule from strengthening its own sources; hierarchical Bayes handles it but in batch. [V; M] An incremental, bounded rule for this is missing. [D]
8. **Source reliability and repetition.** Each input counts as fresh evidence; the same claim repeated is indistinguishable from independent confirmation. [V; D] The 2026 belief-change survey lists "trust and source reliability" among open implementation challenges. [V]
9. **Loss of plasticity through accumulated evidence.** Confidence can only rise; the cap at 0.99 is the only brake. [V; D]
10. **Iterated revision.** AGM is silent on it; the Darwiche–Pearl postulates are reported as too permissive or not sufficient. [A] Ranking theory iterates cleanly but needs a firmness number for every input and says nothing on where it comes from. [M]
11. **Where questions come from.** Interrogative belief revision adds a "research agenda" to the epistemic state and asks how it should change. [A] In NARS questions come from outside or from goals. [V] No implemented mechanism raises a question because a generalisation has just been formed. [D]
12. **Beliefs inside weights.** "It is also not clear that LLMs have coherent, revisable beliefs to begin with." [V Hase] The belief-change survey: beliefs encoded in network weights "require new approaches to belief identification, extraction, and modification". [V]
13. **Evaluation.** NARS evaluations are small control tasks and psychology-style demonstrations [V]; a natural-language benchmark with executable Narsese appeared only in 2026 and labels each problem True, False or Uncertain. [V] I found no shared benchmark for lifelong evidential learning. [D]
14. **Two literatures that do not meet.** The full text of the 2026 survey "From Doyle to AGM" contains no occurrence of "NARS" or "non-axiomatic". [V]

---

## 7. What this means for the design (my inferences)

All points in this section are [D].

1. **Not new**: two-number graded beliefs with revision; induction of a category statement from instances; deduction to unseen instances; online operation without a training phase; bounded memory with eviction. NARS has had all of these, in symbols, for decades. A new system must at least match NARS on them, and the NAL numbers in section 5 are a ready baseline.
2. **New ground 1 — evidence accounting inside tensors, updated online.** Nobody has a tensor engine in which a weight is an evidence-backed belief that a single experience revises in place. ANSNA was the nearest and was dropped; Differentiable PLN and evidential deep learning are batch-trained and lose the meaning of the numbers.
3. **New ground 2 — category-to-instance support with a bound.** The owner's example (rule above its only instance; instance lifted by its own rule) is not NAL. It is a different evidence semantics: fitting into a generalisation counts as support. NARS forbids it to avoid circular self-confirmation. A design that wants it needs an explicit rule for how much coherence may add and a proof or test that the loop instance → rule → instance converges below certainty.
4. **New ground 3 — questions from generalisations.** Nothing in the field turns "a rule was just formed with little support" into "ask for more members of the category". The nearest formal notion is the research agenda of interrogative belief revision.
5. **New ground 4 — learned control and self-measurement.** P4 and P5 are empty columns. The only tensor-side hint is that rule formulas can be made trainable (Differentiable PLN), with the known danger that the numbers lose their meaning.
6. **Tests this field suggests** (each separates a real mechanism from an imitation):
   - Revision is order-independent: the same evidence in any order gives the same belief.
   - Repetition versus independence: the same sentence from the same source twice must add less than two independent confirmations.
   - No runaway: feeding a rule back to its source instances repeatedly must converge, and to a value below what independent evidence would give.
   - Transfer: a rule induced from k instances must change the prediction for an unseen instance immediately, with confidence that grows with k.
   - Plasticity after saturation: after many consistent observations, a run of contrary ones must still move the belief at a stated rate.
   - Locality: revising one belief must leave unrelated beliefs unchanged (the property that spares NARS catastrophic forgetting).
   - A question about the unseen extension of a new rule must appear without being asked.

---

## 8. Sources

**Read directly (primary)**
- ONA source and README, commit dc4efd0abd520cdb79bf53bfa3c285ebb24f2e8a (2025-05-16): https://github.com/opennars/OpenNARS-for-Applications — `src/Truth.c`, `src/Truth.h`, `src/Config.h`, `src/NAL.h`, `src/Stamp.c`, `src/Stamp.h`, `src/Inference.c`, `src/Memory.c`, `src/Cycle.c`, `src/NAR.c`, `src/Decision.c`, `english_to_narsese.py`, `examples/nal/penguin.nal`, `examples/nal/activequestion.nal`, `examples/nal/categoryformationanddetection.nal`
- OpenNARS source (master, fetched 2026-11-14): https://github.com/opennars/opennars — `inference/TruthFunctions.java`, `inference/LocalRules.java`, `main/Parameters.java`, `plugin/mental/InternalExperience.java`, `operator/mental/Wonder.java`, `control/concept/ProcessGoal.java`
- READMEs: https://github.com/patham9/ANSNA , https://github.com/opennars/NARS-GPT , https://github.com/trueagi-io/PLN , https://github.com/trueagi-io/hyperon-pln , https://github.com/patham9/metta-nars , https://github.com/patham9/NACE
- arXiv full texts: 1303.1517, 1312.6764, 1705.11040, 1711.04574, 1805.10872, 1806.01768, 1811.10667, 1904.12584, 1907.04592, 1912.10824, 2002.01464, 2002.06100, 2005.12535, 2006.13155, 2007.06477, 2012.13635, 2104.04597, 2109.14723, 2203.16639, 2210.06316, 2302.01242, 2304.03291, 2305.00304, 2305.02171, 2305.14250, 2305.19951, 2308.12486, 2310.18318, 2402.09056, 2405.03340, 2405.17874, 2405.19498, 2406.19354, 2406.19764, 2412.19524, 2501.04832, 2505.07079, 2506.13157, 2511.08392, 2604.18873, 2607.20902, 2608.14567 (each at https://arxiv.org/abs/<id>)

**Seen only as abstract or snippet (hosts blocked or not on arXiv)**
- ONA paper: https://cis.temple.edu/tagit/publications/ONA.pdf
- Self in NARS: https://www.frontiersin.org/articles/10.3389/frobt.2018.00020/full
- ANSNA paper and slides: https://agi-conf.org/2019/wp-content/uploads/2019/08/ANSNA_Praes.pdf
- Kisby, Blanco, Moss: https://ojs.aaai.org/index.php/AAAI/article/view/29409
- Maene & De Raedt (DeepSoftLog): https://proceedings.neurips.cc/paper_files/paper/2023/hash/bf215fa7fe70a38c5e967e59c44a99d0-Abstract.html
- Thagard, Explanatory coherence: https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/abs/explanatory-coherence/E05CB61CD64C26138E794BC601CC9D7A
- Olsson & Westlund: https://link.springer.com/doi/10.1007/s10670-006-9001-6
- Kemp, Perfors, Tenenbaum: https://digital.library.adelaide.edu.au/items/9d67aa34-4d2e-44a5-ace2-2bdf8d24fb3b
- Leitgeb, Nonmonotonic reasoning by inhibition nets: https://epub.ub.uni-muenchen.de/49649/
- Jürgens et al. (proceedings page): https://proceedings.mlr.press/v235/juergens24a.html

**From memory, not checked**: Wang 2013 (book text), Goertzel et al. 2008 (book text), Alchourrón–Gärdenfors–Makinson 1985, Darwiche–Pearl 1997, Spohn 1988/2012, Doyle 1979, de Kleer 1986, Richardson–Domingos 2006, Jøsang 2016.

**Reproducing the numbers in section 5**: `python3 -I /tmp/claude-0/-workspace-cevahir-ai/79909244-c51d-5307-ad00-1a1e5ef77203/scratchpad/newcore/landscape/_tools_nars/red_example_nal.py`
