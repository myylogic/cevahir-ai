# Landscape: cognitive architectures and theories with an operating process — mapped against P1–P6

Date of survey: 2026-11-14. Scope: Soar, ACT-R, Sigma, LIDA and other global-workspace
implementations, CLARION, hierarchical temporal memory (HTM), Sparse Distributed Memory and its
vector-computing descendants, Leabra, Spaun/NEF, the Thousand Brains proposals (Monty), AERA,
Copycat/Metacat, the Common Model of Cognition, neural realisations of these (CogNGen, shared
workspace), reality-monitoring theories, and LLM-era cognitive architectures (CoALA, Generative
Agents, and an audit of such systems). Three questions were put to each: what learns online, what
forms concepts, what monitors itself.

Four sibling notes in this directory cover neighbouring fields and are not repeated here:
`nars-logic.md` (NARS, PLN, evidential truth values), `art-growth.md` (ART, growing networks,
hierarchical Bayesian concept formation), `online-plasticity.md` (online weight learning),
`predictive-energy.md` (predictive coding, energy-based and active-inference systems).

## Summary

1. No system in this field satisfies P2, P3, P4 or P5 natively, and none satisfies P6 with language. [I]
2. P1 is native in 10 of the 25 works (3 of them only on paper). In every one of them, what changes
   at each step is either a number attached to a symbol structure that a person wrote (ACT-R, Sigma,
   AERA, CLARION) or a local synapse in a system that does not reason (HTM, CogNGen). [I]
3. The field's standard design keeps the graded numbers behind a wall. In ACT-R and Soar, activation
   and utility "are not represented in working memory and they cannot be tested or directly modified
   by agent data processing. Thus, there is a 'wall' between agent data and agent metadata." [V]
   The numbers say how available or how useful an item is, not how far it is believed. [I]
4. Closest to the owner's "red" example: **AERA**. It stores a first observation at 0.50, lowers it on
   contrary evidence without deleting it, builds a general model by replacing a value with a
   variable, and lets a prediction made through the general model carry a higher number (0.92) than
   the instance's own model (0.75). [V for the mechanism, C for the numbers] It does not write that
   number back on the instance, it discards the instance model, it asks no question, it is symbolic
   code, and it did not acquire language. [V]
5. Strongest system that gets part of the way: **ACT-R declarative memory with blending**. After one
   contrary case the blended answer for red is 0.27; after one more confirming case 0.91. A single
   hearing is recalled at full value 1.0; what is weak is the chance of recall. No rule is ever
   formed. [C from equations read in the source]
6. **CLARION** has the most matching parts on paper (implicit graded level, explicit rules extracted
   and generalised online, a monitoring subsystem), but its explicit rule is deleted when it fails
   and has no earlier version to return to. [V] Its technical sources could not be reached; the
   formula used in section 6 is from memory. [M]
7. Concepts formed when needed exist in two systems, and both lean on a teacher: Soar's analogical
   concept memory creates a concept when it meets an unknown word, and stores an example only when
   comprehension fails [V]; Monty opens a new object model when an object is new, with object
   identity and pose supplied in all reported experiments [V].
8. Self-monitoring in this field is symbolic. Of 35 metacognitive architectures reviewed in 2025 only
   one works on sub-symbolic data alone, and an earlier review "identified no emergent architecture
   as metacognitive". [V]
9. A mechanism that tags the system's own constructions as real or not real, and feeds belief
   formation, exists only on paper: the 2023 consciousness report says of perceptual reality
   monitoring "there have been no implementation attempts that we are aware of" [V]; the 2021
   "cognitive reality monitoring network" was not simulated [V]. Two searches in this session found
   no implementation since. [A]
10. Questions arise in this field only from an impasse inside a task (Soar, Rosie) or from a counter
    that triggers an LLM prompt (Generative Agents). None arises from a newly formed category. [V; I]
11. Language is the weakest side. Soar's own author rates it "partial": "even Rosie cannot process
    unrestricted language". [V] Since 2023 the field's answer has been to attach a pre-trained
    Transformer. [V for CLARION, A for Soar and ACT-R]
12. Every central step in these architectures is settled by a single-winner competition (operator
    selection, production utility, coalition, winning chunk, soft-max retrieval). Nothing in the
    field offers a precedent for leaving soft-max and arg-max. [V per system, I for the generalisation]
13. LLM-era architectures learn by writing text into stores while the weights stay fixed. Their known
    failures are self-reinforcing errors in reflective memory, drift under repeated summarising, and
    modules that carry a cognitive name but compute nothing that reaches the output (18 of 25 in one
    audited system). [V]

---

## 0. How to read this note

### Status tags (every statement carries one)

| Tag | Meaning |
|---|---|
| **[V]** VERIFIED | I read the passage in the primary source in this session (full text from arxiv.org). "V in part" in a table means the abstract page plus the passages cited in the text were read, not the whole paper. |
| **[A]** ABSTRACT | Only an abstract page, a search-result snippet or a catalogue entry was seen. |
| **[M]** MEMORY | From my own knowledge; not checked in this session. |
| **[C]** COMPUTED | My arithmetic from equations or parameters tagged [V] (or [M] where stated). Not a number found in a source. |
| **[I]** INTERPRETATION | My reading: the mapping of a mechanism to P1–P6, or a generalisation over sources. Not a claim found in a source. |

### Ratings

N = satisfied natively, P = partially, – = not at all. P† = only prediction-versus-observation
checking, which is a narrow part of P4. N* = native in the proposal, with no implementation. All
ratings are [I].

### The principles (the owner's, short names)

**P1** learning and running are one process: every experience changes the weights at once.
**P2** weights are graded beliefs: (i) a first hearing is stored weakly, (ii) contradicting evidence
refines instead of deleting, (iii) a category rule is induced, (iv) the rule feeds support back to
the instance. **P3** concepts are formed when needed, not given by a fixed embedding table;
categories emerge from instances. **P4** imagination: the system observes its own constructions,
measures how real they are and advances by that measure. **P5** what is acquired is the way of
constructing (revise, generalise, feed back, ask); it improves with experience and transfers.
**P6** an engine of tensors and equations, with language in and out eventually.

### What was done

30 web searches (28 standard, 2 extended) and 27 queries to the arXiv API; 65 papers downloaded
from arxiv.org; the passages cited below were read in 34 of them, the abstract page only in 21, and
10 were downloaded but not used. The worked example was computed by hand with a script
(`tools-cognitive-architectures/red_example_cogarch.py`). No third-party code was run.

### Access limits in this session

- Only arxiv.org and export.arxiv.org were reachable for reading. Two other hosts were tried
  (ojs.aaai.org, cmu-psych-fms.github.io) and did not resolve; after that I stopped trying other
  sites. aclanthology.org was not needed.
- Consequence: the founding publications of several architectures are **not** on arXiv and could not
  be read. This affects ACT-R (Anderson et al. 2004), Sigma (Rosenbloom, Demski, Ustun 2016),
  CLARION (Sun 2016; Sun, Merrill, Peterson 2001), LIDA (Franklin et al. 2014), Leabra (O'Reilly et
  al. 2012), Spaun (Eliasmith et al. 2012), the Standard Model (Laird, Lebiere, Rosenbloom 2017),
  Copycat/Metacat (Mitchell 1993; Marshall 2006), and the 2024 Soar-plus-Transformer proposal. For
  each I used a later arXiv paper by the same group where one exists, and marked the rest [A] or [M].
- arXiv:1412.6616 (a Sigma paper on word vectors) is withdrawn; no text was available.

---

## 1. What unites the field

- A cognitive architecture fixes a small set of memories and one repeating cycle, and claims that
  all behaviour comes from knowledge run through that cycle. The Common Model states it as:
  perception feeds working memory; "On each cognitive cycle, procedural memory, testing the contents
  of working memory, selects a single action"; long-term memories "have associated automatic
  learning mechanisms that incrementally modify and extend their contents". [V, arXiv:2506.07807]
- Learning during operation is therefore the norm here, unlike in deep learning. What is learned is
  mostly numbers on existing structures and compiled versions of the system's own reasoning. [I]
- A 2016–2020 review of 84 architectures sorts them into symbolic, emergent (connectionist) and
  hybrid, and finds the two ends do different things: "many of the emergent architectures, e.g. ART,
  HTM, DAC, BBD, BECCA, simply do not address reasoning". [V, arXiv:1610.08602] (The count 84 is
  from the published version [M]; the arXiv text says about three hundred architectures exist in
  total [V].)
- The same review: about one third of the surveyed architectures, "mainly symbolic or hybrid ones
  with a significant symbolic component, support metacognition", and "its usefullness has only been
  demonstrated in limited domains". [V]
- Since 2023 the field has split in two directions: old architectures attach LLMs for language and
  knowledge [V, A], and LLM agent builders borrow the old vocabulary (memories, cycle, workspace)
  [V].

---

## 2. Main table: 25 works

| # | Title | First author | Year | Venue or arXiv id | Status |
|---|---|---|---|---|---|
| 1 | Introduction to Soar (Soar 9.6) | Laird | 2022 | arXiv:2205.03854 | VERIFIED |
| 2 | An Analysis and Comparison of ACT-R and Soar | Laird | 2022 | ACS 2021; arXiv:2201.09305 | VERIFIED (ACT-R's own papers: MEMORY) |
| 3 | SpeedyIBL: A Comprehensive, Precise, and Fast Implementation of Instance-Based Learning Theory (ACT-R activation and blending) | Nguyen | 2021 | arXiv:2111.10268 | VERIFIED (equations, abstract) |
| 4 | Controlling Synthetic Characters in Simulations: A Case for Cognitive Architectures and Sigma | Ustun | 2018 | I/ITSEC 2018; arXiv:2101.02231 | VERIFIED (main Sigma paper, JAGI 2016: MEMORY) |
| 5 | Enhancing Computational Cognitive Architectures with LLMs: A Case Study (CLARION) | Sun | 2025 | arXiv:2509.10972 (with arXiv:2401.10444) | VERIFIED (CLARION's technical books and papers: MEMORY) |
| 6 | LIDA: A Systems-level Architecture for Cognition, Emotion, and Learning | Franklin | 2014 | IEEE Trans. Autonomous Mental Development 6(1) | ABSTRACT (search snippets) and MEMORY |
| 7 | Continuous online sequence learning with an unsupervised neural network model (HTM) | Cui | 2016 | Neural Computation; arXiv:1512.05463 | VERIFIED |
| 8 | Thousand-Brains Systems: Sensorimotor Intelligence for Rapid, Robust Learning and Inference (Monty) | Leadholm | 2025 | arXiv:2507.04494 | VERIFIED |
| 9 | The Thousand Brains Project: A New Paradigm for Sensorimotor Intelligence; The Thousand Brains Theory 2.0 | Clay; Hawkins | 2024; 2025 | arXiv:2412.18354; arXiv:2507.05888 | VERIFIED in part |
| 10 | Autonomous Learning with High-Dimensional Computing Architecture Similar to von Neumann's (Sparse Distributed Memory lineage) | Kanerva | 2025 | arXiv:2503.23608 | VERIFIED in part (Kanerva 1988 book: MEMORY) |
| 11 | Deep Predictive Learning in Neocortex and Pulvinar (Leabra family) | O'Reilly | 2020 | arXiv:2006.14800 | ABSTRACT plus a few lines (Leabra chapter 2012: MEMORY) |
| 12 | BioSpaun: A large-scale behaving brain model with complex neurons (describes Spaun) | Eliasmith | 2016 | arXiv:1602.05220 | VERIFIED in part (Spaun, Science 2012: MEMORY; Spaun 2.0 thesis: ABSTRACT) |
| 13 | A Proposal to Extend the Common Model of Cognition with Metacognition | Laird | 2025 | arXiv:2506.07807 | VERIFIED (Standard Model, AI Magazine 2017: MEMORY) |
| 14 | Bounded Recursive Self-Improvement (AERA) | Nivel | 2013 | Reykjavik Univ. report RUTR-SCS13006; arXiv:1312.6764 | VERIFIED |
| 15 | Analogical Concept Memory for Architectures Implementing the Common Model of Cognition (Aileen) | Mohan | 2022 | arXiv:2210.11731 (ACS 2020 version arXiv:2006.01962) | VERIFIED |
| 16 | A Computational Model for Situated Task Learning with Interactive Instruction (Rosie) | Mohan | 2016 | arXiv:1604.06849 | VERIFIED in part |
| 17 | Copycat and Metacat (Fluid Concepts and Creative Analogies; a self-watching model of analogy-making) | Hofstadter, Mitchell; Marshall | 1995; 2006 | Basic Books; J. Exp. Theor. AI | MEMORY, ABSTRACT (search), and the description of Metacat in arXiv:2503.13467 VERIFIED |
| 18 | A Neuro-mimetic Realization of the Common Model of Cognition via Hebbian Learning and Free Energy Minimization (CogNGen) | Ororbia | 2023 | arXiv:2310.15177 | VERIFIED |
| 19 | Coordination Among Neural Modules Through a Shared Global Workspace | Goyal | 2021 | ICLR 2022; arXiv:2103.01197 | VERIFIED in part |
| 20 | A Theory of Consciousness from a Theoretical Computer Science Perspective: Insights from the Conscious Turing Machine | Blum | 2021 | arXiv:2107.13704 | VERIFIED in part |
| 21 | Consciousness in Artificial Intelligence: Insights from the Science of Consciousness (indicator properties; perceptual reality monitoring) | Butlin | 2023 | arXiv:2308.08708 | VERIFIED in part |
| 22 | From internal models toward metacognitive AI (cognitive reality monitoring network) | Kawato | 2021 | arXiv:2109.12798 | VERIFIED in part |
| 23 | Cognitive Architectures for Language Agents (CoALA) | Sumers | 2023 | TMLR 2024; arXiv:2309.02427 | VERIFIED |
| 24 | Generative Agents: Interactive Simulacra of Human Behavior | Park | 2023 | arXiv:2304.03442 | VERIFIED in part |
| 25 | Real Computation or Merely Naming? A Pre-Registered Mechanism Audit of 25 Cognitive-Architecture Modules in an LLM-Based Agent System | Chen | 2026 | arXiv:2610.41604 | VERIFIED in part (abstract, introduction) |

### 2.1 Mechanism (two sentences) and P-reading for each row

**1. Soar.** Rules propose, evaluate and apply operators over a symbolic working memory in a fixed
decision cycle; when knowledge does not suffice "an impasse arises" and a substate is created in
which the same machinery works on the gap. [V] Learning is chunking (a result produced in a substate
is compiled into a rule), reinforcement learning on numeric operator preferences, automatic storage
of one episode per decision, and deliberate storage into semantic memory. [V]
- P1 **P**: chunking, RL and episodic storage run during operation [V]; semantic memory "does not
  have an automatic learning mechanism", and "agents must be designed with learning in mind" [V].
- P2 **–**: facts carry no degree; numbers exist only as activation metadata and as values of
  operators [V].
- P3 **P**: symbols and rules are created at run time; but Rosie is "unable to learn new abstract
  symbolic concepts on its own", and Soar "does not have innate capabilities for learning perceptual
  categories" [V].
- P4 **P**: the impasse is an automatic detection that the system's own knowledge is insufficient,
  and substates can simulate actions [V]; nothing measures how real a construction is [I].
- P5 **P**: chunking converts the system's own deliberation into skill [V]; the learning mechanisms
  themselves never change [I].
- P6 **–**: symbolic; language is hand-built comprehension knowledge [V].

**2. ACT-R as analysed beside Soar.** Both keep symbolic chunks and rules ("agent data") and attach
numeric "metadata" (base-level activation, utility) that the architecture updates as a side effect
of every retrieval and rule firing and uses to bias retrieval and selection. [V] ACT-R adds a chunk
to declarative memory whenever a buffer is cleared, and compiles two rules that fire in sequence
into one rule that "is initialized with low utility, so it will not immediately replace the original
pair". [V]
- P1 **N** for the numbers and for chunk storage: there is no separate training phase [V; I].
- P2 **P**: graded, and nothing is deleted ("There are no explicit forgetting mechanisms for DM,
  although DM chunks with sufficiently low base-level activation will never be retrieved") [V]; but
  the numbers measure availability, and the agent cannot read them [V].
- P3 **–**: chunk types, slots and similarities come from the modeller [M].
- P4 **P** (thin): "Module status data ... provides an agent with its only source of information
  about the current state of module processing" [V].
- P5 **P**: production compilation [V].
- P6 **P**: the metadata layer is a handful of equations; the content is symbolic [I].

**3. Instance-based learning with blending (ACT-R declarative memory in decision form).** Each
experienced (situation, action, outcome) is stored as an instance with its time stamps; at decision
time each instance's activation (logarithm of summed power-law-decayed recency, plus similarity,
plus noise) is turned into a retrieval probability by a Boltzmann soft-max, and the expected outcome
is the probability-weighted "blend" of the stored outcomes. [V] The option with the maximum blended
value is chosen. [V]
- P1 **N**: each outcome is stored and used at once [V].
- P2 **P**: graded by recency and frequency, contradicting outcomes coexist and are averaged [V]; a
  single hearing is recalled at full value [C]; no rule is induced [I].
- P3 **–**: "In symbolic IBLT, mismatch is determined by the task variables and partial matching
  functions chosen by the modeler." [V, arXiv:2610.03413]
- P4 **–**. P5 **–**.
- P6 **P**: equations over a growing table; the paper itself names "the curse of exponential growth"
  in computing time as instances accumulate [V, abstract].

**4. Sigma.** Predicates and conditionals written by the modeller compile into a factor graph whose
functions are piecewise-linear over symbolic, discrete and continuous variables; inference is
message passing (the summary-product algorithm). [V] Each cognitive cycle "comprises input, graph
solution, decisions, learning, and output", and the learning phase "modifies the long-term functions
in the graph via a general form of gradient-descent learning", with messages into function nodes
supplying the gradients. [V]
- P1 **N**: learning is a phase of every cycle [V].
- P2 **P**: long-term functions are distributions, so every stored relation is graded and nothing is
  deleted [V; I]; one number per relation, no separate measure of how much evidence stands behind it,
  and no induction of new relations [I].
- P3 **–**: predicates and their types are declared [V].
- P4 **P**: architectural appraisal variables, "attention and curiosity", the latter "grounded in a
  combination of familiarity and surprise", steer exploration [V].
- P5 **–**.
- P6 **N** for the engine: "a single general representation and inference algorithm for processing
  symbols, probabilities, distributed neural vectors, and signals" [V]. Decisions take "the operator
  associated with the highest value" [V], i.e. arg-max. Language: small demonstrations only [M].

**5. CLARION.** Four subsystems (action-centred, non-action-centred, motivational, metacognitive),
each with an explicit symbolic top level and an implicit neural bottom level; top-level "chunks"
(each "defined by a set of dimension-value pairs") are linked to microfeature nodes below, so the two
levels activate each other. [V] In bottom-up learning an explicit rule is extracted from the
bottom level and refined: generalised by testing whether a condition element can be removed "when
the rule is successful", specialised by "going back to a previous version" when it is not, and
deleted "if no such previous version" exists; this is "based on statistical measures (such as
information gain)". [V]
- P1 **N**: the forms of learning "enable incremental, continuous, real-time learning in Clarion from
  situations encountered" [V].
- P2 **P**: the implicit level holds a graded trace and the explicit level holds rules with match
  statistics [V; M]; (i) and (iii) are present, (ii) fails at the explicit level because a failing
  rule with no earlier version is deleted [V], (iv) exists as top-down activation [V] but not as a
  stored increase of the instance [I].
- P3 **P**: chunks and rules are created during learning; the dimensions and values they are built
  from are given [V; I].
- P4 **P**: the metacognitive subsystem regulates the others and generates reinforcement by
  "evaluating ... the overall satisfaction of activated drives" [V]; a 2025 review lists CLARION among
  systems that attach "confidence scores for candidate solutions" [V, arXiv:2503.13467].
- P5 **P**: the metacognitive subsystem chooses among reasoning methods [V]; the methods are given [I].
- P6 **P**: half neural, half symbolic; the author writes that such models "thus far have only
  limited computational capabilities" and proposes LLMs as the bottom level [V].

**6. LIDA.** Cognition is a rapid series of cycles (about 10 per second); in each, attention codelets
form coalitions from the current situational model, one coalition wins the global workspace, and its
content is broadcast to all modules. [A] Learning happens as the broadcast reaches each memory:
"new entities and associations, and the reinforcement of old ones", in perceptual, episodic and
procedural memory. [A] Perceptual memory is a semantic net with activation modelled on Copycat's
slipnet; episodic memory uses sparse distributed memory. [A]
- P1 **N** by design [A]; the implemented learning mechanisms are few [M].
- P2 **P**: nodes have a base-level activation that decays unless reinforced, so a new entity starts
  weak and can vanish [M]; no handling of contradiction that I know of [M].
- P3 **P**: perceptual learning creates new nodes [A].
- P4 **P**: the broadcast makes one content at a time available to everything [A]; it is not a
  measure of that content [I].
- P5 **–**. P6 **–**: codelets and nets; no language at scale [M].

**7. HTM sequence memory.** Sparse binary activity over columns of cells; each cell has dendritic
segments whose synapses carry a scalar "permanence", and a cell whose segment matches the previous
activity enters a predictive state. [V] Learning is a local Hebbian-like rule applied at every time
step ("it learns from every new data point"); a new synapse starts at permanence 0.21, below the
connection threshold 0.5, and moves by +0.1 or −0.1, or by −0.01 for a segment that predicted
wrongly. [V]
- P1 **N** [V].
- P2 **P**: (i) is literal here, a first co-occurrence leaves a synapse that does not yet conduct
  [V; I]; (ii) a wrong prediction costs 0.01, not the synapse [V]; several outcomes can be predicted
  at once [V]; (iii) and (iv) are absent [I].
- P3 **P**: sparse codes are formed by learning from encoders that a person designs [V]; no category
  level [I].
- P4 **P†**: mismatch between predicted and actual input [V].
- P5 **–**.
- P6 **P**: arrays and local rules, not differentiable; on the Reber grammar "HTM achieves an accuracy
  of 98.4% ... whereas LSTM achieves an accuracy of 100%", and it "can approximately learn artificial
  grammars by memorizing example sentences" [V].

**8. Monty (thousand-brains system).** Each learning module stores an object as a set of points
(location, local pose, features) in the object's own reference frame, built by moving a sensor over
the object and binding "active sensory features and active location" in "an instantaneous operation
... the simplest possible form of Hebbian learning". [V] Recognition keeps a set of hypotheses
(object, rotation, location), each with an evidence value raised or lowered by how well each new
movement and sensation fit; modules vote. [V]
- P1 **P**: learning is immediate and local, and 77 objects learned one after another are retained
  with "only a minor degradation" [V]; but in all reported experiments learning episodes and
  evaluation are separate and "Monty is provided with the ground-truth values of the object ID and
  orientation"; unsupervised learning "lies beyond the scope of the present work" [V].
- P2 **P**: evidence over hypotheses is graded during recognition; the stored model is not [V; I].
- P3 **P**: "when an object has not yet been learned, the LM initializes a new reference frame" [V];
  categories and abstract concepts are not handled [A, project forum].
- P4 **P†**: a hypothesis-testing policy moves the sensor to check predictions [V].
- P5 **–**.
- P6 **–**: explicit graphs and poses, no language; inference cost "scales linearly with the number of
  known models" [V].

**9. Thousand Brains Project white paper and Theory 2.0.** Every cortical column is taken to be a
complete sensorimotor modelling unit; many units communicate through a common message format and
reach consensus by voting. [V, abstract] The white paper lists as capabilities still to come:
"Generalizing modeling to abstract concepts derived from concrete models", "Modeling language and
associating it with grounded models of the world", and theory of mind. [V] Theory 2.0 adds roles for
hierarchical and thalamic connections in learning compositional objects, and states: "Some of the
theoretical ideas in this paper have not yet been tested in computational models, in particular,
the implications for learning of compositional objects." [V]
- Ratings as row 8; P3 and P6 are plans, not results [I].

**10. Kanerva's vector computing architecture (Sparse Distributed Memory lineage).** Items are
10,000-dimensional vectors; "Computing in superposition begins with D-dimensional seed vectors
chosen at random", combined by addition, multiplication and permutation, and stored in a
high-capacity associative memory modelled on the cerebellum. [V] Learning is accumulation by
addition in a single pass; the reported experiment identifies 21 languages from letter trigrams. [V]
- P1 **N** on paper: storing is adding [V; I].
- P2 **P**: counts in superposition are graded [I].
- P3 **P**: composite vectors are formed when needed, but every atomic item is a fixed random vector
  with no content [V; I].
- P4 **–**. P5 **–**.
- P6 **N** for the engine [I]; "We also need to exercise the theory in large-scale experiments" [V].

**11. Leabra family (deep predictive learning).** Rate-coded neurons with inhibitory competition
learn from the difference between a prediction and the outcome that arrives about every 100 ms,
"resulting in a biologically-plausible form of error backpropagation learning". [A] In a model of
the visual stream the higher layers come "to systematically categorize 3D objects according to
invariant shape properties, based solely on predictive learning from raw visual inputs" [A]; 156
objects fell into 5 categories [V, one passage].
- P1 **P**: the rule is local and runs on every trial, but models are trained over many epochs [A; M].
- P2 **P**: graded weights only [I]. P3 **P**: categories emerge from instances without labels [A],
  slowly [M]. P4 **P†** [A]. P5 **–**. P6 **N** for the engine; no language at scale [M].

**12. Spaun.** 2.5 million spiking neurons and about 8 billion connections implement vector
operations on "semantic pointers" and perform 8 tasks, from copying a digit to a reinforcement
learning task and induction over digit sequences. [V, as described in arXiv:1602.05220] Spaun 2.0
has about 6.6 million neurons, 12 tasks, and follows general instructions. [A]
- P1 **–**: connection weights are computed offline by the Neural Engineering Framework; online
  weight change is used in the reinforcement task only [M].
- P2 **–**. P3 **–**. P4 **–**. P5 **–**.
- P6 **N** for the engine (neurons computing vector algebra); no language [M].

**13. Common Model of Cognition with metacognition.** The proposal keeps the common cycle and adds
three things to working memory: a process-state buffer for each module, a memory of past
working-memory contents that can be retrieved and reasoned over, and "some means of creating past
or future hypothetical states". [V] Metacognition is then ordinary reasoning whose subject is the
agent's own processing; the authors contrast this with architectures where "specialized modules are
added 'on top' of cognition ..., as exemplified by MIDCA and Clarion". [V]
- P1 **P**, P2 **P**: as rows 1–2 [I]. P3 **–**.
- P4 **P**: hypothetical states and retrospective analysis are provided for; "Our proposal does not
  permit information in long-term memories to be examined or reasoned over by other modules", which
  "sacrifices omniscient metareasoning" [V]. Nothing measures how real a hypothetical state is [I].
- P5 **P**: what is learned about one's own reasoning is stored through the ordinary learning
  mechanisms [V]. P6 **–**.

**14. AERA.** Knowledge is a large set of small causal models (left pattern → right pattern), built
automatically when a goal succeeds unpredicted or a prediction fails, and run in parallel under a
priority scheduler; learning is "'always on' and inherent in the system's core operation". [V] Each
model has a reliability, "the number of times the model predicted correctly ... divided by the total
number of prediction attempts ... plus one", i.e. success rate times experience e/(e+1); below a
first threshold a model is "phased out" but still monitored, below a second it is deleted. [V]
- P1 **N** [V].
- P2 **P**, three parts of four: (i) one positive case gives 0.50 [C]; (ii) phasing-out, return to
  duty, and pre-condition models built from the failing cases [V]; (iii) "abstraction": a differing
  value is replaced by a variable and the general model competes with the specific ones [V]; (iv)
  absent, the specific cases "will be dismissed" [V].
- P3 **P**: "composite states" (conjunctions of facts, nestable) are formed from what is observed
  [V]; the sensor ontology is in the seed [V].
- P4 **P**: predictions are monitored against outcomes; goals and predictions produced in simulation
  carry "a simulation mark"; the system produces data about its own computation as internal inputs
  [V]. "Even though self-modeling for meta-control has not been leveraged in our S1 demonstrator,
  this functionality has been implemented" [V].
- P5 **P**: the stated aim is self-improvement "on the job, by modeling itself and its environment"
  [V]; the learning procedure itself is fixed code [I].
- P6 **–**: Replicode programs; in the demonstration, words came from a commercial speech recogniser
  "filtered through a set of 100 allowed words" [V].

**15. Analogical concept memory for Soar (Aileen).** "When Aileen identifies a new concept in
linguistic content (word red), it creates a new symbol" and a "generalization context" for it;
examples are aligned by structure mapping and merged into a generalisation that "maintains a
probability that a fact belongs in the generalization", the fraction of examples containing it. [V]
"Only facts exceeding" 0.6 "are considered part of the concept". [V]
- P1 **P**: incremental and online, but "Aileen learning is deliberate ... It learns only when the
  current knowledge isn't applicable" [V].
- P2 **P**: fact probabilities are graded and updated per example [V]; the cut-off turns them into
  in/out [V; I].
- P3 **P**, the nearest thing to P3 in the field: a concept is opened at the moment an unknown word
  is met [V]; the percept symbols it is built from come from a separate vision module, and the
  teacher supplies the word [V].
- P4 **P**: it "evaluate[s] the quality of concept definitions" [V].
- P5 **–**. P6 **–**: predicate calculus; "The memory has not be[en] integrated into the
  architecture but is a separate module" [V].

**16. Rosie (interactive task learning in Soar).** A Soar agent learns new tasks from a human's
restricted-English instructions while acting; task knowledge is stored declaratively and compiled
into rules by chunking. [V; A] "An intelligent learner contributes to its own learning by asking
questions", and "metacognitive analysis performed during impasse resolution in Soar can inform" when
to ask, what to ask, and how to integrate the reply. [V]
- P1 **P**, P2 **–**, P3 **P** (new words and tasks, by instruction) [V; I].
- P4 **P**: the impasse-driven question is this field's nearest thing to "the question arises by
  itself"; it arises from being unable to proceed in a task [V; I].
- P5 **P**: learned tasks are reused as parts of new ones [V]. P6 **–** [V].

**17. Copycat and Metacat.** Many small agents (codelets) build and break perceptual structures in a
workspace, guided by a network of concepts (the slipnet) whose activations shift with context; a
global "temperature" measures how well-organised the structures built so far are and sets how random
the next steps will be. [M; A] Metacat adds a trace of its own run: it "records only events that
exceed an (unspecified) importance threshold", chiefly main ideas and "snags", and can notice it is
repeating itself. [V, arXiv:2503.13467 and arXiv:1610.08602]
- P1 **–**: nothing carries over between runs in Copycat [M].
- P2 **P**, P3 **P**: graded, context-dependent concepts within a run; the concept network is
  hand-built [M].
- P4 **P**, and the nearest mechanism in the field: a number the system computes about its own
  constructions decides how it goes on [M; I]. The formula is hand-designed, the domain is
  letter strings, it measures coherence not reality, and it does not improve with experience [M; I].
  The Metacat paper itself calls temperature "a very crude form of self-watching" [A].
- P5 **–**. P6 **–**.

**18. CogNGen.** The Common Model's modules are built from predictive-coding circuits with local
error-driven plasticity, a MINERVA 2 / Hopfield memory with Hebbian storage for working and
declarative memory, and a basal-ganglia-like circuit for action; the whole "can be written down as
optimizing, in a modular fashion, a single free energy function". [V] "To avoid the neural and
behavioural pitfalls of backprop, we only use Hebbian learning in CogNGen." [V]
- P1 **N**: "at the heart of CogNGen is the story of online learning and adaptation" [V].
- P2 **–**, P3 **–**: "At present, the contents of the MINERVA 2 models are hardwired to be histories
  of recent observations"; a semantic memory is future work [V].
- P4 **P†**. P5 **–**.
- P6 **N** for the engine; tested on five grid-world maze tasks; "CogNGen, as implemented, is only a
  prototype" [V]. No language [V].

**19. Shared global workspace in deep networks.** Specialist modules (Transformer positions or
recurrent units) "compete for write access" to a small shared memory by key-query attention with a
soft-max or top-k soft-max, and the memory's contents are broadcast back to all. [V] Trained end to
end. [V]
- P1 **–**, P2 **–**, P3 **–**, P4 **–**, P5 **–**, P6 **N**. A workspace as a trained bottleneck,
  nothing more [I].

**20. Conscious Turing Machine.** A formal model: many long-term-memory processors submit "chunks" to
a competition, one winner reaches the single-slot short-term memory and is broadcast to all. [V]
"Each LTM processor runs a number of algorithms, one of which is the processor's personal Sleeping
Experts algorithm", which adjusts its weights from the feedback of broadcasts. [V]
- P1 **N** on paper, P2 **P** (weights as reliabilities of processors), P3 **–**, P4 **P**, P5 **–**,
  P6 **–**: a definition, not a running system [V; I].

**21. Consciousness-in-AI report: indicator properties and perceptual reality monitoring.** Derives
14 indicator properties from scientific theories; for the workspace theory: parallel modules, a
limited-capacity workspace, global broadcast, state-dependent attention. [V] For higher-order
theories: "HOT-2: Metacognitive monitoring distinguishing reliable perceptual representations from
noise" and "HOT-3: Agency guided by a general belief-formation and action selection system, and a
strong disposition to update beliefs in accordance with the outputs of metacognitive monitoring". [V]
- Not a system. P4 **N*** as a specification: "a 'reality monitoring' mechanism, which operates
  automatically, is used to discriminate between these different kinds of activity" (stimulus-driven,
  sustained, generated top-down "through expectations, imagination, dreaming or episodic memory", or
  noise) "and assess the reliability of first-order representations". [V]
- Status of the idea: "PRM researchers claim that there are no current AI systems that meet all of
  the requirements ..., and there have been no implementation attempts that we are aware of". [V]
- Sketches offered: a second-order network that outputs "a single scalar, representing the
  probability that the first-order representation of that layer is veridical", trained by
  supervision, by predictability, or as a GAN discriminator. [V]

**22. Cognitive reality monitoring network.** A proposal: many paired generative and inverse models
run in parallel; a prefrontal network computes for each pair a "responsibility signal" from "the
mismatch between computations by generative and inverse models, as well as reward prediction
errors", through "a soft max". [V] The signal "gates selection and learning of pairs"; awareness is
tied to low entropy of the signals, and "minimization of responsibility-signal entropy" is a driving
force of learning. [V]
- P1 **P**, P2 **P** (graded responsibilities) [I]. P3 **–**.
- P4 **N***: a measure of how well each of the system's own models fits is computed and steers both
  use and learning [V; I]. The authors raise the supervision problem of the GAN reading ("it is
  almost identical to assuming the homunculus") and replace it by reward and entropy [V].
- P5 **P**: aimed at learning "from small samples" by choosing the right abstraction [V].
- P6 **P**: stated for neural networks; "we plan to computationally simulate a simplified CRMN" [V],
  i.e. not done in the paper.

**23. CoALA.** A language agent is described as an LLM plus working memory and long-term episodic,
semantic and procedural memories, with internal actions (retrieval, reasoning, learning) and
external ones, in a repeated decision cycle. [V] "Learning occurs by writing information to
long-term memory"; updating the LLM's weights is one option, but "Fine-tuning the agent's LLM is a
costly form of learning; thus, present studies specify learning schedules". [V]
- P1 **P**: stores change during operation, weights on a schedule or never [V].
- P2 **–**: stored items are text [V; I]; "modifying and deleting ... are understudied" [V].
- P3 **P**: new items and skills are written at will; the model's embedding table is fixed [I].
- P4 **P**: the LLM reasons over its own outputs; no measure [I].
- P5 **P**: prompts, code and skills can be updated; learning new procedures for learning: "we are
  not aware of any language agents that implement this form of learning" [V].
- P6 **–**: the engine is a pre-trained Transformer plus hand-written control code [I].

**24. Generative Agents.** Every observation is appended to a "memory stream" in natural language;
retrieval scores each record by recency (exponential decay), importance (a 1–10 rating by the LLM)
and relevance (embedding cosine). [V] When "the sum of the importance scores for the latest events
... exceeds a threshold (150 in our implementation)", the agent asks the LLM for "3 most salient
high-level questions we can answer", retrieves for each, asks for insights with citations, and
stores them as "reflections", which can be reflected on again. [V]
- P1 **P**, P2 **–** [I].
- P3 **P**: higher-level statements are induced from instances, with pointers to their evidence [V].
- P4 **P**: reflection on its own records, by prompt [V]. Reported failures: failed retrieval and
  "hallucinated embellishments" [V].
- P5 **–**. P6 **–**.
- The questions here are generated on a counter, and are questions the memory can already answer
  [V; I].

**25. Mechanism audit of an LLM-based "cognitive architecture".** A static audit with call-chain and
value-to-output tracing of an open repository that couples 25 named cognitive modules (global
workspace, theory of mind, causal reasoning and others) to an LLM kernel. [V] "In 18 of 25 modules
the computed value never reaches any production output"; three patterns are named:
connected-but-never-called, called-but-inert, called-but-content-blind. [V]
- Not a system; evidence about a failure mode. The authors "make no statistical sampling claim". [V]

### 2.2 Matrix (all cells [I])

| # | Work | P1 | P2 | P3 | P4 | P5 | P6 |
|---|---|---|---|---|---|---|---|
| 1 | Soar | P | – | P | P | P | – |
| 2 | ACT-R (metadata) | N | P | – | P | P | P |
| 3 | IBL / blending | N | P | – | – | – | P |
| 4 | Sigma | N | P | – | P | – | N (no language) |
| 5 | CLARION | N | P | P | P | P | P |
| 6 | LIDA | N (design) | P | P | P | – | – |
| 7 | HTM | N | P | P | P† | – | P |
| 8 | Monty | P | P | P | P† | – | – |
| 9 | Thousand Brains (plan) | P | P | P | P† | – | – |
| 10 | Kanerva vector architecture | N (paper) | P | P | – | – | N (no language) |
| 11 | Leabra family | P | P | P | P† | – | N (no language) |
| 12 | Spaun | – | – | – | – | – | N (no language) |
| 13 | Common Model + metacognition | P | P | – | P | P | – |
| 14 | AERA | N | P (3 of 4) | P | P | P | – |
| 15 | Aileen concept memory | P | P | P | P | – | – |
| 16 | Rosie | P | – | P | P | P | – |
| 17 | Copycat / Metacat | – | P | P | P | – | – |
| 18 | CogNGen | N | – | – | P† | – | N (no language) |
| 19 | Shared global workspace | – | – | – | – | – | N |
| 20 | Conscious Turing Machine | N (paper) | P | – | P | – | – |
| 21 | Reality monitoring (report) | – | – | – | N* | – | P |
| 22 | Cognitive reality monitoring network | P | P | – | N* | P | P |
| 23 | CoALA | P | – | P | P | P | – |
| 24 | Generative Agents | P | – | P | P | – | – |
| 25 | Audit | not a system | | | | | |

What the matrix says [I]:
- No column other than P1 and P6 contains an N for a running system.
- No row has N in both P1 and P6 together with more than "–" in P2 and P3. Sigma comes nearest
  (N, P, –, P, –, N), then CogNGen (N, –, –, P†, –, N).
- The rows with the most P's in P2–P5 (CLARION, AERA, Soar with its extensions) all have "–" or P in
  P6. The engine rows (10, 11, 12, 18, 19) have almost nothing in P2–P5. This is the same split the
  40-year review reports.

---

## 3. Supporting works (read or located, not in the main table)

| Work | What it adds | Status |
|---|---|---|
| Kotseruba & Tsotsos, "A Review of 40 Years of Cognitive Architecture Research", arXiv:1610.08602 (AI Review 2020) | Field statistics and the statements on reasoning, metacognition, language and scale quoted in sections 1 and 5 | V in part |
| Nolte et al., "How Metacognitive Architectures Remember Their Own Thoughts: A Systematic Review", arXiv:2503.13467 (2025) | 101 reports on 35 metacognitive architectures; "only 17% of CMAs were quantitatively evaluated"; 56% symbolic, 22% hybrid, one (3%) purely sub-symbolic | V in part |
| Wray, Kirk, Laird, "Applying Cognitive Design Patterns to General LLM Agents", arXiv:2505.07087 (AGI-25) | Two patterns "have not yet been taken up in the Agentic LLM mainstream: reconsideration and knowledge compilation" | V in part |
| Laird, Wray, Jones, Kirk, Lindes, "Proposal for Cognitive Architecture and Transformer Integration: Online Learning from Agent Experience", AAAI Symposium Series 2024 | A Transformer "trained online in real-time using an agent's ongoing experiences" as a new long-term memory of Soar; a proposal that lists challenges | A (host not reachable) |
| Sun, "Can A Cognitive Architecture Fundamentally Enhance LLMs? Or Vice Versa?", arXiv:2401.10444 | CLARION's structure; LLMs as its implicit level; cognitive architectures "need an overhaul" | V |
| Poovendran, Stocco, Bushnell, "AIBL: Augmented Instance-Based Learning with Structured Memory and Neural Embeddings", arXiv:2610.03413 (2026) | Instance memory split into active, forgotten and surprise stores: a weakly supported observation "is stored in surprise memory and withheld from ordinary blending. If similar observations recur, the graduation rule can promote them". Embeddings come from pre-trained encoders (Sentence-BERT, autoencoders) | V in part |
| Kelly, Arora, West, Reitter, "Holographic Declarative Memory" (Cognitive Science 2020); Ray & Dancy, arXiv:2508.15630 (2025) | A vector memory replacing ACT-R's declarative memory, bringing "scalability and architecturally defined similarity between DM chunks"; vectors accumulate as chunks are added | A |
| Bricken et al., "Sparse Distributed Memory is a Continual Learner", arXiv:2303.11934 (ICLR 2023) | An MLP derived from SDM that resists forgetting; main results use "256 dimensional latent embeddings taken from the last layer of a frozen ConvMixer that was pre-trained on ImageNet32" | V in part (see also `art-growth.md`) |
| VanRullen & Kanai, arXiv:2012.10390; Chateau-Laurent & VanRullen, arXiv:2503.01906; Maytié et al., arXiv:2502.21142 | Global latent workspace by translation between latent spaces; routing through a workspace to chain operations; "dreaming" inside the workspace latent space for RL | A |
| Liu et al., arXiv:2305.17375; Farrell, Ziman, Graziano, arXiv:2411.00983; Premakumar et al., arXiv:2407.10188 | Attention schema (a model of the agent's own attention) in neural agents; a network trained to predict its own internal states becomes simpler | A |
| Ranjan et al., arXiv:2607.23927; Khezresmaeilzadeh et al., arXiv:2611.09014 (2026) | Reality monitoring tested in LLMs and vision-language models: source attribution is fragile; injected concepts "are reported as seen" | A |
| Du, arXiv:2603.07670; Lam et al., arXiv:2603.11768 (2026) | "The central risk of reflective memory is self-reinforcing error"; "semantic drift" through repeated summarising; "Nobody evaluates forgetting well" | V (the quoted lines) |
| Mohan, Kirk, Laird / Kirk et al., arXiv:2208.09554 | Online one-shot task learning in Soar drawing on instruction, search and GPT-3 | A |
| Ororbia & Kelly, arXiv:2204.00619 | First CogNGen: predictive processing plus hyperdimensional memory, four maze tasks | A |
| Struye, Mets, Latré, "HTMRL", arXiv:2009.08880 | HTM-only reinforcement learner; adapts to a shuffled bandit in a third of the original steps | A |
| Dzhivelikian & Panov, arXiv:2510.03286 (2025) | Episodic memories structured into cognitive maps online with local Hebbian-like rules, grid world | A |
| Su, arXiv:2602.11516 (2026) | "Learning how to think": internal reasoning trajectories as learning material; one sensor-anomaly task | A |
| Shang, arXiv:2604.08206 (2026); Hu & Ying, arXiv:2503.03459 (2025); Goldstein & Kirk-Giannini, arXiv:2410.11407 | Global-workspace architectures built from LLM agents; an argument that language agents may already satisfy workspace theory | A |
| Romero et al., arXiv:2308.09830; West et al., arXiv:2403.18827; Wu et al. (LLM-ACTR), arXiv:2408.09176 | Three ways of joining LLMs to cognitive architectures | A |
| Rosenbloom, Laird, Lebiere, Stocco, arXiv:2506.12224 | Four neural theories of consciousness mapped onto the Common Model | A |
| Chella, Frixione, Lieto, arXiv:1804.08299 | Representational limits of declarative memories in Soar, ACT-R and Sigma; conceptual spaces proposed | A |
| Malloy et al., arXiv:2610.00399 (2026) | Instance-based learning used to set the compute of an energy-based model from its uncertainty | A |
| Choo, "Spaun 2.0", PhD thesis, Waterloo 2018 | 6.6 million neurons, 12 tasks, instruction following, motor adaptation | A |
| Dijkstra & Fleming and colleagues, reality-monitoring framework (preprint 2025) and Neuron 2025 | Reality judgments as higher-order inference on sensory precision; imagery and perception share a signal compared with a "reality threshold" | A |
| "The Art of Not Forgetting" (Cognitive Memory Primitive), arXiv:2607.17944 / 2607.26523 (2026) | Gradient-free local learning with sparse relational codes; not a cognitive agent | A (see `online-plasticity.md`) |
| Huang, "Reimplementation and Reinterpretation of the Copycat Project", arXiv:1811.04747 | Workspace, coderack, slipnet; "these architectures are often complex and include handcrafted knowledge" | A plus a few lines |

Not located on any reachable host, and therefore only [M]: ICARUS (Langley), the Companions
architecture and SAGE in their own papers (Forbus), DUAL/AMBR (Kokinov), MIDCA (Cox), OpenCog
CogPrime (Goertzel; its logic is in `nars-logic.md`).

---

## 4. Reading by principle

### 4.1 What learns online (P1)

- Numbers on fixed structures, at every step: ACT-R activation and utility [V], Soar's numeric
  preferences [V], Sigma's long-term functions [V], AERA's reliabilities [V], CLARION's two levels
  [V], HTM permanences [V].
- New structures, at every step: Soar episodes (one per decision) [V]; ACT-R chunks (on buffer
  clearing) [V]; IBL instances [V]; AERA models, on a surprise [V]; Monty points [V]; Generative
  Agents records [V].
- New structures, only on a trigger: Soar chunks (a result in a substate) [V]; CLARION rules (a
  success) [V]; Aileen examples (a comprehension failure) [V]; reflections (importance sum above
  150) [V].
- Not online: Spaun [M], shared workspace [V], Leabra models in practice [A; M], the LLM in LLM
  agents [V].
- Two stabilisers that the field added after trouble, both of the "weak first, strong later" kind:
  ACT-R gives a newly compiled rule low utility so that "With enough repetitions, the new rule will
  be selected" [V]; in Soar "we have found it necessary to modify the architecture so that learned
  rules must be learned multiple times before they are used (Stearns, 2021)" [V]. Soar's author also
  plans to admit chunks from uncertain decisions only "when there is sufficient accumulated
  experience to ensure that they have a high probability of being correct" [V]. These are the
  field's own steps toward P2(i), applied to rules. [I]

### 4.2 What holds graded beliefs (P2)

| Part of P2 | Where it exists in this field | Tag |
|---|---|---|
| (i) weak first storage | HTM (0.21 below threshold 0.5); AERA (experience factor 1/2); ACT-R compiled rules (low utility); AIBL (surprise store); LIDA (decaying new nodes) | V, V, V, V, M |
| (ii) refine, not delete | AERA (phase-out, pre-conditions); ACT-R (all instances kept, blended); Sigma (distribution shifts); HTM (−0.01); SAGE (probabilities) | V, V, I, V, V |
| (iii) rule induced from instances | AERA (abstraction); CLARION (generalisation by information gain); SAGE (generalisation); Generative Agents (reflection by LLM) | V, V, V, V |
| (iv) rule feeds support back to the instance | Nowhere as a stored change. At use time only: AERA (likelihood through the general model), ACT-R (partial matching lets other instances weigh in), CLARION (top-down activation) | I; C; C; V |

- Deletion on contradiction still exists in the systems that have (iii): AERA below the second
  threshold [V], CLARION when a failing rule has no earlier version [V]. AERA's authors note the
  cost: "A model that predicts correctly in some cases can still be discarded because it predicts
  incorrectly in some other cases." [V]
- In none of these systems can the system's own reasoning read the number, except AERA (reliability
  feeds likelihoods and scheduling) and CLARION's metacognitive subsystem. For ACT-R and Soar this is
  the "wall", and Laird asks: "But is this strict separation necessary?" [V]

### 4.3 What forms concepts (P3)

- At the moment of need: Aileen opens a concept on an unknown word [V]; Monty opens an object model
  on an unknown object [V]; AERA builds composite states from co-occurring facts [V]; HTM grows a
  segment when nothing matches [V].
- From accumulated instances: Leabra's visual categories [A]; SAGE generalisations [V]; reflections
  [V].
- Given by a person: ACT-R chunk types and similarities [M; V], Sigma predicates [V], Copycat's
  slipnet [M], CLARION's dimensions and values [V; I], Spaun's vocabularies [M], the random seed
  vectors of vector architectures (no content) [V], the embedding table of the LLM [I].
- Stated as missing by the field itself: "semantic learning, generalized perceptual category
  learning" in Soar [V]; abstract concepts in the Thousand Brains Project [V; A]; a semantic memory
  in CogNGen [V].

### 4.4 What monitors itself (P4)

| Kind of self-monitoring | Systems | What is measured | Tag |
|---|---|---|---|
| Automatic detection of a gap in own knowledge | Soar impasse; Rosie | whether a decision can be made | V |
| Module status | ACT-R, Soar, Common Model proposal | retrieval succeeded or failed, module busy | V |
| A separate monitoring subsystem | CLARION, MIDCA | drive satisfaction, performance, confidence | V (MIDCA: as named by Laird) |
| Prediction against outcome | AERA monitors, HTM, Monty, Leabra, CogNGen | prediction error | V, V, V, A, V |
| A trace of own run | Metacat, Soar episodic memory, AERA internal inputs | events, snags, successes and failures | V |
| A number about the quality of own constructions | Copycat temperature | structural coherence, within one run | M, A |
| Appraisals | Sigma | surprise, familiarity, curiosity | V |
| Reflection by prompt | Generative Agents, CoALA agents | nothing numeric | V |
| Realness of own representations | reality monitoring (report), cognitive reality monitoring network | probability that a representation is veridical; responsibility | V, on paper only |

- Laird lists what ACT-R and Soar do not have: "Memory and processing appraisals. Humans can judge
  many properties of their memories and processing, such as familiarity in a belief, feelings about
  ease of processing; feelings of knowing". [V]
- AERA's authors about Soar: "truth being axiomatic, and the system's operation not being considered
  a target for reasoning". [V]

### 4.5 What improves its own way of working (P5)

- Compiling one's own reasoning into skill: Soar chunking, ACT-R production compilation [V]. What
  improves is speed and the skill at a task; the compiler is fixed. [I]
- Learning about one's own reasoning and storing it: the Common Model proposal gives the example of a
  robot that reviews its episodes, sees a wasted action, and "learns to inhibit that action in
  similar situations in the future" [V].
- Self-modelling for control: AERA, implemented, unused in its demonstration [V].
- In LLM agents: updating prompts and code skills exists; updating the procedures of learning or
  decision making does not [V].
- Transfer across domains of a learned way of constructing: I found no demonstration in any of the
  25 works. [I]

### 4.6 Engine and language (P6)

- Engines of vectors and equations in this field: Sigma, the vector architectures, Leabra, Spaun,
  CogNGen, the shared workspace. [I]
- Their language record: Sigma small [M]; Kanerva, language identification by letter statistics [V];
  Leabra and Spaun, small models [M]; CogNGen none [V]; shared workspace none [V].
- The symbolic systems' language record: restricted English through hand-built comprehension
  (Soar/Rosie) [V]; a 100-word filtered recogniser (AERA) [V]; off-the-shelf software in most
  architectures ("More commonly, off-the-shelf software is used for speech recognition and parsing")
  [V].

---

## 5. Known failure modes: what broke

### 5.1 The knowledge has to be put in by hand

- "agents do not just learn to do new tasks from scratch without any of the pre-existing procedural
  and semantic knowledge" [V, Soar].
- "What I feel is most missing from Soar is its ability to 'bootstrap' itself up from the
  architecture and a set of innate knowledge into being a fully capable agent across a breadth of
  tasks. Our agents do well when restricted to specific well-defined tasks." [V]
- A 2025 paper on generating Soar rules with LLMs gives manual rule coding as what holds back Soar's
  adoption. [A, arXiv:2510.09355 via search]
- The 40-year review: demonstrated interactions "are heavily scripted". [V]

### 5.2 Stability

- **Learned rules that are wrong or too general.** Soar's chunking was "completely reimplemented" so
  that rules are "correct relative to the reasoning in the substate and as general as possible
  without being over general" [V]; it still "requires that substate decisions be deterministic" [V].
  Both ACT-R and Soar had to delay the use of newly learned rules (section 4.1) [V].
- **Rigidity of counted evidence.** AERA: "The more a model has accumulated experience, the more it
  needs to accumulate future contrary experience to switch modes", which the authors call "undesired
  inertia"; their fix is to compute reliability over "a reference fixed number of evidences" [V].
  Any belief stored as a count has this problem. [I]
- **Discarding a partly right model.** AERA, quoted in 4.2 [V].
- **Thin margin at a threshold.** In HTM a connected synapse sits just above 0.5; one or two wrong
  predictions disconnect it [C from V parameters].
- **Sensitivity to noise in time.** HTM "can lose high-order sequence context if elements in the
  sequence are replaced by a random symbol" [V].
- **Self-reinforcing error.** In reflective LLM memory: an agent that wrongly concludes a call
  always fails will avoid it "forever, never collecting evidence to overturn the false belief" [V].
  This is exactly the hazard of P2(iv) when a rule built from an instance is allowed to raise that
  instance. [I] (NARS forbids the loop by design; see `nars-logic.md`.)
- **Drift.** Agents "may gradually distort facts through repeated summarization (semantic drift)"
  [V]; Generative Agents "hallucinated embellishments to their knowledge" [V].

### 5.3 Forgetting and growth

- ACT-R forgets by losing access, not by deleting [V]. IBL memory grows with every instance, and
  computing time with it [V].
- Soar's episodic memory "does grow over time, and the cost to retrieve old episodes slowly
  increases" [V].
- Monty "continuously expands its representational capacity when learning", and inference cost grows
  linearly with the number of models [V].
- Episodes are usually "time-stamped snapshots", an approach "suitable for agents with a short
  life-span"; "more elaborate solutions will be needed for life-long learning and managing
  large-scale memory storage" [V, 40-year review].
- CogNGen's goal "to learn different tasks consecutively without catastrophic forgetting" is stated
  as future work; "The main obstacle in scaling CogNGen to larger tasks ... is memory architecture"
  [V].
- Leabra-style cortical learning needs interleaved exposure, which is why that school pairs it with
  a fast hippocampal system [M].

### 5.4 Scale and evaluation

- "time and space complexity is glossed over in the literature"; scale problems "are currently
  addressed by only a handful of architectures" [V].
- Sizes of the neural demonstrations: 77 household objects (Monty) [V]; five grid-world tasks
  (CogNGen) [V]; 8 tasks with 2.5 million neurons (Spaun) [V]; "we have only tested HTM on
  low-dimensional categorical or scalar data streams" [V].
- Metacognition: "only 17% of CMAs were quantitatively evaluated" on the use of their own traces;
  the vocabulary "is highly fragmented" [V].
- ACT-R "scales poorly to big data applications and lacks a robust model for learning association
  strengths between stimuli" [A, from the holographic-memory papers].

### 5.5 Language

- Soar, rated by its author: "Use language, both natural and artificial (partial, no) ... this is not
  a capability that is available to any agent developed in Soar, and even Rosie cannot process
  unrestricted language." [V]
- The 40-year review: "most existing NLP systems are limited both in their domain of application and
  in terms of the syntactic structures they can understand." [V]
- HTM: weaker than LSTM on an artificial grammar, by memorising sentences [V]. Thousand Brains:
  language is on the list of things to come [V]; the project's forum says no idea about language has
  been implemented or tested [A].
- The response since 2023 is to attach an LLM: as CLARION's implicit level [V], as a knowledge
  source or a memory for Soar [A], as an adapter trained on ACT-R traces [A]. Each of these gives up
  P1 and P3 for the language part. [I]
- No system in this field acquires language through its own online learning mechanism. [I]

### 5.6 The numbers are not beliefs, and the system cannot see them

- The "wall" [V]. "The meaning of metadata is thus fixed by the architecture" [V].
- Activation answers "will I recall this, and how fast"; utility answers "did this pay". Neither
  answers "how far is this true". [I]
- The symbolic content is all-or-none: AERA's authors on Soar, "truth being axiomatic" [V].

### 5.7 One winner at every central step

- Soar selects "a single operator" [V]; the Common Model "selects a single action" per cycle [V];
  Sigma takes the operator with the highest value [V]; ACT-R retrieval is a Boltzmann soft-max and
  IBL choice an arg-max [V]; the Conscious Turing Machine admits one chunk [V]; the shared workspace
  uses soft-max or top-k [V]; the reality monitoring network uses "a soft max" [V]; "WTA and its
  variants are common in the emergent architectures such as HTM, ART, SPA" [V].
- The field treats the bottleneck as a principle (one content at a time, broadcast to all). A design
  that leaves soft-max and arg-max departs from the whole field here, and must say what replaces
  the bottleneck's function: keeping all modules on one content. [I]

### 5.8 Names without computation (LLM era)

- In the audited system "A name can be attached to a keyword counter, a constant, or to code that is
  never called"; 18 of 25 modules had no effect on any output; two repairs raised a planner's task
  success "from 0.028 to 1.000" [V].
- Two earlier audits by the same line of work are cited: "emergence" markers that did not survive a
  random-autonomy control, and a modulator in which "a single einsum-level defect" made the whole
  mechanism inert [V, as cited in the paper].
- The method (for each module: is a value computed, and does it reach the output) applies to any
  design with named cognitive parts. [I]

---

## 6. The closest existing system on the "red" example

Owner's example: "red is dark" 0.23 → "dark but sometimes light" 0.57 → "all colours are sometimes
dark, sometimes light" 0.77 → red 0.88 → the question "which other colours exist?" arises by itself.

### 6.1 Which system, and why

- **Closest in this field: AERA.** It is the only running architecture here in which a graded number
  is attached to a piece of knowledge from the first observation, is lowered by contrary evidence
  without removal, a more general model is built from instances during operation, and that model
  then yields a better-supported prediction for the instance. [V; I]
- **Strongest that comes near: ACT-R's declarative memory with blending.** It is the most widely
  validated mechanism in the field [M] and its equations were read in a primary source, so its
  numbers below are exact for the stated parameters. [V; C]
- CLARION may be as close as AERA on paper; I could not read its technical sources, so its trace is
  given last and marked [M].
- For the same example in evidential logic see `nars-logic.md`; AERA's reliability is the product of
  two factors, success rate e+/e and experience e/(e+1), which are the same two quantities as NARS
  frequency and confidence with horizon 1. [C]

Event stream used for all traces (the same as in `art-growth.md`) [C]: E1 hear "red is dark" once;
E2 red seen light once; E3 red seen dark again; E4 blue dark once, light once; E5 green dark once,
light twice; E6 yellow dark twice, light once. "Purple" is never observed. All numbers come from
`tools-cognitive-architectures/red_example_cogarch.py`, which applies the published equations by
hand; no author code was run.

### 6.2 What a person must supply before the first sentence

- AERA: a seed with the sensor ontology, "a few drives" and "a few initial models" [V]. Sentences are
  not input; "red is dark" must arrive as observed facts about an entity (its colour, its
  brightness), or the seed must hold a model that turns the heard words into such facts [I].
- ACT-R: a chunk type with slots (colour, brightness), the productions that store and retrieve, the
  similarity values between colours, and the parameters d, s and the mismatch penalty [M; V].
- Neither system takes the sentence itself. [I]

### 6.3 AERA, step by step

| Stage | What AERA does | Number | Tags |
|---|---|---|---|
| E1 | A model is built: x is red → x is dark. Reliability = e+/(e+1). | 1/2 = **0.50** (success rate 1.0 × experience 0.5) | V formula, C |
| E2 (a light red) | The prediction fails. Reliability drops. If it is under the first threshold the model is phased out: it makes only "silent predictions", which "are still monitored, thus giving the possibility to improve"; under the second threshold it is deleted. The failure also creates a pre-condition model recording the failing case. | 1/3 = **0.33** | V, C |
| E3 (red dark again) | Reliability recovers; a phased-out model returns to duty above the first threshold. | 2/4 = **0.50** | V, C |
| "dark but sometimes light" | No single model says this unless the value is abstracted. A model x is red → x is dark or light would never fail. | 3/4 = **0.75** | I, C |
| E4–E6 (other colours) | When a new success differs from an earlier one in one value, that value is replaced by a variable and a general model is created; "their fate will depend on subsequent evidences". | any colour → dark: 6/12 = **0.50**; any colour → dark or light: 11/12 = **0.92** | V, C |
| Back to red | A prediction's likelihood is "the product of the reliability of the models that were involved". Through the general model a prediction about red carries 0.92, through red's own model 0.75. | 0.75 → **0.92** at prediction time | V formula, C |
| Afterwards | "Shall the particular cases ... turn out to be actually anecdotal ... they will be dismissed ... and the general case ... will prevail". The 0.92 is not stored on red; red's model tends to disappear. | — | V |
| Question | None. New models are built when "the unpredicted success of a goal or the failure of a prediction" occurs. Nothing asks which other values the variable can take. | — | V, I |

- Thresholds: the report names them THR1 and THR2 and gives no values in the passages read. [V]
- Evidence-count sequence n/(n+1): 0.50, 0.67, 0.75, 0.80, 0.83, 0.86, 0.88, 0.89. The owner's 0.57,
  0.77, 0.88 follow the same rising shape but are not these values. [C]

### 6.4 ACT-R with blending, step by step

Parameters as read: decay d = 0.5, noise s = 0.25, temperature τ = s·√2 = 0.354; noise left out;
similarity 0 for the same colour and −1 for another colour; mismatch penalty α as shown. The cue is
"colour = red"; the number is the blended value of "dark". [V equations and defaults; C]

| Asked after | exact match | α = 1.0 | α = 2.5 | Highest activation |
|---|---|---|---|---|
| E1 | 1.000 | 1.000 | 1.000 | 0.00 |
| E2 | 0.273 | 0.273 | 0.273 | 0.00 |
| E3 | 0.906 | 0.906 | 0.906 | +0.46 |
| E4 | 0.884 | 0.845 | 0.883 | +0.02 |
| E5 | 0.879 | 0.586 | 0.872 | −0.27 |
| E6 | 0.878 | 0.727 | 0.874 | −0.45 |
| 100 unrelated steps later | 0.877 | 0.830 | 0.876 | −1.66 |
| purple, after E6 | no instance matches: retrieval failure | 0.554 | 0.554 | −0.75 / −2.25 |

What the table shows [C; I]:
- **One hearing is not weak.** After E1 the answer is 1.000: one instance is recalled as stored. What
  is weak is its activation: 0.00 at age 1, −1.15 at age 10, −2.30 at age 100. With a retrieval
  threshold the instance is soon not recalled at all. [C; threshold default M]
- **Contradiction refines.** After E2 both instances exist and the answer is 0.273 (the newer one
  weighs more). Nothing was deleted.
- **The blend is sharp, not a frequency.** Two dark and one light give 0.906, not 0.667, because the
  low temperature raises activation ratios to the power 1/τ ≈ 2.8. At long delays the weights tend
  to (number of occurrences)^2.8, here 7.1 : 1, hence 0.877.
- **No rule.** Nothing in memory says anything about colours in general. With partial matching
  (α = 1.0) recent instances of other colours pull red's answer about (0.586 after two light greens).
  That is influence from the category on the instance, but it follows recency, and the amount is the
  modeller's number.
- **A value for an unseen colour** exists only with partial matching: 0.554 for purple, the
  recency-weighted share of dark among all colour instances.
- **Nothing ever rises through support.** The blend is an average of stored outcomes; "dark but
  sometimes light" cannot become a statement held at 0.57, 0.77 or 0.88, because there is no such
  statement, only the average.
- **No question.** A failed retrieval is reported as module status; what follows is whatever
  production the modeller wrote. [V; M]

### 6.5 Four other systems on the same stream, briefly

- **Soar with the analogical concept memory.** One example is stored as it is. From two examples on,
  the concept "red" holds fact probabilities: after E2 dark 0.50, light 0.50; after E3 dark 0.67,
  light 0.33. With the published cut-off 0.6, after E2 neither fact belongs to the concept, after E3
  only "dark" does; "sometimes" has no place. [V mechanism; C] The agent stores an example only when
  comprehension fails, so the confirming E3 may not be stored at all. [V; I] A concept "colour"
  exists only if a teacher names it; with all 11 examples it would hold dark 0.55, light 0.45, both
  under the cut-off. [C] When it cannot ground a word, the agent signals failure to the instructor
  or asks. [V] That question is "what does this word mean?", raised by a failure, not "which other
  colours exist?". [I]
- **HTM.** E1 creates a synapse group red → dark at 0.21: nothing is predicted. E2 creates red →
  light at 0.21. E3 raises red → dark to 0.31. It takes two more dark reds to reach 0.51 and predict
  "dark"; a light red then costs 0.01, and a second one drops it to 0.49: no prediction again.
  [C from V parameters] Both outcomes can be predicted together once both groups are connected,
  which is the nearest native form of "sometimes dark, sometimes light" in the field. [V; I] No
  rule, no feedback, no question. [I]
- **CLARION** [M for the formula, V for the refinement rules]. The implicit level moves a little
  with each case. If the first case counts as a success, an explicit rule red → dark is extracted.
  At E2 the rule fails; it has no earlier version to return to, so it is deleted. [V rules; I
  mapping] On this stream the information gain of the rule against the match-all rule stays between
  0.00 and +0.26, i.e. red predicts darkness no better than anything else, so nothing is generalised.
  [C with an M formula] The category statement of the owner's example ("sometimes dark, sometimes
  light") is not a rule of the kind this mechanism extracts. [I]
- **Generative Agents / CoALA agents.** Each sentence is stored as text with an importance score.
  After enough importance has accumulated, the LLM is asked for questions and insights and may well
  write "colours can be dark or light" with pointers to the records it used. [V procedure; I] No
  number of belief exists for any record, a contradiction is just another record, and the LLM's own
  weights do not change. [V; I]

### 6.6 The five steps of the example against the field

| Step of the owner's example | AERA | ACT-R blending | Soar + concept memory | HTM | CLARION | LLM agents |
|---|---|---|---|---|---|---|
| First hearing stored weakly (0.23) | yes, 0.50 | no: value 1.0, weak access | no | yes, 0.21 under 0.5 | implicit level only | no |
| Contradiction refines (0.57) | yes: 0.33, phased out, pre-condition | yes: 0.27, both kept | partly: 0.5/0.5, both under cut-off | yes: second group | explicit rule deleted | another text record |
| Category rule induced (0.77) | yes, by abstraction: 0.92 | no | only if a teacher names "colour" | no | not on this stream | by LLM prompt |
| Rule raises the instance (0.88) | at prediction time only (0.75 → 0.92), not stored | no | no | no | top-down activation at use time | no |
| Question arises by itself | no | no | on comprehension failure | no | no | on a counter |

All cells [I], resting on the [V] and [C] statements above.

---

## 7. What is still unsolved

### 7.1 In the field's own words

1. **Bootstrapping.** "without a human to closely guide it, Rosie is unable to get very far on its
   own, especially in being unable to learn new abstract symbolic concepts on its own, and similarly
   being unable to independently learn new abstract operators". [V]
2. **Missing kinds of learning.** "there are still types of architectural learning that are missing,
   such as semantic learning, generalized perceptual category learning, and learning expectations or
   predictions of its own actions or the dynamics of its environment." [V]
3. **A self.** "Be self-aware and have a sense of self (no, no) ... there is no explicit
   representation of the agent's 'self.'" [V]
4. **The wall.** "Can metadata be available in WM and testable by PM without disrupting other
   commitments in these cognitive architectures?"; memory appraisals such as feeling of knowing are
   listed as missing. [V]
5. **Grading learned rules by experience.** Chunks from uncertain decisions are to be admitted
   "when there is sufficient accumulated experience"; stated as a plan. [V]
6. **Metacognition beyond toy domains, and in sub-symbolic systems.** Usefulness shown "in limited
   domains" [V]; no emergent architecture counted as metacognitive [V]; "if the underlying processes
   are poorly understood, how can systems be designed to reflect them?" [V]
7. **Reality monitoring.** No implementation attempts known to the 2023 report [V]; the 2021 network
   awaits simulation [V]; in 2026 tests, LLMs and vision-language models confuse the source of
   content [A].
8. **Language.** Unrestricted language is not handled by any Soar agent [V]; in the Thousand Brains
   Project it is future work [V].
9. **Lifelong memory at scale.** Stated as needing "more elaborate solutions" [V]; Monty needs
   "future learning rules that merge existing models over time" [V].
10. **Compositional and abstract models in the sensorimotor line.** Not yet tested computationally
    [V]; unsupervised learning in Monty not yet evaluated [V].
11. **Changing and deleting what was learned, and learning how to learn, in LLM agents.**
    "understudied"; no agent implements learning of new learning procedures [V]. Reconsideration and
    knowledge compilation "have not yet been taken up" [V].
12. **Consecutive tasks without forgetting in a neural architecture.** Future work for CogNGen [V].
13. **A Transformer trained online from an agent's own experience.** A proposal with open challenges
    [A].
14. **Common evaluation.** Fragmented vocabulary, few quantitative comparisons [V].

### 7.2 Not solved because the field has not posed it

All items [I]; "not found" means not found in 30 searches and 65 papers, not proven absent.

1. **The number as the belief, readable by the system.** The field stores numbers about availability
   and payoff, behind a wall. A weight that is itself the degree of belief, with "how much" and "how
   sure" both present and both usable by the system's own operations, appears only in AERA (and in
   NARS, outside this note).
2. **A rule that returns support to the instance it came from, as a stored change.** Not found. The
   systems that induce rules either drop the instance (AERA) or keep the two apart (SAGE, CLARION).
   The known hazard, self-reinforcing error, is documented for LLM memory and is the reason NARS
   forbids the loop. A design that wants this step has to say how it avoids counting the same
   evidence twice.
3. **A question that arises from a newly formed category.** Not found. Questions arise from being
   stuck in a task or from a timer. "Which other members does this category have?" would be a
   question about the extent of a variable the system has just created.
4. **Measuring how real one's own construction is, and learning from that number, in a running
   system.** Specified on paper twice. The only running relative is Copycat's temperature, which
   measures coherence inside one run and does not learn.
5. **The way of constructing as the thing that improves.** Skill at tasks improves (chunking). The
   operations of revising, generalising, feeding back and asking are fixed code in every system read.
6. **Language acquired by the same online mechanism as everything else.** Not attempted in this
   field; language is a parser, a recogniser or an attached LLM.
7. **All of the above in one engine of tensors.** The systems with P2–P5 parts are symbolic; the
   engines of vectors have almost none of those parts. Sigma and CogNGen are the two attempts to
   stand on both sides, and neither forms concepts or monitors its own constructions.
8. **A central step without a single winner.** Not posed; the bottleneck is taken as given.

---

## 8. Vocabulary bridge (the field's names for the owner's ideas)

| Owner's idea | This field's terms |
|---|---|
| Learning and running are one process | online incremental learning; learning as a side effect of processing; "always on" learning (AERA); learning from the conscious broadcast (LIDA); continuous or one-pass learning (HTM) |
| Weights are graded beliefs | metadata: base-level activation, utility (ACT-R, Soar); reliability (AERA); match statistics (CLARION); permanence (HTM); fact probability (SAGE); predicate function (Sigma); evidence value (Monty) |
| A first hearing is stored weakly | initial permanence below the connection threshold; low initial utility; experience factor; surprise memory and graduation (AIBL) |
| Contradiction refines instead of deleting | phasing out and silent predictions; pre-conditions; blending; specialisation; reconsideration |
| A category rule is induced | abstraction (AERA); generalisation (CLARION RER, SAGE); chunking and knowledge compilation (procedural); reflection (LLM agents) |
| The rule feeds support back | top-down activation (CLARION); partial matching (ACT-R); spreading activation in a slipnet (Copycat, LIDA) |
| The question arises by itself | impasse and substate; active learning in interactive task learning; curiosity appraisal |
| Imagination; observing one's own constructions | hypothetical states in working memory; internal simulation; simulation marks (AERA); mental imagery; dreaming in a workspace latent space |
| Measuring how real they are | perceptual reality monitoring; reality threshold; responsibility signal; temperature (Copycat); confidence; feeling of knowing; metacognitive experience |
| The consciousness that operates the network | global workspace and broadcast; cognitive cycle; metacognitive subsystem; meta-level control; operational reflectivity (AERA); process-state buffers |
| The way of constructing improves and transfers | knowledge compilation; bounded recursive self-improvement; meta-learning by modifying agent code; strategy selection |
| Concepts formed when needed | concept creation on an unknown word (Aileen); new reference frame (Monty); composite states (AERA); perceptual learning (LIDA) |

---

## 9. Sources

### Read in full text or in the cited parts ([V])

arXiv:2205.03854 (Laird, Introduction to Soar) · arXiv:2201.09305 (Laird, ACT-R and Soar) ·
arXiv:2111.10268 (Nguyen, Phan, Gonzalez, SpeedyIBL) · arXiv:2101.02231 (Ustun, Rosenbloom et al.,
Sigma) · arXiv:2401.10444 and arXiv:2509.10972 (Sun, CLARION and LLMs) · arXiv:1512.05463 (Cui, Ahmad,
Hawkins, HTM) · arXiv:2507.04494 (Leadholm et al., Monty) · arXiv:2412.18354 (Clay et al., Thousand
Brains Project) · arXiv:2507.05888 (Hawkins et al., Theory 2.0; abstract and one passage) ·
arXiv:2503.23608 (Kanerva) · arXiv:1602.05220 (Eliasmith et al., BioSpaun) · arXiv:2506.07807 (Laird
et al., metacognition in the Common Model) · arXiv:1312.6764 (Nivel et al., AERA) · arXiv:2210.11731
(Mohan & Klenk, concept memory) · arXiv:1604.06849 (Mohan, Kirk, Laird, Rosie) · arXiv:2310.15177
(Ororbia & Kelly, CogNGen) · arXiv:2103.01197 (Goyal et al.) · arXiv:2107.13704 (Blum & Blum) ·
arXiv:2308.08708 (Butlin et al.) · arXiv:2109.12798 (Kawato & Cortese) · arXiv:2309.02427 (Sumers et
al., CoALA) · arXiv:2304.03442 (Park et al.) · arXiv:2610.41604 (Chen et al., audit) ·
arXiv:1610.08602 (Kotseruba & Tsotsos) · arXiv:2503.13467 (Nolte et al.) · arXiv:2505.07087 (Wray,
Kirk, Laird) · arXiv:2610.03413 (Poovendran, Stocco, Bushnell, AIBL) · arXiv:2303.11934 (Bricken et
al.) · arXiv:2603.07670 (Du) · arXiv:2603.11768 (Lam et al.) · arXiv:2006.14800 (O'Reilly et al.;
abstract and a few lines) · arXiv:1811.04747 (Huang; abstract and a few lines) · arXiv:2508.15630
(Ray & Dancy; abstract and a few lines).

### Abstract page only ([A]), arXiv

2012.10390 · 2503.01906 · 2502.21142 · 2305.17375 · 2411.00983 · 2407.10188 · 2607.23927 · 2611.09014 ·
2208.09554 · 2006.01962 · 2204.00619 · 2009.08880 · 2510.03286 · 2602.11516 · 2604.08206 · 2503.03459 ·
2410.11407 · 2308.09830 · 2403.18827 · 2408.09176 · 2506.12224 · 1804.08299 · 2610.00399 · 2606.28045.

Downloaded but not used: 2111.05498, 2111.06077, 1503.07469, 1511.00083, 2105.07308, 2412.16231,
2510.09355 (only its search snippet was used).

### Abstract or snippet only ([A]), not on arXiv

Franklin et al. 2014 and LIDA descriptions (search snippets) · Laird, Wray, Jones, Kirk, Lindes 2024,
AAAI Symposium Series (search snippet; host did not resolve) · Kelly, Arora, West, Reitter 2020,
Cognitive Science (search snippet) · Choo 2018, Spaun 2.0 thesis (search snippet) · Marshall, Metacat
papers (search snippets) · Dijkstra & Fleming reality-monitoring preprints and the Neuron 2025 paper
(search snippets) · Thousand Brains Project forum threads on language and abstract concepts (search
snippets) · O'Reilly, Hazy, Herd 2012, Leabra chapter (search snippet).

### From memory only ([M])

Anderson et al. 2004 (ACT-R) · Rosenbloom, Demski, Ustun 2016 (Sigma) · Sun 2016 and Sun, Merrill,
Peterson 2001 (CLARION and the information-gain formula) · Eliasmith et al. 2012 (Spaun) and the
Neural Engineering Framework · Laird, Lebiere, Rosenbloom 2017 (Standard Model) · Hofstadter &
Mitchell (Copycat) · Kanerva 1988 (Sparse Distributed Memory) · ICARUS, Companions, DUAL, MIDCA,
CogPrime.

### Search log (30 web searches; numbers 24 and 28 in extended mode)

1 Soar learning mechanisms · 2 ACT-R declarative memory equations · 3 Sigma factor graphs and
learning · 4 LIDA cycle and learning · 5 CLARION rule extraction and metacognition · 6 HTM online
learning, limits, comparison with LSTM · 7 Thousand Brains Project and Monty · 8 Spaun, Nengo, online
learning rule · 9 Leabra principles · 10 Common Model and metacognition · 11 CoALA · 12 Sparse
Distributed Memory and hyperdimensional computing · 13 global workspace implementations in neural
networks · 14 Metacat and Copycat · 15 AERA · 16 open problems and critiques of cognitive
architectures · 17 LLMs with Soar and ACT-R · 18 the Soar-plus-Transformer proposal · 19 CLARION
information gain · 20 holographic declarative memory · 21 surveys of metacognitive architectures ·
22 interactive task learning and Rosie · 23 Thousand Brains Theory 2.0 · 24 implementations of
perceptual reality monitoring (extended) · 25 LIDA perceptual learning details · 26 Thousand Brains
roadmap for language and abstract concepts · 27 attention schema and self-models in neural agents ·
28 new architectures of 2026 that learn while running without backpropagation (extended) · 29
failure modes of LLM agent memory · 30 Spaun 2.0.

arXiv API queries (27): ACT-R with blending; instance-based learning theory; ACT-R declarative
memory; LIDA; Sigma; CLARION; Leabra; semantic pointer; Neural Engineering Framework; hierarchical
temporal memory; thousand brains; Copycat; global workspace; common model of cognition; Soar;
sparse distributed memory; Spaun; cognitive architecture with concept and online; reality
monitoring; Sigma by author; LIDA by author; analogical generalization; cognitive architecture with
continual learning; cognitive architecture with Hebbian; second-order network with metacognition;
and two that returned nothing.

### Files in this directory

- `/tmp/claude-0/-workspace-cevahir-ai/79909244-c51d-5307-ad00-1a1e5ef77203/scratchpad/newcore/landscape/cognitive-architectures.md` — this note.
- `/tmp/claude-0/-workspace-cevahir-ai/79909244-c51d-5307-ad00-1a1e5ef77203/scratchpad/newcore/landscape/tools-cognitive-architectures/red_example_cogarch.py` — the worked example; run with `python3 -I`, standard library only.
- `/tmp/claude-0/-workspace-cevahir-ai/79909244-c51d-5307-ad00-1a1e5ef77203/scratchpad/newcore/landscape/tools-cognitive-architectures/red_example_output.txt` — its output.
- `/tmp/claude-0/-workspace-cevahir-ai/79909244-c51d-5307-ad00-1a1e5ef77203/scratchpad/newcore/landscape/tools-cognitive-architectures/` — download and query helpers, logs, a short README.
- `/tmp/claude-0/-workspace-cevahir-ai/79909244-c51d-5307-ad00-1a1e5ef77203/scratchpad/newcore/landscape/sources-cognitive-architectures/` — one directory per arXiv id with the abstract page, the PDF and two text extractions. The directories `1412.6616` and `1412.6616v2` hold no paper (withdrawn); I tried to remove them and the removal was blocked by a safety check, so they remain.
