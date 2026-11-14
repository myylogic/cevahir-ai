# Landscape: predictive coding, free energy and active inference, equilibrium propagation, energy-based and Hopfield networks, contrastive Hebbian, forward-forward and local learning rules, read against P1–P6

Field covered: predictive coding networks (PCNs) and their variants; implementations of the
free-energy principle and active inference (discrete state-space agents, structure learning,
hierarchical Gaussian filtering); equilibrium propagation and contrastive Hebbian learning;
energy-based models and classical, dense and modern Hopfield networks; the forward-forward
algorithm; Hebbian and other local learning rules; and whether any of them does language or
symbolic concept formation online.

Date of this note: 2026-11-14. 26 web searches (1 in extended mode), 91 arXiv papers downloaded;
the ones marked [V] below were read in the parts cited.

## Summary

Tags are explained in §0; each line below is supported in the section named.

- The field is strong as an engine (one scalar functional plus local tensor updates) and has many
  systems that learn while running (P1 native in 14 of 25 rows). A rule level for P2 exists in 3
  rows, all over a hypothesis space written by the designer; units formed on demand (P3) exist
  only in its memory and Bayesian branches; P4 is partial almost everywhere and native nowhere;
  P5 is nearly absent. No row is native or partial on all six. [I] (§2, §4)
- Two halves of the field hold opposite halves of the owner's idea. The neural half (PCNs,
  equilibrium propagation, forward-forward, Hopfield) learns with local rules but its weights are
  point values that an error overwrites. The Bayesian half (discrete active inference) keeps
  graded parameters as Dirichlet counts, selects rules and asks questions, but over a state space
  and a hypothesis space written by the designer, and it does no language. [I] (§4)
- Closest to the "red" example: a discrete active-inference agent with Dirichlet learning,
  Bayesian model reduction and epistemic action selection (the "three-ball" agent of 2017, redone
  in arXiv:2512.21129). On the red stream it gives: first hearing 0.65 for "always dark" (not weak);
  one contradiction puts "sometimes dark, sometimes light" at 0.81; two further colours lift the
  rule to 0.98 and red to 0.995 without new evidence about red; an unseen colour is the most
  informative question (0.30 against 0.14 nats), but only if the designer provided a spare colour
  slot. Its commit ("aha") comes only at 27 colours, and then freezes the belief. [C] (§6)
- The owner's rule "all colours are sometimes dark, sometimes light" is the kind of rule this
  formalism does not look for: its structure prior asks for sparse, deterministic mappings. In a
  rule space widened by hand to contain it, the agent selects it as "shade does not depend on
  colour". [V] / [C] (§6.4)
- Closest neural systems: the generalized hierarchical Gaussian filter (online, mean and
  precision per belief: 0.61 after the first hearing, back to 0.49 after the contradiction) and
  BayesPCN (one-shot continual writes into a posterior over weights). Both lack a rule level. [C] / [V] (§6.5)
- Language: predictive-coding transformers reach perplexity 143 to 176 against 113 to 163 for
  backpropagation, with one or two blocks of width 128; equilibrium propagation does sentiment
  and inference classification on frozen word2vec vectors; Hinton's own next-character
  forward-forward result was withdrawn; a Hebbian assembly model learns nouns and verbs from
  about ten grounded sentences per word, then basic word order; the one GPT-2-scale "Hebbian"
  model trains its slow weights by backpropagation through time. [V] (§5.5)
- What broke: depth (PC works to 5–7 layers), the cost and tuning of iterative inference,
  two-phase clocks, forgetting once weights are shared, discrete symbols under Gaussian error
  correction, hand-built generative models and brittleness in active inference. [V] (§5)
- Unsolved in the field's own terms: scaling local learning to depth and to language models;
  uncertainty over synaptic weights with online structure growth in a neural network (proposed
  in the field's 2025 roadmap, not built); separating the wake and sleep phases of
  forward-forward; where the priors, the model space and the curriculum of an active-inference
  agent come from; moving fast Hebbian state into long-term weights. [V] (§7.1)
- Not posed by the field, as far as 26 searches show: a conviction that rises when a belief
  changes form; rules that are not sparse; a question about the extent of a category; learned
  operators (one source states "one cannot learn to be curious"); a reality measure on
  self-generated constructions used as an online learning signal; language without a vocabulary
  table inside these networks. [I] (§7.2)

## 0. How to read this note

### Status tags (every statement carries one)

| Tag | Meaning |
|---|---|
| [V] | VERIFIED: I read the source text itself (PDF from arxiv.org) for this statement. |
| [V-s] | VERIFIED in a secondary source that I read (mostly arXiv:1805.07092, arXiv:2512.21129, arXiv:2308.07870, arXiv:2312.09257). The original paper was not read. |
| [A] | ABSTRACT: abstract page or search-result snippet only. |
| [M] | MEMORY: from my own knowledge, not checked in this session. |
| [C] | COMPUTED here: output of `tools-predictive-energy/red_example_predictive_energy.py`. Not a literature claim. |
| [I] | My own inference or judgement from the tagged facts. Not a literature claim; treat as opinion. |

The P1–P6 codes, in the table and in §2.1, are all [I] (my reading), based on the mechanism
statements next to them. A reason that quotes or restates a source carries that source's tag; a
reason with no tag is [I].

### Access limits in this session

- Only arxiv.org, export.arxiv.org and aclanthology.org were reachable. A probe with curl got a
  proxy refusal (403) for ncbi.nlm.nih.gov, pmc.ncbi.nlm.nih.gov, biorxiv.org, frontiersin.org,
  europepmc.org, openreview.net, direct.mit.edu, api.semanticscholar.org and ar5iv.labs.arxiv.org;
  WebFetch failed on www.ncbi.nlm.nih.gov with `ENOTFOUND`. Those hosts would have to be allowed
  in the session's network settings.
- Three works that matter for §6 are therefore not read in the original: Friston et al. 2017
  "Active inference, curiosity and insight" (Neural Computation), Smith et al. 2020 on concept
  learning (Frontiers in Computational Neuroscience) and Friston et al. 2020 on linguistic
  communication (Neuroscience & Biobehavioral Reviews). The first is covered by two arXiv papers
  by the same group that restate its method and paradigm ([V-s]); the other two rest on
  abstracts and snippets ([A]) and on memory ([M]).
- Classical items (Hopfield 1982, Ackley–Hinton–Sejnowski 1985, Movellan 1991, Rao & Ballard
  1999, Xie & Seung 2003, Whittington & Bogacz 2017, wake–sleep 1995) are [M] or [V-s].
- Two-column PDFs (arXiv:1810.07411, arXiv:2209.09626, arXiv:2506.23293) were read through
  interleaved text; quotations from them were taken from single-column runs.

### Principles (the owner's, as given)

P1 learning and running are one process, no train/freeze/eval split. P2 weights are graded
beliefs: weak first storage, contradiction refines instead of deleting, a category rule is induced
and feeds support back to the instance. P3 concepts are formed when needed, categories emerge from
instances, no fixed embedding table. P4 imagination: the system observes its own constructions,
measures how real they are, advances by that measure. P5 what is acquired is the way of
constructing (revise, generalise, feed back, ask), improving with experience and transferring
across domains. P6 an engine of tensors and equations, language in and out eventually.

Codes: **N** = natively, **P** = partially, **–** = not at all. **N\*** (P3) = units or categories
emerge natively, but over a substrate or a feature basis that is given. **N†** (P2) = natively,
but over a hypothesis space fixed by the modeller.

Conventions for the three columns where a code needs one:
- P4 is P when something the system itself produced (a prediction, a free-phase state, a
  candidate, an imagined future, a replay) is scored by a measure and the score is used. It
  would be N only if this also happened without input and the measure were of how real the
  product is.
- P5 is – when the operators (revise, generalise, feed back, ask) are fixed equations, even if
  all of them are present; which of them each family has is tabulated in §4. It is P when the
  setting of an operator, or which part of the network operates, is itself learned from
  experience.
- P6 is N when the tensor engine has been shown modelling language at the scale of a small
  standard language model; P when it has no language, or language only as classification, at
  toy scale, or through a hand-built lexicon.

## 1. What unites the field

Every system here defines one scalar over the state of the network (an energy, a variational free
energy, a "goodness") and obtains both its answer and its weight change from that scalar with
local operations. [I]

- Predictive coding: each layer predicts the next and "PC minimizes a local energy function for
  each layer"; activities are relaxed first, then weights follow a local rule. [V] (arXiv:2202.09467 §2.1, Fig. 1)
- Equilibrium propagation: one kind of computation in two phases, a free one and one where the
  outputs are "slightly nudged towards their target"; the update contrasts the two states. [A] (arXiv:1602.05179)
- Contrastive Hebbian learning: "a subtractive measurement between the cross-products of
  activation values collected from a clamped phase and those collected from a free running
  phase"; it "underpins the foundations of Boltzmann learning". [V-s] (arXiv:2312.09257, section
  on contrastive Hebbian learning)
- Forward-forward: "Each layer has its own objective function which is simply to have high
  goodness for positive data and low goodness for negative data". [V] (arXiv:2212.13345 abstract)
- Hopfield networks: memories are minima of an energy; the modern form has "three types of energy
  minima", one averaging all patterns, metastable ones averaging a subset, and single patterns,
  and its update "is equivalent to the attention mechanism used in transformers". [A] (arXiv:2008.02217)
- Active inference: perception, learning and structure selection minimise variational free energy
  and action minimises expected free energy, which is "expected information gain and expected
  value". [V] (arXiv:2512.21129, Eq. 1–2)

Two consequences run through the rest of this note. First, in most of these systems learning is a
difference between two states of the same network, what it settles to by itself and what it
settles to when the world constrains it; this is the field's nearest relative of P4 (§4). Second,
the scalar is available at every moment, so in principle learning does not need a separate
training mode; in practice almost every system still has phases, loops or epochs (§5.2, §5.3). [I]

## 2. Main table: 25 works

P-codes are in the order P1 P2 P3 P4 P5 P6.

| # | Title | First author | Year | Venue / arXiv | Status | P1 P2 P3 P4 P5 P6 |
|---|---|---|---|---|---|---|
| 1 | Predictive Coding: Towards a Future of Deep Learning beyond Backpropagation? (survey; stands for the standard PCN) | Millidge | 2022 | IJCAI 2022 survey track [A]; arXiv:2202.09467 | [V] | P – – P – P |
| 2 | A Stable, Fast, and Fully Automatic Learning Algorithm for Predictive Coding Networks (iPC) | Salvatori | 2024 | ICLR 2024; arXiv:2212.00720 | [V] | P – – P – P |
| 3 | Continual Learning of Recurrent Neural Networks by Locally Aligning Distributed Representations (P-TNCN) | Ororbia | 2018 | arXiv:1810.07411; IEEE TNNLS [M] | [V] | N – – P – P |
| 4 | Lifelong Neural Predictive Coding: Learning Cumulatively Online without Forgetting (S-NCN) | Ororbia | 2019 | NeurIPS 2022 [A]; arXiv:1905.10696 | [V] | N – P P P P |
| 5 | BayesPCN: A Continually Learnable Predictive Coding Associative Memory | Yoo | 2022 | arXiv:2205.09930; NeurIPS 2022 [M] | [V] | N P – P – P |
| 6 | Bayesian Predictive Coding | Tschantz | 2025 | arXiv:2503.24016 | [V] | P P – P – P |
| 7 | The generalized Hierarchical Gaussian Filter | Weber | 2023 | arXiv:2305.10937 | [V] | N P – P P P |
| 8 | Active Inference, Curiosity and Insight | Friston | 2017 | Neural Computation 29:2633–2683 [V-s] | [V-s] + [A] | N N† – P – P |
| 9 | Active inference and artificial reasoning | Friston | 2025 | arXiv:2512.21129 | [V] | N N† – P – P |
| 10 | An Active Inference Approach to Modeling Structure Learning: Concept Learning as an Example Case | Smith | 2020 | Frontiers in Computational Neuroscience [A] | [A] + [M] | N N† P P – P |
| 11 | Supervised structure learning | Friston | 2023 | arXiv:2311.10300 | [V] | N P N\* P – P |
| 12 | AXIOM: Learning to Play Games in Minutes with Expanding Object-Centric Models | Heins | 2025 | arXiv:2505.24784 | [V] | N P N\* P – P |
| 13 | Generative models, linguistic communication and active inference | Friston | 2020 | Neuroscience & Biobehavioral Reviews [A] | [A] | – – – P – P |
| 14 | Active Exploration based on Information Gain by Particle Filter for Efficient Spatial Concept Formation (SpCoAE) | Taniguchi (A.) | 2022 | arXiv:2211.10934 | [V] | N P N\* P – P |
| 15 | Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation | Scellier | 2017 | arXiv:1602.05179 | [A] | P – – P – P |
| 16 | Sequence Learning Using Equilibrium Propagation | Bal | 2023 | IJCAI 2023 [A]; arXiv:2209.09626 | [V] | P – – P – P |
| 17 | The Forward-Forward Algorithm: Some Preliminary Investigations | Hinton | 2022 | arXiv:2212.13345 | [V] | P – – P – P |
| 18 | Hopfield Networks is All You Need | Ramsauer | 2020 | arXiv:2008.02217; ICLR 2021 [M] | [A] | P – P – – P |
| 19 | A Sparse Quantized Hopfield Network for Online-Continual Memory (SQHN) | Alonso | 2023 | arXiv:2307.15040; Nature Communications 2024 [A] | [V] | N P N\* – – P |
| 20 | Prototype Analysis in Hopfield Networks with Hebbian Learning | McAlister | 2024 | arXiv:2407.03342 | [A] + parts [V] | N P N\* – – P |
| 21 | Energy-Based Models for Continual Learning | Li | 2020 | CoLLAs 2022, PMLR 199 [A]; arXiv:2011.12216 | [V] | P – – P – P |
| 22 | Energy-Based Transformers are Scalable Learners and Thinkers (EBT) | Gladstone | 2025 | arXiv:2507.02092; ICLR 2026 [A] | [V] | – – – P – N |
| 23 | Simulated Language Acquisition in a Biologically Realistic Model of the Brain (NEMO) | Mitropolsky | 2025 | arXiv:2507.11788 | [V] | N P N\* – – P |
| 24 | Self-Organizing Language | Eugenio | 2025 | arXiv:2506.23293 | [V] intro and §5; [A] rest | N – N\* P – P |
| 25 | The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain (BDH) | Kosowski | 2025 | arXiv:2509.26507 | [V] | P P P – – N |

Counts [I]: P1 native in 14 rows; P2 native (as N†) in 3 and partial in 10; P3 native (as N\*)
in 7 and partial in 4; P4 native in 0 and partial in 20; P5 native in 0 and partial in 2; P6
native in 2. No row is N or P on all six. Six rows reach five: rows 4 and 7 (missing P2 and P3
respectively) and rows 10, 11, 12 and 14 (missing P5, and without language).

### 2.1 Mechanism (two sentences) and P-reading for each row

**1. Standard PCN.** Value units x and error units ε are arranged in layers; each layer predicts
the next, the activities are relaxed for T steps to lower the summed squared prediction error
(the variational free energy of a hierarchical Gaussian model), and then each weight is changed
by a product of a local error and a local activity. The same network "can function as
classifiers, generators, and associative memories simultaneously". [V]
P1 P: the rule is local and could be applied per input, but the standard use is training in
epochs and then testing, with a relaxation loop and a switch between activity and weight updates
("an external control signal is needed", [V] arXiv:2212.00720). P2 –: weights are point
estimates and the covariance is fixed to identity ([V] §2); "current machine learning
implementations of PC do not treat the model parameters as random variables" ([V]
arXiv:2308.07870). P3 –: fixed architecture. P4 P: each layer's prediction is a construction
tested against what arrives, and the free energy is "a measure of surprise that is always readily
available to the model" ([V] arXiv:2308.07870); nothing is produced and scored when no input is
present. P5 –. P6 P: language only through row 2 and the work in §3.

**2. iPC.** The activity update and the weight update of predictive coding are run
"simultaneously ... at every time step t", which removes the control signal and the wait for
convergence and gives convergence guarantees from incremental EM. On transformer language models
with two blocks, one head and hidden size 128 it reaches perplexity 106.19 (masked) and 142.54
(conditional) against 120.02 and 113.32 for backpropagation, where plain PC gets 523.08 and
206.34. [V]
P1 P: weights move at every step of processing a batch, which is the nearest thing in PCNs to
"running is learning", but the experiments are still train-then-evaluate. P2 –. P3 –: an
8001-token vocabulary. P4 P as row 1. P5 –. P6 P: language at toy scale.

**3. P-TNCN.** A recurrent stack in which every layer predicts the state of the layer below at
the next time step; mismatches correct the hidden states (an "error-correction phase") and drive
local weight updates (Local Representation Alignment), so the model "requires neither unrolling
in time nor the derivatives of its internal activation functions". Weights are updated at every
time step (over 50 parallel sequences for text; with batch size 1 in the continual experiment);
it reaches 1.70 bits per character on Penn Treebank, against 1.38 for an LSTM trained with
backpropagation through time. [V]
P1 N: learning happens step by step inside the forward processing; the state correction also
adapts the model "even if its synaptic weights are held fixed". [V] P2 –. P3 –. P4 P: its
next-step prediction is tested against what arrives, as in row 1. P5 –. P6 P: character-level
language at small scale.

**4. S-NCN.** A predictive-coding "generative cortex" learns from a single pass over a stream
while a second small network (the "functional neural basal ganglia") watches the output error
units for spikes, declares a task shift, creates or recalls a task context, and that context gates
the cortex by lateral inhibition. Forgetting is reduced because each context uses a different
sparse sub-network. [V]
P1 N: stream-like, one pass, no stored data. P2 –: point weights. P3 P: a new context is grown
when the error spikes ("a dynamically-expanding architecture"), but it is a task context, not a
concept. [V] P4 P: the error of its own prediction is the signal for both learning and shift
detection. P5 P: one system "learns to direct and control" another, yet what it controls is only
which sub-network is active. [V] abstract P6 P: no language.

**5. BayesPCN.** A predictive-coding associative memory whose weights carry a Gaussian (matrix
normal) posterior; a write is a "local conjugate normal-normal Bayesian update" done once per
datapoint, a read is activity relaxation, and a "forget" operation diffuses the posterior back
towards the prior. It recalls data written hundreds to a thousand writes earlier. [V]
P1 N: read, write and forget can be interleaved, with no offline refitting. P2 P: the weights
are graded beliefs with uncertainty and nothing is erased by a new write, but updates are
order-invariant averages, a write is strong enough to make an attractor at once, and there is no
rule level. [V] / [I] P3 –. P4 P: reading is error relaxation as in row 1; but a read "can
return a nonsensical output if the memory query is too far away from the stored datapoints or if
the memory is overloaded", and nothing detects that. [V] P5 –. P6 P: images only.

**6. Bayesian predictive coding.** Predictive coding with a Matrix Normal Wishart posterior over
each layer's weights and noise covariance; after the activities are relaxed, the posterior is
updated in closed form from local pre- and post-synaptic statistics. [V]
P1 P: the update is a closed-form accumulation of statistics, tested in full-batch and
mini-batch epochs. P2 P: uncertainty over every weight, no rule level. P3 –. P4 P as row 1.
P5 –. P6 P.

**7. Generalized HGF.** A network of belief nodes, each a Gaussian with mean μ and precision π,
updated in one step per input by precision-weighted prediction errors; a parent node can predict
a child's value (value coupling) or its rate of change (volatility coupling), and several
children can share a parent. For a value parent the update is π_b = π̂_b + α²π̂_a and
μ_b = μ̂_b + (απ̂_a/π_b)δ_a. [V] (Eq. 24)
P1 N: closed-form filtering, one update per observation, no training phase. P2 P: every belief
has a strength (precision) that sets how far the next observation moves it, and a shared parent
gives a top-down path from a group to a member; but a contradiction pulls the mean back rather
than refining the belief's form (§6.5), and the shared parent has to be wired by the modeller.
P3 –: the nodes are given. P4 P: every node's prediction is tested against what arrives and the
precision-weighted error is the only signal. P5 P: how fast to learn is itself a belief
(volatility) updated by experience, and "global" volatility states can be shared across beliefs
[V]; the other operators are fixed. P6 P: no language.

**8. Active inference, curiosity and insight.** An agent with a discrete generative model learns
the likelihood mapping by accumulating Dirichlet counts while it acts, and chooses actions that
resolve uncertainty about those counts; between exposures, Bayesian model reduction compares
the full model with simpler ones and replaces it when a simpler one has more evidence. The paper
presents this as "aha moments" and relates the reduction step to sleep. [V-s] (arXiv:1805.07092
§7.2; arXiv:2512.21129 introduction) / [A]
P1 N: counts change on every trial of normal behaviour; the reduction step is separate. P2 N†:
graded counts, additive updates and rule selection that rewrites the instance-level posterior,
all over "extra (hyperprior) constraints that defined the class of models the simulated subjects
entertained". [V-s] P3 –: states are given. P4 P: reduction is "simply 'thinking about things'
off-line"; it scores alternative models of the same experience, and planning scores imagined
outcomes. [V-s] P5 –: revise, generalise, feed back and ask all exist, as fixed equations (§4).
P6 P.

**9. Active inference and artificial reasoning.** The same agent and "three-ball" task with one
addition: expected free energy also contains the expected information gain over models, computed
by Bayesian model reduction from the accumulated counts, so the agent picks the observations
that best separate 81 candidate rules; when the log odds of the best rule pass 16 nats it
replaces its posterior by that rule. Learning is a ← a + o⊗s. [V] (Eq. 6, 9–13)
P1 N. P2 N†: as row 8; after the commit "one can multiply the Dirichlet counts in the selected
model by some arbitrarily high number (e.g., 512)", which ends gradedness. [V] P3 –: "we have
assumed that the agent knows the generative model". [V] P4 P: "the homologue of 'thinking' in
active inference is the planning (as inference) in which counterfactual futures are evaluated".
[V] P5 –: the operators are fixed, and the paper says so: "although one can learn to infer ...
one cannot learn to be curious". [V] P6 P: "specified by a set of tensors"; no language. [V]

**10. Concept learning as structure learning.** An agent's state space contains unused concept
"slots"; a novel pattern of features is assigned to a spare slot and learned by Dirichlet
accumulation, and Bayesian model reduction decides whether the extra slot is warranted or should
be merged back. [A] / [M]
P1 N. P2 N†. P3 P: concepts are acquired when needed, but the slots are provided beforehand. P4 P.
P5 –. P6 P. All codes rest on the abstract, snippets and memory.

**11. Supervised structure learning.** For each new observation the model compares itself with
versions that have one more state, one more path or one more factor, using free energy plus a
prior from expected free energy (a mutual-information constraint), and grows when the augmented
model wins: "do we augment the model—given this new observation—or not?". Parameters are
Dirichlet counts and small counts are pruned afterwards. [V]
P1 N: "learning becomes a kind of inference ... enabling one-shot learning". [V] P2 P: counts,
no rule level. P3 N\*: states and factors are created on demand, in a discrete space whose
"directions of growth are well-defined", and only if data arrive in the right order. [V] P4 P.
P5 –. P6 P.

**12. AXIOM.** Four mixture models (slots for objects, object identity, motion prototypes,
object interactions) are updated "sequentially one frame at a time with variational Bayesian
updating" without gradients or replay; a datapoint that no component explains above a threshold
creates a new component, and every 500 frames Bayesian model reduction greedily merges
interaction clusters. Actions are chosen by expected utility and information gain. [V]
P1 N. P2 P: conjugate posteriors; merging lets it "generalize dynamics from single events ... by
merging multiple single event clusters", so a general cluster replaces its instances rather than
supporting them. [V] / [I] P3 N\*: components grow on demand, over object-centric priors that
"are themselves engineered rather than discovered". [V] P4 P: merges are scored "with respect to
data generated from the model through ancestral sampling", i.e. on the model's own samples, and
plans are rollouts. [V] P5 –. P6 P: games from pixels, no language.

**13. Linguistic communication.** Two synthetic agents with the same hand-built hierarchical
generative model play Twenty Questions; the questioner selects the question expected to resolve
most uncertainty about the hidden scene. [A]
P1 –, P2 –, P3 –: the abstract and snippets describe inference under a given model, not learning.
P4 P: questions are evaluated by expected information gain. P5 –: asking is native, from a given
repertoire, and fixed. P6 P: language in and out through a hand-built lexicon and syntax.

**14. SpCoAE.** A robot learns place categories with a multimodal Dirichlet process mixture over
positions and the words a user says there, updated online by a Rao-Blackwellized particle
filter; it then moves to the position where asking "What kind of place is this?" has the highest
expected information gain, which the paper equates with expected free energy under a uniform
preference. [V]
P1 N: online learning in the action loop. P2 P: posterior counts. P3 N\*: categories and their
word associations form as needed, up to a set maximum L. P4 P: gain is computed on predicted
answers. P5 –: the system asks, with one fixed sentence; where to ask is computed. P6 P: a
probabilistic model, words in as a bag of words, one question out.

**15. Equilibrium propagation.** A recurrent energy-based network settles freely on the input,
then the outputs are nudged slightly towards the target and it settles again; each weight
changes by the difference of local activity products between the two states, which estimates the
gradient of the loss. [A]
P1 P: the rule is local in space; in the standard form it is "not local in time" and the input
must stay fixed through both phases; a later variant updates weights continually during the
second phase. [A] (arXiv:2005.04168) / [V] (arXiv:2209.09626) P2 –. P3 –. P4 P: the free state is
the network's own answer and the update is its distance from the nudged one. P5 –. P6 P.

**16. Sequence learning with EP.** A modern Hopfield layer is placed inside the convergent
network as an attention module so that a whole sequence can be presented as one static input;
with it, equilibrium propagation reaches 88.9% on IMDB sentiment and 81.4% on SNLI. [V]
P1 P. P2 –. P3 –: "due to limitations of EP, we employed fixed word-embeddings" (300-d
word2vec). [V] P4 P. P5 –. P6 P: classification only.

**17. Forward-forward.** Two forward passes replace forward and backward: each layer raises its
goodness (for instance the sum of squared activities) on real data and lowers it on negative
data, "which could be generated by the network itself". It "can learn while pipelining
sequential data through a neural network without ever storing the neural activities". [V]
P1 P: no backward pass and no stored activities, but a negative pass is needed and training is by
epochs. P2 –. P3 –. P4 P: this is the field's most literal reality measure, a per-layer score
trained to tell real input from the net's own product; the version in which the net's own
negatives are processed in a separate phase could not be reproduced (§5.3). [V] P5 –. P6 P.

**18. Modern Hopfield network.** Continuous patterns are stored as rows of a matrix and a query
is updated by a softmax-weighted average of the stored patterns; capacity is exponential in the
dimension and retrieval takes one update. Minima are of three kinds: all patterns averaged, a
subset averaged (metastable), or one pattern. [A]
P1 P: storing a pattern is a write, not training, but as a layer it is trained by
backpropagation. [A] / [I] P2 –: a pattern is stored at full strength (§6.5). P3 P: a metastable
state that averages similar patterns is an emergent prototype. [A] / [I] P4 –. P5 –. P6 P: the
update "is equivalent to the attention mechanism used in transformers", so its route to language
is the Transformer the owner wants to leave. [A] / [I]

**19. SQHN.** A tree of nodes with one-hot ("sparse quantized") states learns online by maximum
a posteriori updates; the weight rule is a running average with step 1/(c+1), where c counts
how often the parent value was active, and "a new neuron is grown at node l if no neuron at the
node has a value greater than some threshold", a threshold that decays as α/(t+α). [V]
P1 N: one pass, non-i.i.d., local, no buffer. P2 P: count-based weights, no rule level; the first
presentation is memorised at once ("one-shot memorization"). [V] P3 N\*: neurons are allocated on
mismatch. P4 –. P5 –. P6 P: memory tasks only; "future work will need to assess the SQHN on
tasks that require generalization". [V]

**20. Prototype formation in Hopfield networks.** With plain Hebbian storage, many correlated
examples make an unlearned state stable that represents the whole group, and its attractor
"strength grows with the number of examples and the agreement of examples". [A] The paper
derives a stability condition and a prototype capacity. [V] (located in text)
P1 N: Hebbian storage is one additive step per pattern. [M] P2 P: the prototype's strength is
graded by evidence, but the examples themselves stop being retrievable. [I] P3 N\*: a category
emerges from instances without being allocated. P4 –: an emergent state is not tested against
anything. P5 –. P6 P.

**21. EBMs for continual learning.** Classification is trained as an energy over (input, label)
pairs: the energy of the true label is lowered and that of one sampled other label is raised,
instead of normalising over all classes. The authors argue that "the softmax operation itself
introduces competitive, winner-take-all dynamics that make the classifier catastrophically
forget past tasks". [V]
P1 P: works in a stream with no task boundaries, still gradient training of a deep net. P2 –.
P3 –. P4 P: "a wake-sleep process" on labels. [V] P5 –. P6 P.

**22. EBT.** A transformer outputs one scalar energy for a context and a candidate prediction;
a prediction is made by gradient descent on the candidate, and the model is trained by
backpropagating the final loss through that descent, which needs second derivatives. It was
scaled to 800M parameters on next-token prediction. [V]
P1 –: pre-trained, then fixed. P2 –. P3 –: a standard tokenizer. P4 P: the model "learn[s] to
explicitly verify the compatibility between inputs and candidate-predictions", generates several
candidates and keeps the lowest-energy one ("self-verifying their own predictions"); the measure
steers the answer, not the learning. [V] / [I] P5 –. P6 N.

**23. NEMO language acquisition.** Brain areas of randomly connected excitatory neurons fire by
k-winners-take-all; a synapse whose two neurons fire in succession is multiplied by (1+β), with
β around 0.05. From grounded two- and three-word sentences the system forms a stable assembly
per word, sorts nouns and verbs into two areas, learns the word order and produces new
sentences, with "just around 10 sentences per word". [V]
P1 N: plasticity is part of firing. P2 P: the first exposure changes a synapse by a few percent,
so first storage is natively weak, and strength grows with repetition; there is no mechanism for
a contradiction and no rule that supports an instance. [V] / [I] P3 N\*: word assemblies form in
empty areas, but the phonological and sensory input assemblies are initialised by hand. [V] P4 –.
P5 –. P6 P: language in and out at the level of "the dog runs"; "adjectives, adverbs, and
functional categories are not considered". [V]

**24. Self-Organizing Language.** A hierarchy of local Hopfield-type memories (the
"retokenizer") stores pairwise correlations of letters, then of the tokens these define, and so
on; during replay a new neuron is bound to the features that fire together and becomes a
long-term "object token (a.k.a an embedding)". "Long-term memories live in parallel channels;
adding one does not overwrite others." [V]
P1 N: "continuous-learning", local events. [A] P2 –: correlations are stored or forgotten, not
graded as beliefs. [I] P3 N\*: the embedding of a word is created when the word has been
replayed, not looked up in a table; but the units are word forms whose "symbols are entirely
meaningless". [V] P4 P: "continued learning arises from the retrieval", so the model's own
replay is what it learns from, without a reality test. [V] / [I] P5 –. P6 P: tensor machinery,
toy demonstrations of word structure.

**25. BDH.** A language model whose state is a matrix of synaptic strengths σ that grow by a
Hebbian rule while text is read ("fast weights"), on top of fixed weights G; "All models are
trained with truncated backpropagation through time". The abstract reports that it "rivals GPT2
performance on language and translation tasks, at the same number of parameters (10M to 1B)".
[V] / [A]
P1 P: synapses change during use, but that state is working memory; moving it into long-term
weights is left open ("we do not provide a direct answer"). [V] P2 P: graded synaptic state
without revision or rules. P3 P: sparse positive activations that the authors report as
monosemantic emerge, over character input. [A] / [V] P4 –. P5 –. P6 N.

## 3. Supporting works (read or located, not in the main table)

| Work | Status | Why it matters here |
|---|---|---|
| Pinchetti et al. 2022, Predictive Coding beyond Gaussian Distributions, NeurIPS 2022 [A], arXiv:2211.03481 | [V] | First PC transformer language model: one block, hidden 128, 32 tokens, 8001-token vocabulary; perplexity 175.90 (KL form) and 590.08 (standard PC) against 162.64 for BP. States that to its knowledge none of the neuroscience-inspired frameworks had trained language models before. |
| Pinchetti et al. 2024, Benchmarking Predictive Coding Networks – Made Simple, arXiv:2407.01163 | [V] | PC matches BP up to depth 7, then falls behind; energy stays in the last layer. |
| Innocenti et al. 2025, μPC: Scaling Predictive Coding to 100+ Layer Networks, NeurIPS 2025, arXiv:2505.13124 | [V] | Two depth pathologies; 128-layer residual nets on simple tasks; ill-conditioning not solved. |
| Salvatori et al. 2023–2025, Brain-inspired Computational Intelligence via Predictive Coding, arXiv:2308.07870 (v3, October 2025) | [V] | The field's roadmap: limitations and open directions (§7.1). |
| Ororbia 2023, Brain-Inspired Machine Intelligence: A Survey of Neurobiologically-Plausible Credit Assignment, arXiv:2312.09257 | [V] parts | Taxonomy of local learning; contrastive Hebbian learning. |
| Friston, Parr, Zeidman 2018, Bayesian model reduction, arXiv:1805.07092 | [V] | The Dirichlet reduction formula; hierarchical (empirical Bayes) use; summary of the 2017 simulations. |
| Heins et al. 2022, pymdp, JOSS, arXiv:2201.03904 | [V] parts | `update_obs_likelihood_dirichlet(..., lr=1.0)`; the novelty term `calc_pA_info_gain`. |
| Friston et al. 2024, From pixels to planning: scale-free active inference, arXiv:2407.20292 | [V] | Fast structure learning ("a new latent cause to explain each (unique) observation"); language not addressed; brittleness. |
| Zaghw et al. 2026, Renormalising Generative Models for Active Inference: Foundations, Derivations, and Verification, arXiv:2608.09512 | [A] | Re-derivation and open implementation of the 2024 models; evaluation on benchmarks named as future work. |
| Millidge, Tschantz, Buckley 2020, Whence the Expected Free Energy?, arXiv:2004.08128 | [A] | Expected free energy "is not simply the free energy in the future"; exploration does not follow from it directly. |
| Mazzaglia et al. 2022, The Free Energy Principle for Perception and Action: A Deep Learning Perspective, arXiv:2207.06415 | [V] parts | Scaling active inference to large or continuous state spaces is done with deep-learning techniques and amortised inference. |
| Neacsu et al. 2022, Structure learning enhances concept formation in synthetic Active Inference agents, PLOS ONE; van Erp et al. 2024, Online Structure Learning with Dirichlet Processes through Message Passing | [A] snippets | Further structure-learning agents: concept formation while foraging; a message-passing model that grows through a Dirichlet process. Not read. |
| Sandved-Smith et al. 2021, deep parametric active inference, Neuroscience of Consciousness | [A] | A higher level infers and sets the precision of a lower level: attention as a "mental action" (P5 vocabulary). |
| Taniguchi 2024, collective predictive coding hypothesis; arXiv:2501.00226 | [A] | Symbol emergence as decentralised Bayesian inference between agents (naming game). |
| Ernoult et al. 2020, EP with Continual Weight Updates, arXiv:2005.04168 | [A] | Weight update made local in time. |
| Laborieux et al. 2020, arXiv:2006.03824; Laborieux & Zenke 2022, arXiv:2209.00530; Lin et al. 2025, arXiv:2508.15989; Scellier et al. 2023, arXiv:2312.15103 | [A] | Scaling EP: 11.7% error on CIFAR-10, ImageNet 32x32, deep VGG; depth limits. |
| Ororbia & Mali 2023, Predictive Forward-Forward, arXiv:2301.01452; Chen et al. 2025, Self-Contrastive FF, Nat. Commun., arXiv:2409.11593; Gandhi et al. 2023, arXiv:2307.04205; Krutsylo 2025, arXiv:2501.03176 | [A] | FF follow-ups: a generative circuit learned jointly; negative-data problem; IMDb sentiment as the first use outside vision. |
| Deperrois et al. 2022, perturbed and adversarial dreaming, eLife, arXiv:2109.04261 | [A] | Wake, NREM and REM phases; in REM the feedback path generates and the feedforward path discriminates. |
| Hopfield, Feinstein, Palmer 1983, 'Unlearning' has a stabilizing effect in collective memories, Nature | [A] snippet | Start from noise, relax, unlearn the state reached; reduces spurious states; too much destroys memories. |
| Fachechi, Agliari, Barra 2018, Dreaming neural networks, arXiv:1810.12217 | [A] | Online Hebbian storage awake, offline removal of spurious states and reinforcement of pure ones asleep; capacity from 0.14 to 1. |
| Agliari et al. 2021, The emergence of a concept in shallow neural networks, arXiv:2109.00454; Benedetti et al. 2026, arXiv:2602.01393; Cowsik & Sriram 2026, arXiv:2609.12079; Pham et al. 2025, arXiv:2505.21777 | [A] | Archetypes learned from blurred examples above a critical sample size; spurious states as the onset of generalisation. |
| Krotov & Hopfield 2016, Dense Associative Memory, NeurIPS, arXiv:1606.01164; Hoover et al. 2023, Energy Transformer, NeurIPS, arXiv:2302.07253; Millidge et al. 2022, Universal Hopfield Networks, arXiv:2202.04557 | [A] | Feature-matching versus prototype regimes; attention designed as energy descent; memory as similarity, separation, projection. |
| McAlister et al. 2024, Sequential Learning in the Dense Associative Memory, arXiv:2409.15729 | [V] parts | Dense memories learn tasks in sequence about as other deep networks do; rehearsal is "extremely effective"; stability limits the number of tasks. |
| Li, Qiu, Huang 2023, Meta predictive learning model of languages in neural circuits, Phys. Rev. E, arXiv:2309.04106 | [A] + skim | Predictive coding where each weight is a spike-and-slab distribution; word-level Penn Treebank with an embedding layer; below a transformer. |
| Mitropolsky & Papadimitriou 2023, arXiv:2306.15364; Mitropolsky, Collins, Papadimitriou 2021, arXiv:2108.02189 | [A] | Earlier assembly-calculus language organ and parser. |
| Eugenio 2025, Hebbian learning the local structure of language, arXiv:2503.02057 | [A] + skim | Earlier version of row 24: Hebbian tokenizer hierarchy plus embedding neurons. |
| Miconi et al. 2018, arXiv:1804.02464; Najarro & Risi 2020, arXiv:2007.02686; Lindsey & Litwin-Kumar 2020, arXiv:2006.09549; Kirsch & Schmidhuber 2021, arXiv:2012.14905; Zucchet et al. 2022, arXiv:2104.01677 | [A] | Plasticity rules found by an outer loop (gradient or evolution); one "can even meta learn new [learning algorithms] that ... generalize to datasets outside of the meta training distribution". |
| Song et al. 2024, prospective configuration, Nature Neuroscience; Millidge et al. 2022, arXiv:2207.12316 | [A] | PCNs "outperforming [BP] on tasks such as online, few-shot, and continual learning". |
| Cui, Ahmad, Hawkins 2016, HTM sequence memory, arXiv:1512.05463; Journé et al. 2023, SoftHebb, arXiv:2209.11883; Li, Teh, Pascanu 2025, NoProp, arXiv:2503.24322 | [A] | Other local or feedback-free learners (HTM is treated in art-growth.md). |
| Wen 2025, arXiv:2508.05619; Prakki 2024, arXiv:2412.10425; Zhu et al. 2026, HeLa-Mem, arXiv:2604.16839 | [A] | Active inference or Hebbian memory placed around a pre-trained language model; the language part is a standard LLM. |

## 4. Reading by principle

**P1.** Native where the update is a closed-form accumulation per event: Dirichlet counts in
active inference (rows 8–12, 14), conjugate Gaussian writes (row 5), one-step filtering (row 7),
count-averaged Hebbian writes with neuron growth (row 19), Hebbian storage (rows 20, 23, 24), and
online local alignment in a recurrent net (rows 3, 4). Partial wherever an error must first be
relaxed or contrasted: PCNs, equilibrium propagation, forward-forward, energy-based training.
The gradient-trained language systems are not online at all (row 22) or only in their fast state
(row 25). [I]

**P2.** Three separate things exist, never together in a network. (a) A strength per parameter:
Dirichlet counts, weight posteriors (rows 5, 6), node precisions (row 7), counts (row 19). (b)
Non-destructive refinement: counts only add; posteriors sharpen. (c) Rule induction with
feedback to the instance: only in rows 8–10, as Bayesian model reduction over a designer's model
space, and in the hierarchical empirical-Bayes use of the same formula, where lower-level
posteriors are recomputed "under successive empirical prior[s] afforded by higher levels" ([V]
arXiv:1805.07092 §7), an analysis tool for group data, not an online agent. A natively weak first
hearing appears only as a small Hebbian multiplier (row 23) or a learning rate. [I]

**P3.** Allocation on demand exists in the Bayesian half as model expansion (rows 11, 12, 14)
and as spare slots (row 10); in the neural half as neuron growth for memory (row 19), assemblies
(row 23), replay-bound embedding neurons (row 24) and emergent prototype attractors (rows 18,
20). Every language experiment in PC, EP and EBT uses a fixed token vocabulary or frozen word
vectors. [V] rows 2, 16, 22; [I]

**P4.** This is the field's home ground in one narrow sense: its basic act is to produce
something and compare it with what the world supplies. Five partial forms, none complete. (1)
A prediction tested against the input that is present: prediction error in every PC-family
system (rows 1–7), with free energy as a standing surprise measure. (2) The network's
unconstrained state contrasted with a constrained one as the learning signal: the free phase
against the nudged or clamped phase (rows 15, 16, Boltzmann learning), and the model's own
negatives against data (rows 17, 21). (3) A learned verifier applied to the system's own
candidates at answer time (row 22). (4) Imagined futures scored by expected free energy (rows
8–14) and model merges scored on the model's own samples (row 12). (5) Dreaming that removes what
the network produces from noise (Hopfield unlearning) or that pits a generator against a
discriminator (PAD), and replay that the system learns from (row 24). What is missing in all:
a product made with no input present is never scored for how real it is and then used to
advance, inside one continuously running system; and the measure is not itself something that
improves with experience. [I]

**P5.** Fixed operators only. Which of the owner's four verbs each family has, as an equation: [I]

| Family | revise | generalise | feed back to the instance | ask |
|---|---|---|---|---|
| PCN, EP, FF, EBM (rows 1–6, 15–17, 21, 22) | error-driven weight change | only implicitly, through shared weights | only implicitly | – |
| HGF (row 7) | precision-weighted update | a shared parent, if wired by hand | the parent's prediction to its child | – |
| Discrete active inference (rows 8–10, 13) | count update | model reduction over a given model space | posterior averaged or replaced under the rule | epistemic value over given queries |
| Growing Bayesian models (rows 11, 12, 14) | count or conjugate update | expansion and merging of components | the merged component predicts for its members | information gain; one fixed question in row 14 |
| Hopfield and Hebbian assemblies (rows 18–20, 23, 24) | additive Hebbian write | prototype attractor, replay-bound unit | the attractor pulls the instance in | – |

None of these operators is acquired, and one source says that for curiosity it cannot be (row
9). Two things are learned about operating: the HGF learns how fast to learn, through
volatility (row 7), and S-NCN learns which sub-network should be active (row 4). Deep parametric
active inference has a higher level that sets the precision of a lower one, with hand-given
roles (§3). Meta-learned plasticity acquires a learning rule and transfers it to new datasets,
but in an outer optimisation loop outside the system's life (§3). [I]

**P6.** Everything is a tensor engine. Language at scale exists only where training is
backpropagation (rows 22, 25). [I]

## 5. Known failure modes: what broke

### 5.1 Depth and scale
- PC: "the performance of models of depth up to 7 is comparable to those of backprop, while those
  of deeper models lag behind"; the best PC results came from the shallowest network (VGG5 >
  VGG7 > VGG9 > ResNet), the reverse of backpropagation. [V] (arXiv:2407.01163)
- Cause named by the field: "the energy in the last layer is orders of magnitude larger than the
  one in the input layer", which "leads to exponentially small gradients when the depth of the
  model increases". [V] (arXiv:2407.01163 §5)
- μPC trains 128-layer residual nets on "simple classification tasks" but "failed to solve the
  growing ill-conditioning of the inference landscape with the network depth". [V] (arXiv:2505.13124 §7)
- The roadmap: PCNs "face a scalability bottleneck that limits their effectiveness on large-scale
  deep learning tasks, preventing any kind of industrial adoption". [V] (arXiv:2308.07870)
- EP: without symmetric nudging, training "does not scale to visual tasks harder than MNIST";
  "prior studies on EP have been constrained to shallow architectures, as deeper networks suffer
  from the vanishing gradient problem". [A] (arXiv:2006.03824; arXiv:2508.15989)
- FF: "somewhat slower than backpropagation and does not generalize quite as well on several of
  the toy problems ... so it is unlikely to replace backpropagation for applications where power
  is not an issue". [V] (arXiv:2212.13345 §1) It "has failed to reach state-of-the-art
  performance on most standard benchmark tasks, in part due to unreliable negative data
  generation methods for unsupervised learning". [A] (arXiv:2409.11593)
- Local rules in general "have so far failed to match the performance of backpropagation in deep
  networks". [A] (arXiv:2006.09549)

### 5.2 Stability and cost of iterative inference
- PC has "three limitations: first, an external control signal is needed to switch ...; second,
  the update of the neural activities is slow, as it can require dozens of iterations to
  converge; third, convergence is uncertain and highly dependent on the choice of
  hyperparameters". [V] (arXiv:2212.00720 §1)
- On language models "the poor performance of PC is due to its poor training stability": 3 of 10
  seeds of the masked model converged under plain PC, 10 of 10 under iPC, 7 of 10 under
  backpropagation. [V] (arXiv:2212.00720 §5, Table 3)
- The inference landscape of a PCN "becomes increasingly ill-conditioned with model size and
  training time", and its forward pass vanishes or explodes with depth. [V] (arXiv:2505.13124)
- The number of relaxation steps must be at least the depth, with memory above backpropagation's. [V] (arXiv:2308.07870)
- BayesPCN "read can take up to thousands of activation gradient descent iterations". [V]
- S-NCN: for harder data the number of steps per sample "will quite likely need to be much
  higher, increasing the computation time further needed to conduct online inference". [V]
- EBT: step size and step count are new hyperparameters and "poorly chosen values often lead to
  unstable training"; training needs second derivatives; it "struggle[s] with data distributions
  that have many modes". [V]
- Discrete active inference: "data are ingested or assimilated rather slowly ... the MNIST example
  can take about an hour to assimilate 10,000 digits". [V] (arXiv:2311.10300)

### 5.3 Two phases and their clock
- EP "requires the input to the model ... to be static in both the phases of training", so "it is
  not possible to design a model for sequence classification using EP with an LSTM or GRU like
  architecture". [V] (arXiv:2209.09626)
- FF: Hinton reported that separating many positive updates from many negative updates on
  self-generated data worked for next-character prediction, then wrote: "I have been unable to
  replicate this result and I now suspect it was due to a bug." Whether the phases can be
  separated "is probably the most important outstanding question about FF as a biological
  model". [V] (arXiv:2212.13345 §5)
- Boltzmann-style negative phases need sampling: "Utilizing MCMC to sample negative images makes
  EBMs difficult to scale to complex datasets and makes training much slower". [V] (arXiv:2011.12216)

### 5.4 Forgetting
- The PC associative memory of 2021 "is not amenable to continual learning. Due to its lack of
  safeguards against catastrophic forgetting", it forgets or needs "a costly offline model
  refitting on all observations whenever a new datapoint is observed". [V] (arXiv:2205.09930)
- Where forgetting is avoided it is by isolation, not by a shared code surviving: task contexts
  that gate sub-networks (row 4), one-hot codes with a neuron grown per pattern (row 19, "we
  primarily use parameter isolation"), parallel channels (row 24). [V] / [I]
- Online-continual learning with local rules is itself named open: "essentially all of this work
  uses BP in some capacity". [V] (arXiv:2307.15040)
- Dense associative memories trained in sequence behave "reasonably typically with respect to
  other deep learning architectures": rehearsal is "extremely effective", and the authors ask
  whether the "stability issues" can be resolved to allow "more than a small handful of tasks
  sequentially". [V] (arXiv:2409.15729 §6)
- A softmax output layer is singled out as a cause of forgetting. [V] (arXiv:2011.12216 §3.2)
- The classical Hopfield network stores about 0.14 patterns per neuron. [A] (arXiv:1810.12217)

### 5.5 Language
- 2022: "none of these frameworks has been successfully used to train language models to date". [V] (arXiv:2211.03481)
- PC transformers since then: one block, width 128, perplexity 175.90 against 162.64 (2022); two
  blocks, width 128, 142.54 against 113.32 on the conditional model, with 3 of 10 seeds not
  converging (2024). [V] (arXiv:2211.03481; arXiv:2212.00720) The roadmap still lists language
  models as where PC has to catch up. [V] (arXiv:2308.07870)
- Discrete symbols: "Predictive coding models have almost exclusively been applied to real-valued
  data and it is not clear if there is some property of discrete/symbolic data that might
  interact negatively with the process of iterative error correction." [V] (arXiv:1810.07411)
  The Gaussian assumption is the stated obstacle for softmax layers. [V] (arXiv:2211.03481)
- EP: sentence classification only, with frozen word vectors and symmetric weights; time-varying
  input is future work. [V] (arXiv:2209.09626)
- FF: IMDb sentiment is described as its first use outside vision. [A] (arXiv:2307.04205)
- Active inference: renormalising models "could be applied to language ... We will not address
  this here". [V] (arXiv:2407.20292) The Twenty Questions agents use a given generative model. [A]
- Hebbian assemblies: learning stops at nouns, verbs and basic word order; abstract words need
  "a representation scheme for such words and the introduction of new semantic brain areas";
  hierarchical sentences "need further work". [V] (arXiv:2507.11788)
- The self-organising memory explains word forms; its symbols carry no meaning. [V] (arXiv:2506.23293)
- BDH is trained with truncated backpropagation through time; with the through-time part of the
  gradient removed the model "retained some ability to model language, but lost the ability to
  match concepts between different languages". [V] (arXiv:2509.26507 §7.2)

### 5.6 Active inference: the designer's hand, brittleness, premature commitment
- "applications of active inference have typically been confined to small-scale tasks with
  carefully designed priors". [V] (arXiv:2505.24784)
- The model space for rule selection is a hyperprior the designer writes: "An outstanding issue
  is the elaboration of alternative models; i.e., the specification of the model space." [V] (arXiv:2512.21129)
- Order dependence: "the order in which data are encountered—or presented, i.e. the scheduling or
  curriculum—matters ... there is some requisite supervision of structure learning". [V] (arXiv:2311.10300)
- Brittleness: models grown to be minimal "will not recognise or respond to events they have not
  previously encountered". [V] (arXiv:2407.20292)
- "there is always a possibility of jumping to conclusions, even with a stringent threshold";
  one of 64 runs committed to the wrong rule after six trials. [V] (arXiv:2512.21129)
- Scaling to large or continuous state spaces has been done with deep-learning techniques and
  amortised inference, not with the count-based scheme. [V] (arXiv:2207.06415, abstract and introduction) / [I]
- The objective itself is contested: exploration "does not directly follow from free energy
  minimization into the future". [A] (arXiv:2004.08128)

### 5.7 No uncertainty on the weights of the neural systems
- PC as used in machine learning is "an expectation maximization (EM) procedure, where the M-step
  ignores uncertainty about the parameters". [V] (arXiv:2308.07870)
- The Bayesian fix needs structured posteriors: for larger networks "structured low-rank
  approximations will become necessary". [V] (arXiv:2503.24016)

### 5.8 The network's own products are not checked
- Hopfield storage creates spurious stable states; whether they are errors or the first sign of
  generalisation is now argued both ways. [A] (Hopfield et al. 1983 snippet; arXiv:2505.21777)
- BayesPCN can return nonsense for a far query (row 5); grown active-inference models do not
  respond to the unseen (§5.6). [V]
- EBT's energy and FF's goodness are reality measures, the first trained offline, the second not
  yet shown to work on the network's own negatives in a separate phase. [V] / [I]

## 6. The closest existing system on the "red" example

### 6.1 Which system, and why
The owner's steps are: "red is dark" 0.23 → "dark but sometimes light" 0.57 → "all colours are
sometimes dark, sometimes light" 0.77 → red 0.88 → the question "which other colours exist?".

The system in this field that has a mechanism for each step is the discrete active-inference
agent with Dirichlet learning, Bayesian model reduction and epistemic action selection: rows 8
and 9, with rows 10 to 12 as its structure-learning extensions. [I] It is "strongest" in the
sense of covering most steps, not in scale: its published tasks have a few hundred likelihood
columns. [V] (arXiv:2512.21129: 432 state combinations, 81 rules)

### 6.2 What the designer must supply before the first sentence
[I], from the method as read in arXiv:2512.21129:
1. A state factor "colour" with its levels, and an outcome modality "shade" with levels dark and
   light. The sentence "red is dark" has to arrive already parsed as (state = red, outcome =
   dark); the agent has no language.
2. A Dirichlet prior over the likelihood tensor P(shade | colour) and a learning rate.
3. A space of alternative likelihood tensors (the rules it may discover), a prior over them and a
   commit threshold.
4. An action repertoire, for instance "sample the shade of colour c" for each c.
5. If new colours are to be possible, spare levels in the colour factor (row 10) or an
   expansion rule (rows 11, 12).

### 6.3 Step by step
All numbers [C], from `tools-predictive-energy/red_example_predictive_energy.py`. Settings
taken from the paper: counts add one per observation (Eq. 6); reduced models have columns that
sum to about one, with shrinkage value 1/32; the full prior is the average of the reduced priors
(Eq. 8); model posteriors are exact evidence ratios (Eq. 7, 9); commit at 16 nats (Eq. 12).
Settings that are mine: three per-colour hypotheses, D "always dark" = (1, 1/32), L "always
light" = (1/32, 1), M "sometimes dark, sometimes light" = (1/2, 1/2); and six rules over all
colours with a uniform prior: R_M every colour mixed at its own rate, R_D all dark, R_L all
light, R_F each colour one fixed shade, R_I no rule (each colour D, L or M independently), R_P
shade does not depend on colour (one shared rate).

| Owner's step | What the agent holds | Number |
|---|---|---|
| "red is dark", 0.23 | counts for red (1.51, 0.51); expected P(dark) | 0.747 |
| | posterior of "red is always dark" | 0.646 (0.531 at learning rate 0.25) |
| "red is light" → "dark but sometimes light", 0.57 | counts (1.51, 1.51); expected P(dark) | 0.500 |
| | posterior of "red is sometimes dark, sometimes light", red alone | 0.807 |
| | the same with the rule space | 0.842 |
| blue and green each heard dark once and light once → "all colours ...", 0.77 | P(R_M) + P(R_P), after blue | 0.921 |
| | P(R_M) + P(R_P), after green (R_P 0.699, R_M 0.280) | 0.979 |
| red after the rule, 0.88 | posterior of "red is mixed", with no new evidence about red: after blue, after green | 0.972, 0.995 |
| | the same without a rule space (columns independent) | stays 0.807 |
| "which other colours exist?" | expected information gain of asking the shade of red, blue or green | 0.139 nats each |
| | the same for an unseen colour, if a spare slot exists | 0.303 nats |

What happens, in words:
1. First hearing. One count is added to a prior of about one half per cell, so the agent already
   expects red to be dark with 0.75 and gives "always dark" 0.65. The first hearing is not weak;
   it can be made weaker only by a learning rate or a heavier prior chosen by hand. [C]
2. Contradiction. Nothing is deleted; a count is added to the other cell. The belief about the
   next shade goes to 0.5 and the hypothesis "mixed" jumps to 0.81. One counter-example nearly
   settles it, because "always dark" tolerates exceptions only at 1/32. [C] This is the same
   behaviour art-growth.md reports for the hierarchical Bayesian model.
3. Rule. With a rule space over whole tensors, evidence from other colours raises the rule's
   posterior, and by model averaging raises red's. The feedback from rule to instance is real and
   needs no new observation of red. If red had only ever been heard as dark, two other mixed
   colours would still lift "red is mixed" from 0.333 to 0.942. [C] With plain Dirichlet learning
   and no rule space, nothing learned about blue changes red. [C]
4. Commit. The best rule passes 16 nats only at the 27th colour (5.2 nats at 10 colours, 16.5 at
   27), each colour having been heard dark once and light once. The agent then replaces its
   posterior by that rule, and if the counts are multiplied by 512 as the paper suggests, one
   further observation moves a belief by 0.0003. Graded revision stops there. [C] / [V]
5. Question. The agent computes the expected information gain of each available query and takes
   the largest. A colour it has never sampled wins. This is selection among given queries about
   given states: with no spare colour level the quantity does not exist, and the sentence "which
   other colours exist?" is never composed. [C] / [I] The nearest implemented case is SpCoAE,
   which decides where to ask its one fixed question, and whose Dirichlet process lets a new
   category appear in the answer. [V]

### 6.4 Where it departs from the owner's example
- The numbers run 0.65 → 0.81 → 0.98 → 0.995 against the owner's 0.23 → 0.57 → 0.77 → 0.88: the
  same direction, much steeper, and strong from the first sentence. [C]
- The owner's conviction belongs to a formulation that changes ("dark" becomes "dark but
  sometimes light"). Here nothing changes form: there is one distribution per colour from the
  start, and "refinement" is a shift of posterior mass between hypotheses the designer listed. [I]
- A rule of the "sometimes, sometimes" kind is never confirmed sharply. For a single colour the
  posterior of "mixed" saturates at 0.84, about 1.6 nats of log odds, however many observations
  arrive (0.833 after 10, 0.839 after 10,000), because the shrinkage prior for "always dark" is
  itself a Dirichlet that adapts. A never-contradicted "always dark" is also slow: 0.836 after 10
  confirmations, 0.938 after 100. The commit is fast only among sharp rivals that the data
  contradict, which is what the three-ball task consists of: there the rule was selected at
  trial 18 in the example run. [C] / [V]
- This is by design. The model space is built from the requirement of maximum mutual information
  between states and outcomes, for which "each column of the likelihood tensor should have a
  unique nonzero entry". [V] (arXiv:2512.21129) "All colours are sometimes dark, sometimes
  light" says that colour carries no information about shade, the opposite of what the prior
  rewards. In my rule space it wins as R_P, "shade does not depend on colour", and it wins over
  "each colour mixed at its own rate" by Occam's factor (0.699 against 0.280 at three colours,
  0.964 against 0.036 at seven). Under R_P red inherits a pooled rate: with red heard dark once
  and five other colours mixed, P(next red is dark) is 0.542 instead of 0.747. [C]
- After the commit the colour factor is redundant for shade, so curiosity about the shade of
  further colours falls to nothing. The owner's question comes after the generalisation and is
  about the extent of the category; this agent's questions come before it and are about cells
  of a table. [I]

### 6.5 What the neural systems of the field do on the same stream
All [C] unless tagged; each is the mechanism of a row reduced to one belief.

| System | After "red is dark" | After "red is light" | Rule level | Question |
|---|---|---|---|---|
| Binary HGF node (row 7; prior μ=0, π=1, tonic volatility −2) | P(dark) 0.609, precision 1.13 | P(dark) 0.486, precision 1.22 | only if a shared parent node is wired by hand | none |
| One error-driven weight, learning rate 0.23 (rows 1–4) | 0.230 | 0.177 | none; shared weights generalise implicitly | none |
| Conjugate Gaussian weight (rows 5, 6) | mean 0.500, precision 2 | mean 0.333, precision 3 | none | none |
| Associative memory read-out (rows 18–20) | dark 1.0 | dark 0.5, light 0.5 | a prototype attractor once examples are many; the instances stop being retrievable [A] / [I] | none |
| Hebbian assembly (row 23) | co-active synapses ×1.05 [V] | no mechanism for a contradiction [I] | noun and verb areas only [V] | none |

Three readings. (1) An error-driven weight can reproduce the weak first hearing exactly, by its
learning rate, and then does the opposite of the owner's second step: the contradiction lowers
the number. (2) The Bayesian weight's precision rises with every observation, agreeing or not,
which resembles conviction growing through contradiction, but its content is only a mean. (3)
The memory stores the first hearing at full strength and holds the contradiction as a mixture.
None forms a rule that supports the instance, and none asks. [I]

## 7. What is still unsolved

### 7.1 In the field's own words
- Scale: "one of the main open problems in the field, scalability". [V] (arXiv:2407.01163)
  "there is still a significant amount of research to be performed to be able to make PC match
  the performance of BP on larger architectures, such as language models". [V] (arXiv:2308.07870)
- Method: "what are the dropout, batch norm, and Adam optimizer equivalents for PC?" [V] (arXiv:2308.07870)
- Efficient inference: faster schemes or amortisation for the relaxation loop. [V] (arXiv:2308.07870; arXiv:1905.10696)
- Uncertainty over weights and growth of structure in a neural PC system, stated as future work:
  replace fixed-dimension reductions with nonparametric priors so that prediction errors
  "simultaneously (i) adjust beliefs about existing components and (ii) signal when residual
  error warrants 'breaking a new stick' (creating a component) or dropping a redundant one",
  a system "that learns both its parameters and its effective dimensionality online". [V]
  (arXiv:2308.07870) One search for a predictive coding network that grows components online
  through such a prior found none. [A]
- Forward-forward: whether positive and negative phases can be separated in time; "Can FF
  produce a generative model of images or video that is good enough to create the negative data
  needed for unsupervised learning?" [V] (arXiv:2212.13345 §5, §10)
- Equilibrium propagation: time-varying inputs, learnable embeddings, asymmetric weights. [V] (arXiv:2209.09626)
- Active inference: discovering the core priors instead of engineering them [V]
  (arXiv:2505.24784); specifying the model space [V] (arXiv:2512.21129); learning the generative
  model from raw input before any rule learning, which the reasoning paper assumes done [V];
  "where do the requisite curricula come from?" [V] (arXiv:2311.10300); expected free energy over
  continuous trajectories, which "to our knowledge, has not yet been addressed fully" [V]
  (arXiv:2311.10300); language [V] (arXiv:2407.20292); evaluation on machine-learning benchmarks
  [A] (arXiv:2608.09512).
- Consolidation: how fast Hebbian state becomes long-term weight; "we do not provide a direct
  answer". [V] (arXiv:2509.26507)
- Online-continual learning with local rules that also generalises. [V] (arXiv:2307.15040)
- Sequences in a continually written PC memory. [V] (arXiv:2205.09930)
- Abstract words, function words and hierarchy in Hebbian language acquisition. [V] (arXiv:2507.11788)
- How many examples a network needs before a concept emerges: "still an open problem in machine
  learning". [A] (arXiv:2109.00454)

### 7.2 Not solved because the field has not posed it
All [I]; "not found" means not found in 26 searches and the sources read.
1. Conviction as something other than a probability. The field has probability of an outcome,
   precision of a belief and posterior of a hypothesis; none is attached to a formulation that
   is rewritten when contradicted and gains strength by being rewritten.
2. Rule to instance feedback inside an online network with weights. It exists as model averaging
   and as empirical Bayes in the Bayesian half, over a fixed model space, and as nothing more
   than shared weights in the neural half.
3. Rules that are not sparse. The structure priors in use reward deterministic mappings; a rule
   of the form "every member varies" is expressible only as the absence of a dependency (§6.4).
4. A weak first hearing as a property of the system. It appears as a tuned learning rate or as a
   small Hebbian multiplier, not as a consequence of having heard something once.
5. Questions about what is not yet represented. Epistemic value is defined over existing states
   and given actions; expansion rules add a state after an unexplained observation arrives, they
   do not seek one.
6. Learned operators. The one source that addresses it denies it for curiosity ([V] row 9); the
   meta-learning line learns an update rule outside the system's life.
7. A reality measure on the system's own constructions as an online learning signal. The pieces
   exist separately: the measure (free energy, goodness, energy), the constructions (generation,
   replay, dreams), the contrast (negative phases). The combination in one continuously living
   system is where forward-forward's sleep experiment failed.
8. Language without a vocabulary table inside these networks. Only the two Hebbian systems
   (rows 23, 24) form their word units, one from hand-initialised sound and sense assemblies,
   the other without meaning.
9. One system with both halves: a local-learning network whose parameters are graded beliefs,
   that grows units, selects rules and asks. The roadmap names the first two parts as a
   direction (§7.1); no source read here proposes all of it.

## 8. Vocabulary bridge (the field's names for the owner's ideas)

| Owner's idea | Field's term | Where |
|---|---|---|
| conviction of a belief | precision; Dirichlet concentration (counts); posterior probability of a model | rows 7, 8–12 |
| first hearing stored weakly | learning rate as a ratio of precisions; `lr` on the count update | rows 7; arXiv:2201.03904 |
| contradiction refines | belief updating; accumulation of counts | rows 8–12 |
| inducing the category rule | structure learning; Bayesian model reduction; "insight", "aha moment"; Occam's razor | rows 8, 9 |
| rule supports the instance | empirical priors; parametric empirical Bayes; Bayesian model average | arXiv:1805.07092 |
| a concept formed when needed | model expansion; "grow a model"; spare slots; stick-breaking prior; neurogenesis | rows 10–12, 14, 19 |
| a category emerging from instances | prototype formation; archetype; metastable state; spurious state | rows 18, 20; §3 |
| the question arises | epistemic value; salience (states), novelty (parameters), information gain over models; curiosity | row 9 |
| imagination | planning as inference; counterfactual rollouts; sophisticated inference; replay | rows 9, 12, 24 |
| measuring how real | free energy as surprise; energy; goodness; discriminator | rows 1, 17, 22; PAD |
| sleep that improves the model | negative phase; unlearning; dreaming; offline model reduction | rows 8, 17; §3 |
| the consciousness that operates the network | precision control; attention as mental action (deep parametric active inference); hyperpriors; volatility | §3; row 7 |
| learning and running as one | continual weight updates; incremental EM; online filtering; "learning becomes a kind of inference" | rows 2, 7, 11; arXiv:2005.04168 |

## 9. Sources

### Read in full text or in the cited parts ([V])
arXiv: 1805.07092, 1810.07411, 1905.10696, 2011.12216, 2201.03904 (appendix on learning),
2202.09467, 2205.09930, 2207.06415 (introduction), 2209.09626, 2211.03481, 2211.10934 (method
overview), 2212.00720, 2212.13345, 2305.10937 (update equations, binary nodes), 2307.15040,
2308.07870 (limitations, future directions), 2311.10300, 2312.09257 (CHL, discussion),
2407.01163, 2407.20292 (structure learning, music, conclusion), 2409.15729 (introduction,
conclusions),
2503.24016 (discussion), 2505.13124 (introduction, discussion), 2505.24784, 2506.23293
(introduction, §5), 2507.02092, 2507.11788, 2509.26507 (training, §7.2, lifelong learning
remarks), 2512.21129 (whole paper).

### Abstract page only ([A]), arXiv
1512.05463, 1602.05179, 1606.01164, 1804.02464, 1810.12217, 1903.08689, 1909.10863, 2001.07203,
2004.08128, 2004.11714, 2005.04168, 2006.03824, 2006.04120, 2006.04182, 2006.09549, 2006.14800,
2007.02686, 2008.02217, 2008.06996, 2012.03405, 2012.14905, 2104.01677, 2107.12979, 2108.02189,
2109.00454, 2109.04261, 2109.08063, 2201.13180, 2202.04557, 2204.00619, 2207.12316, 2209.00530,
2209.11883, 2210.13461, 2210.16414, 2301.01452, 2302.07253, 2303.18187, 2305.11982, 2306.15364,
2307.04205, 2311.09589, 2312.15103, 2407.04117, 2408.16429, 2409.00102, 2409.11593, 2412.10425,
2501.00226, 2501.03176, 2503.24322, 2505.21777, 2508.05619, 2508.15989, 2512.00619, 2602.01393,
2604.16839, 2608.09512, 2609.12079. With a skim of the text: 2309.04106, 2407.03342, 2503.02057.

### Abstract or snippet only ([A]), not on arXiv
- Friston, Lin, Frith, Pezzulo, Hobson, Ondobaka 2017, Active Inference, Curiosity and Insight, Neural Computation 29(10):2633–2683.
- Smith, Schwartenbeck, Parr, Friston 2020, An Active Inference Approach to Modeling Structure Learning: Concept Learning as an Example Case, Frontiers in Computational Neuroscience.
- Friston, Parr, Yufik, Sajid, Price, Holmes 2020, Generative models, linguistic communication and active inference, Neuroscience & Biobehavioral Reviews.
- Sandved-Smith, Hesp, Mattout, Friston, Lutz, Ramstead 2021, Towards a computational phenomenology of mental action, Neuroscience of Consciousness.
- Song et al. 2024, Inferring neural activity before plasticity as a foundation for learning beyond backpropagation, Nature Neuroscience.
- Hopfield, Feinstein, Palmer 1983, 'Unlearning' has a stabilizing effect in collective memories, Nature 304.
- Taniguchi 2024, Collective predictive coding hypothesis: symbol emergence as decentralized Bayesian inference, Frontiers in Robotics and AI.
- van Erp, Nuijten, de Vries 2024, Online Structure Learning with Dirichlet Processes through Message Passing (workshop paper).
- Neacsu, Mirza, Adams, Friston 2022, Structure learning enhances concept formation in synthetic Active Inference agents, PLOS ONE.

### Search log (26 queries; number 21 in extended mode)
1 predictive coding networks survey, local learning; 2 predictive coding language model without
backpropagation; 3 active inference language model, online learning, concept formation; 4
equilibrium propagation scaling, language, sequence learning; 5 forward-forward follow-ups,
language, limitations; 6 active inference structure learning, concept learning, Bayesian model
reduction, spare states; 7 "Active inference, curiosity and insight"; 8 modern Hopfield, dense
associative memory, online continual, local rule; 9 Hebbian learning, language acquisition,
neural assemblies; 10 predictive coding continual learning, prospective configuration; 11
Hopfield concept emergence, prototypes, dreaming; 12 forward-forward for text; 13 Energy-Based
Transformers; 14 "Dragon Hatchling"; 15 linguistic communication and active inference, twenty
questions; 16 adversarial dreaming, reality monitoring; 17 AXIOM expanding object-centric
models; 18 meta-learning plasticity rules; 19 deep parametric active inference, meta-awareness;
20 collective predictive coding, symbol emergence; 21 language models trained with predictive
coding, equilibrium propagation or local rules, 2026; 22 Hopfield unlearning, 1983; 23 active
inference, grammar and syntax learning; 24 associative memory language model, Hebbian storage;
25 renormalising generative models, language, 2026; 26 predictive coding with nonparametric
growth of components.

### Files in this directory
- `predictive-energy.md`: this note.
- `tools-predictive-energy/red_example_predictive_energy.py`: the computation behind §6 (pure
  Python; run with `python3 -I`).
- `tools-predictive-energy/fetch_arxiv.sh`, `absmeta.py`: download and abstract-listing helpers;
  `fetch_batch*.log`: download logs.
- `sources-predictive-energy/<arXiv id>/`: `abs.html` and `paper.txt` for each paper named above
  (the PDFs were deleted after conversion to save space; the fetch script restores them).
