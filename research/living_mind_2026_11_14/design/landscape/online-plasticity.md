# Landscape: learning while running (fast weights, plasticity, test-time training, online continual learning), read against P1–P6

Field covered: fast weights and Hebbian plasticity inside the forward pass, differentiable and
meta-learned plasticity rules, Hebbian/plastic Transformers, test-time training and test-time
adaptation (including In-Place TTT), online continual learning without task boundaries, streaming
learning, belief-valued (Bayesian) weights learned online, lifelong language learning and the
writing of knowledge into deployed language models, and systems or programmes that state they have
no train/eval split.

Date of this note: 2026-11-14. 33 web searches (2 in extended mode). 118 arXiv papers downloaded
into `sources-online-plasticity/` (index: `tools-online-plasticity/source_index.txt`); 41 of them
were opened in the text, and what I cite from that text carries [V]; the rest were read as
abstracts only.

## Summary

Tags are explained in §0; each line is supported in the section named.

- The field splits into three islands that nobody has joined (five families in §1). (i) Rules
  that learn inside one episode or one document and are then reset (fast weights, plastic
  networks, the LLM-scale test-time-training models of rows 9–13). (ii) Learners that never stop
  (streaming RL, online continual learning, continual backpropagation), all small and without
  language. (iii) Weights that are beliefs (online Bayesian learners), small, with no rule
  level. [I] (§1, §2.2)
- Of 25 systems in the main table none is native on P2, P3, P4 or P5, and none is native on both
  P1 and P6: the six that learn at every step for ever are not language systems, and every language
  system has a training phase, a reset, or both. [I] (§2.2)
- The field's own verdict on LLM test-time training: "the acquired in-context knowledge does not
  persist once the current context is removed". In-Place TTT resets its fast weights "at document
  boundaries"; TTT-E2E restarts every context from meta-learned weights. [V] (§2.1 rows 11–13)
- Every learned learning rule found here was learned in a separate outer loop and frozen. The
  newest blueprint for autonomous learning (March 2026) proposes that the controller of learning
  be "hardwired". Inside one life the stream itself adapts only a few scalars of the rule (step
  sizes, decay rates), shown on linear or last-layer problems. [V]/[A] (§4.6)
- Closest to the "red" example: online Bayesian weights (the MESU family; exact form a Kalman
  filter). Fitted to the owner's first value only, the exact filter gives conviction 0.230 for
  "red is dark", 0.374 after the contradiction (mean moves to "as much dark as light"), 0.479 for
  the colour weight after three other colours, and red rises to 0.461 without being mentioned
  again; a colour never met gets 0.240. The owner's values are 0.23, 0.57, 0.77, 0.88. [C] (§5.2)
- What that system cannot do: change the form of a belief, be told a rule, create the category
  unit, represent "sometimes" as content, ask anything. Kept exactly it ends unable to learn
  ("catastrophic remembering"); with MESU's forgetting window the belief is gone after 100
  unrelated steps (conviction 0.004 at N = 20). [C]/[V] (§5.2)
- The strongest working language system (TTT-E2E, In-Place TTT) would process the three sentences
  of the example without using a single changed weight: updates happen per 512–1024 tokens, the
  sentences stay in the attention window, and the document boundary discards everything. [V]/[I]
  (§5.3)
- A point estimate cannot tell "never heard" from "heard both": online SGD gives p(dark|red) =
  0.473 after "dark" then "light" and 0.487 for a colour never met; the Bayesian learner gives
  the same mean with conviction 0.374 against 0.093. [C] (§5.4)
- What broke: stability (collapse, gradient explosion, norm growth), forgetting (1% of earlier
  facts answered after twenty one-at-a-time writes into an LLM; locate-and-edit methods collapse
  within a few hundred edits), loss of plasticity, scale (learned plasticity rules and Bayesian
  weights stay on small image and control benchmarks), and language (facts written into weights
  do not propagate to their consequences; "the model can state the fact but fails to use it
  correctly"). [V]/[I] (§4)
- Unsolved in the field's own terms: persistent test-time learning without resets, forgetting
  ("not 'solved' in general"), plasticity over an unbounded life, learning the learning rule
  within one life, propagation of learned facts, selecting what is worth learning, belief-valued
  weights at scale, evaluation without a train/test split. [V]/[A] (§6.1)
- Not posed at all, as far as 33 searches show: a conviction attached to a statement, a
  contradiction that changes the form of a belief, a told rule that supports an instance, a
  question that arises from a category's unknown members, and an operating procedure that is
  itself learned in the same life and carried to a new domain. [I] (§6.2)

## 0. How to read this note

### Status tags (every statement carries one)

| Tag | Meaning |
|---|---|
| [V] | VERIFIED: I read the source text itself (PDF from arxiv.org) for this statement. Quotation marks mean the words are the source's. |
| [A] | ABSTRACT: abstract page or search-result snippet only. |
| [M] | MEMORY: from my own knowledge, not checked in this session. |
| [C] | COMPUTED here: output of `tools-online-plasticity/red_example_online_plasticity.py` (saved in `red_example_output.txt`) or arithmetic shown in the text. Not a literature claim. |
| [I] | My own inference or judgement from the tagged facts. Not a literature claim; treat as opinion. |

### The principles, as I applied them (all codes are [I])

| | N (native) | P (partial) | 0 |
|---|---|---|---|
| P1 learning = running | long-lived weights change on every experience; the method needs no separate phase and no reset | weights change while running, but only a fast subset, or they are reset at a sequence/episode/document boundary, or a separate phase trained the rest | weights fixed while running |
| P2 graded beliefs | weak first storage, growth with evidence, refinement by contradiction, and a category level that supports instances | some of these | none |
| P3 concepts formed | units for new concepts are created when needed; categories emerge from instances | units can be added or recycled online, meaning still comes from a fixed encoder or table | fixed vocabulary and encoder |
| P4 imagination measured | the system generates constructions, scores them from outside against experience, and learns by that score | it learns from its own generations with some score | none, or self-training with no score |
| P5 way of constructing | the procedure of learning is changed by experience during deployment and transfers across domains | the procedure is learned, but in a separate outer loop and then fixed; or it self-modifies and transfer is not shown | fixed hand-written rule |
| P6 engine, language | tensor engine demonstrated on natural language | tensor engine, no natural-language demonstration | not a tensor engine |

"Evaluated with a held-out test set" does not by itself cost a system its P1 code; the code is
about what the method needs. [I]

### What could not be reached

`ojs.aaai.org` and `www.cs.cmu.edu` did not resolve from this session (outside the allowed network
domains), so the NELL paper was not read; it is cited from a search snippet and memory. The Nature
paper on loss of plasticity was read in its arXiv version (2306.13812). (Facts about this
session, not literature claims.)

## 1. The field in one picture

Five families, by where each one puts the line between learning and running. [I]

| Family | What changes while the system runs | What was fixed beforehand | What happens at the end of a sequence | Examples (rows in §2) |
|---|---|---|---|---|
| A. Plasticity inside the forward pass | a designated fast weight matrix or Hebbian trace, at every step | the slow weights and the plasticity rule, by gradient descent or evolution | fast state returns to zero (or to random) | 1–6, 14 |
| B. Test-time training in sequence models | a small inner model or some MLP blocks, by gradient steps per token or per chunk | everything else, including the starting point of the inner model | back to the trained starting point (stated in rows 9, 12, 13; rows 10 and 11 are evaluated per sequence) | 9–13 |
| C. Never-stop learners | all weights, on every sample or mini-batch, for ever | at most an initialisation; sometimes a frozen lower network | nothing: there is no end | 18–22 |
| D. Belief-valued weights | mean and uncertainty of every weight, on every sample or mini-batch | the prior | nothing | 23–25 |
| E. Writing into a deployed language model | all or selected weights, per document, per fact or per self-edit | the pre-trained model (and, for SEAL, the policy that writes self-edits) | weights are kept; forgetting follows | 8, 15, 16, 20 and §3 |

Families A and B answer "can a network learn within one context?" and reset. Families C and D
answer "can a network learn for ever?" and stay small. Family E is where the two meet a language
model, and it is where most of the failures in §4 were measured. [I]

## 2. The 25 most relevant works

Status: V = I read the method and limitation passages; A = abstract only. P-codes are [I].

| # | Work | First author | Year | Venue / arXiv | Status | P1 | P2 | P3 | P4 | P5 | P6 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Using Fast Weights to Attend to the Recent Past | Ba | 2016 | NeurIPS 2016 [M]; arXiv:1610.06258 | V | P | P | 0 | 0 | 0 | P |
| 2 | Differentiable plasticity: training plastic neural networks with backpropagation (with the follow-up Backpropamine) | Miconi | 2018 | ICML 2018; arXiv:1804.02464 (ICLR 2019; 2002.10585) | V | P | P | 0 | 0 | P | P (N for the follow-up) |
| 3 | Linear Transformers Are Secretly Fast Weight Programmers | Schlag | 2021 | ICML 2021 [M]; arXiv:2102.11174 | V | P | P | 0 | 0 | P | N |
| 4 | A Modern Self-Referential Weight Matrix That Learns to Modify Itself | Irie | 2022 | ICML 2022; arXiv:2202.05780 | V | P | P | 0 | 0 | P | P |
| 5 | Meta-Learning through Hebbian Plasticity in Random Networks | Najarro | 2020 | NeurIPS 2020; arXiv:2007.02686 | V | P | 0 | 0 | 0 | P | P |
| 6 | Meta Learning Backpropagation And Improving It (VSML) | Kirsch | 2021 | NeurIPS 2021; arXiv:2012.14905 | V | P | 0 | 0 | 0 | P | P |
| 7 | Meta-Learning Representations for Continual Learning (OML; with ANML) | Javed | 2019 | NeurIPS 2019; arXiv:1905.12588 (2002.09571) | V (ANML A) | P | 0 | 0 | 0 | P | P |
| 8 | Dynamic Evaluation of Neural Sequence Models (with its use on document streams) | Krause | 2018 | ICML 2018 [M]; arXiv:1709.07432 (2102.01951, 2403.01518) | V | P | P | 0 | 0 | 0 | N |
| 9 | Learning to (Learn at Test Time): RNNs with Expressive Hidden States | Sun | 2024 | arXiv:2407.04620 | V | P | 0 | 0 | 0 | P | N |
| 10 | Titans: Learning to Memorize at Test Time | Behrouz | 2024 | arXiv:2501.00663 | V | P | P | 0 | 0 | P | N |
| 11 | Nested Learning: The Illusion of Deep Learning Architectures (Hope) | Behrouz | 2025 | NeurIPS 2025; arXiv:2512.24695 | V | P | P | 0 | 0 | P | N |
| 12 | End-to-End Test-Time Training for Long Context (TTT-E2E) | Tandon | 2025 | arXiv:2512.23675 | V | P | P | 0 | 0 | P | N |
| 13 | In-Place Test-Time Training | Feng | 2026 | ICLR 2026 (oral); arXiv:2604.06169 | V | P | 0 | 0 | 0 | P | N |
| 14 | Enabling Robust In-Context Memory and Rapid Task Adaptation in Transformers with Hebbian and Gradient-Based Plasticity | Chaudhary | 2025 | arXiv:2510.21908 | V | P | P | 0 | 0 | P | P |
| 15 | Self-Adapting Language Models (SEAL) | Zweiger | 2025 | arXiv:2506.10943 | V | P | 0 | 0 | P | P | N |
| 16 | Deductive Closure Training of Language Models for Coherence, Accuracy, and Updatability (DCT) | Akyürek | 2024 | ACL Findings 2024; arXiv:2401.08574 | V | 0 | P | 0 | P | 0 | N |
| 17 | Tent: Fully Test-time Adaptation by Entropy Minimization (with CoTTA, SAR) | Wang | 2021 | ICLR 2021; arXiv:2006.10726 (2203.13591, 2302.12400) | A (SAR V) | P | 0 | 0 | 0 | 0 | P |
| 18 | Task-Free Continual Learning | Aljundi | 2019 | CVPR 2019; arXiv:1812.03596 | V | N | P | 0 | 0 | 0 | P |
| 19 | REMIND Your Neural Network to Prevent Catastrophic Forgetting | Hayes | 2020 | ECCV 2020; arXiv:1910.02509 | V | P | 0 | 0 | 0 | 0 | P |
| 20 | Episodic Memory in Lifelong Language Learning (MbPA++) | de Masson d'Autume | 2019 | NeurIPS 2019; arXiv:1906.01076 | V | P | 0 | 0 | 0 | 0 | N |
| 21 | Loss of plasticity in deep continual learning (arXiv version: Maintaining Plasticity in Deep Continual Learning) | Dohare | 2024 | Nature 632 [A]; arXiv:2306.13812 | V | N | 0 | P | 0 | 0 | P |
| 22 | The Alberta Plan for AI Research (with Stream-X, 2410.14606) | Sutton | 2022 | arXiv:2208.11173 | V (Stream-X A) | N | 0 | P | P | P | P |
| 23 | Task Agnostic Continual Learning Using Online Variational Bayes (BGD) | Zeno | 2018 | arXiv:1803.10123 | V | N | P | 0 | 0 | 0 | P |
| 24 | Bayesian continual learning and forgetting in neural networks (MESU) | Bonnet | 2025 | arXiv:2504.13569 | V | N | P | 0 | 0 | 0 | P |
| 25 | Low-rank extended Kalman filtering for online learning of neural networks from streaming data (LO-FI) | Chang | 2023 | CoLLAs 2023; arXiv:2305.19535 | V | N | P | 0 | 0 | 0 | P |

### 2.1 Mechanism in two sentences, and the reason for each code

**1. Fast weights (Ba 2016).** A recurrent network carries a second weight matrix that is rewritten
at every step by A(t) = λA(t−1) + η h(t)h(t)ᵀ (the paper's defaults: η = 0.5, λ = 0.95), and that
pulls the hidden state toward recent states in a short settling loop. The ordinary weights are
trained by gradient descent with this memory in the loop. [V]
P1 P: the fast matrix changes at every step while running, but it fades within tens of steps and
the slow weights are trained beforehand. P2 P: strength grows with repetition and fades; a
contradiction is superposed and cancels (trace in §5.4: 0.500 then −0.025). P5 0: the rule and its
two constants are set by hand. P6 P: tasks are associative retrieval, MNIST, facial expressions
and the game Catch. [V]/[C]/[I]

**2. Differentiable plasticity (Miconi 2018).** Each connection has a fixed weight w and a Hebbian
trace that is a running average of pre- times post-synaptic activity; the effective weight is
w + α·Hebb, and w, the per-connection coefficient α and the trace rate η are trained by gradient
descent between episodes. The trace "is initialized to zero at the beginning of each
lifetime/episode: it is purely a lifetime quantity"; in the 2019 follow-up the network emits a
signal M(t) that gates its own trace update. [V]
P1 P: reset per episode, structure trained beforehand. P2 P: graded trace, no contradiction
handling. P5 P: how plastic each connection is, is learned, but between lifetimes. P6: the
follow-up reports about 1.7 perplexity points gained by a plastic LSTM on Penn Treebank. [V]/[I]

**3. Fast weight programmers (Schlag 2021).** Linear attention is shown to be the same computation
as a slow network that emits keys and values and adds value⊗key outer products into a fast weight
matrix, read out by a query. The paper replaces the additive write by a delta rule
W ← W + β(v − W k)⊗k whose write strength β the slow network also emits, so that a stored
association can be corrected; a matrix with key dimension d cannot hold more than d associations
without retrieval error. [V]
P1 P: the fast matrix starts at zero for each sequence. P2 P: the delta rule replaces a stored
value "to which extent" β says, which is a refinement, but there is no count of evidence (§5.4).
P5 P: the slow network learns how to write, including "dynamically changing learning rates",
during training only. P6 N: WMT14 translation and WikiText-103. [V]/[I]

**4. Self-referential weight matrix (Irie 2022).** One weight matrix produces from the current
input its output, a key, a query and learning rates, and then rewrites itself by
W ← W + σ(β)(Wφ(q) − Wφ(k))⊗φ(k). "The initial values of the SRWM W₀ are the only parameters in
this layer which are trained by gradient descent"; every part, including the part that produces
the update, changes at run time. [V]
P1 P: self-modification runs during use; W₀ comes from training and state is reset at episode
boundaries. P5 P: this is the nearest thing in the field to a learning procedure that modifies
itself, but the ability is acquired in meta-training, the authors note that "in the long run,
there is no guarantee that the SRWM's self-modifications preserve the original objective function
it is trained for", and transfer across domains is not shown. P6 P: few-shot image
classification and procedurally generated games. [V]/[I]

**5. Hebbian plasticity in random networks (Najarro 2020).** Weights start random in every lifetime
and change at every time step by a per-synapse rule Δw = η(A·pre·post + B·pre + C·post + D); the
coefficients, not the weights, are found by an evolution strategy. No reward or error is given
during the lifetime, and a simulated quadruped self-organises in under 100 time steps and adapts
to damage not seen in training. [V]
P1 P: inside a lifetime learning and running are one process, but nothing is kept from one
lifetime to the next and the rule is fixed. P5 P: the rule is found offline. [I]

**6. VSML (Kirsch 2021).** Every weight of a network is replaced by a tiny LSTM with shared
parameters, with messages passed forwards and backwards, so that running the network forward on
inputs and errors is a learning algorithm. The shared parameters are meta-trained; "during meta
testing, no explicit gradient descent is used", and an algorithm meta-trained on MNIST also learns
Fashion-MNIST. [V]
P1 P: at meta-test time learning is literally the forward pass; the algorithm itself is frozen.
P5 P: the one clear case of a learned learning algorithm carried to another dataset, on small
image sets. [I]

**7. OML and ANML (Javed 2019).** A representation network is meta-trained so that plain online
SGD on a small prediction network on top of it interferes little; the representation is "fixed
at meta-test time" and "the online updates for continual learning also only modify the PLN" (the
prediction network). ANML adds a meta-trained gating network and reports sequential learning of
up to 600 classes. [V]/[A]
P1 P: online learning of the head only. P5 P: a representation learned for the purpose of online
learning, frozen afterwards. [I]

**8. Dynamic evaluation (Krause 2018).** While a language model is evaluated, each segment it has
just predicted is used for one gradient step on all weights, with a term that pulls the weights
back to their trained values: θ ← θ − η∇L(s) + λ(θg − θ). Used across a time-ordered stream of
documents it "reduces the speed of the model becoming outdated" and lowers perplexity on newly
emerging words by 39.62%, but "it causes catastrophic forgetting of the past data"; in a 2024 LLM
study the best results reset the weights at each book boundary. [V]
P1 P: the same weights that predict are trained by every segment, which is the plainest form of
"learning while running" on language; a trained starting point, a decay toward it, and resets
remain. P2 P: one small step is weak first storage, but the stored thing is a point estimate
(§5.4). P5 0: the update rule is fixed and tuned on validation data. [V]/[I]

**9. TTT layers (Sun 2024).** The hidden state of a sequence layer is the weight matrix W of a
small model (linear, or a two-layer MLP), and reading a token means one gradient step on W of a
self-supervised reconstruction loss, at training time and at test time. The surrounding network,
the starting point W₀ and the self-supervised task itself are trained in an outer loop. [V]
P1 P: W is retrained on every test sequence from W₀. P5 P: what the inner model should learn is
itself learned, outside. P6 N: language modelling at 125M to 1.3B parameters. [V]/[I]

**10. Titans (Behrouz 2024).** A small MLP is a long-term memory whose weights are updated at test
time by M ← (1 − α)M + S with S ← ηS − θ∇‖M(k) − v‖², i.e. gradient descent on an associative
loss with momentum ("past surprise") and a forgetting gate, the three gates being functions of
the input. At test time "persistent memory parameters are fixed", attention works in context, and
"the long-term memory module is still learning". [V]
P1 P: only the memory module learns, per sequence. P2 P: the size of a write is the size of the
surprise, so what is new or contradicting is written hardest, the reverse of "a first hearing is
stored weakly" (§5.4: surprise 1.0 at the first hearing, 1.5 at the contradiction). P5 P: gates
are learned in the outer loop. [V]/[C]/[I]

**11. Nested Learning and Hope (Behrouz 2025).** A model is described as nested optimisation
problems with different update frequencies; Hope combines a "continuum memory system" (MLP blocks
updated every so many tokens, at several frequencies) with a Titans variant that generates its own
keys, values and learning rates. The paper states that "for a neural learning module, there is no
boarder and clear distinction between training and test time", keeps pre-training as the
lowest-frequency level, and says forgetting "is not 'solved' in general … a natural consequence
of compression". [V]
P1 P: the position is P1; the experiments start from Llama-3 backbones with "continual
pre-training with 15B tokens" and test class-incremental text classification and learning two
new languages in context. P2 P: several timescales give graded persistence, not graded belief.
P5 P: a module that produces its own update, learned in training. [V]/[I]

**12. TTT-E2E (Tandon 2025).** A Transformer with sliding-window attention keeps training on the
test context by next-token prediction in mini-batches of 1K tokens, updating only the MLP layers
of the last quarter of its blocks; the weights from which this starts are meta-learned through
"gradients of gradients". Embedding, normalisation and attention layers are frozen at test time
"since updating them in the inner loop causes instability in the outer loop", and a second static
MLP is added as "'safe' storage for pre-trained knowledge". [V]
P1 P: the paper says "most human learning happens without a boundary between training and
testing" and uses the training loss at test time; each context still starts from W₀ and the
meta-learning phase is separate. P2 P: as row 8. P5 P: the starting point is learned so that
test-time learning works. P6 N: 3B-parameter models, 164B training tokens, 128K context. Limits
the authors give: full attention "dramatically outperforms" it on needle-in-a-haystack recall,
and training latency "is still a significant limitation". [V]/[I]

**13. In-Place TTT (Feng 2026).** The output projection of existing MLP blocks is treated as fast
weights and updated chunk by chunk during inference by W_down ← W_down + η V̂ᵀZ, where the target
V̂ is a learned projection of following-token embeddings and chunks of 512 to 1024 tokens work
best. "At document boundaries, the fast weights are reset to their pre-trained state to prevent
context leakage across independent sequences." [V]
P1 P: weights of the deployed LLM change during inference, then are reset. P2 0: the update is a
plain sum of outer products with no error term and no strength per association. P5 P: the target
projection is trained. P6 N: a 4B-parameter model, contexts up to 128k. [V]/[I]

**14. Plastic Transformers (Chaudhary 2025).** Decoder-only Transformers get fast weights that are
"initialized to zero at the start" of each sequence and updated at every step, either by a
neuromodulated Hebbian rule w ← (1 − η(t))w + η(t)α∘pqᵀ or by a gradient step toward a target
the model emits itself. Static weights, per-connection rates and the modulation signal are
meta-trained; the tasks are copying, regression and few-shot image classification. [V]
P1 P, P2 P (the Hebbian variant is "sharply gated around salient events"), P5 P, P6 P. [V]/[I]

**15. SEAL (Zweiger 2025).** Given a new passage, the language model writes a "self-edit" (for
example implications of the passage, or augmentation and optimiser settings) and is fine-tuned on
it, which gives "persistent weight updates". An outer reinforcement-learning loop rewards
self-edits after which the updated model answers held-out questions better. [V]
P1 P: weights change at deployment, but each self-edit evaluation takes "approximately 30–45
seconds" in training and sequential self-edits forget. P4 P: the model constructs implications of
what it read, the constructions are scored from outside by what the updated model can then answer,
and the constructing improves by that score; the score needs reference questions for every
passage. P5 P: what is learned is how to restructure knowledge for one's own update, in an outer
loop, in two separately trained domains. [V]/[I]

**16. Deductive Closure Training (Akyürek 2024).** For each seed statement the language model
generates statements implied by it and statements contradicting it, gives each a probability of
being true, and selects the most probable truth assignment that respects every implication and
contradiction. It is then fine-tuned on the statements inferred true; the seeds may come from a
trusted source or from the model itself. [V]
P1 0: a batch procedure. P2 P: truth is graded and the consequences of a statement are written
into the weights by generating and training on them, but the resolution is a hard true/false
assignment, so the less probable of two contradicting statements is dropped, not refined. P4 P:
own constructions, measured by the model's own truth estimate plus logical consistency, drive the
update; "only … a single layer of implications" was used. P5 0: the procedure is fixed prompts
and a fixed inference step. [V]/[I]

**17. Tent, CoTTA, SAR.** A trained classifier is adapted during testing, batch by batch and
without labels, by minimising the entropy of its own predictions while updating normalisation
statistics and affine parameters. CoTTA adapts all weights with averaged teacher predictions and
restores a random part of the source weights at each step; SAR removes high-gradient samples and
seeks flat minima because entropy minimisation ends in "collapsed trivial solutions, i.e.,
assigning the same class label for all samples". [A]/[V]
P1 P. P2 0: the objective raises confidence without evidence. P4 0: the model learns from its own
outputs with no check on them, which is the documented cause of the collapse. [I]

**18. Task-free continual learning (Aljundi 2019).** A network learns from a stream with no task
boundaries; a per-weight importance penalises changes to weights that matter, and it is
re-estimated whenever the loss reaches a plateau, with a small buffer of hard samples.
Demonstrations are face recognition learned by watching a television series and a robot learning
to avoid collisions. [V]
P1 N: the system "keeps on learning over time in a streaming fashion" with all weights. P2 P: a
strength per weight that grows with use and resists change, about importance, not about content.
[V]/[I]

**19. REMIND (Hayes 2020).** After an offline "base initialization" on the first 100 classes the
early layers are "no longer plastic"; the later layers learn one example at a time, each step
mixing the new example with old ones replayed from product-quantised feature maps. It runs on
ImageNet and on streaming visual question answering. [V]
P1 P: one example at a time, on top of a frozen lower network. [I]

**20. MbPA++ (de Masson d'Autume 2019).** A text model learns in a single pass over a stream of
examples with no dataset identifiers, storing examples in an episodic memory and replaying 100 of
them every 10,000 new examples. At inference it retrieves the 32 nearest stored examples, takes
30 gradient steps on them, predicts, and its "parameters are reset" afterwards. [V]
P1 P: both halves are forms of learning while running, and the second is thrown away after each
answer. P6 N: text classification and question answering. [I]

**21. Loss of plasticity, continual backpropagation (Dohare 2024).** Backpropagation run without
end loses the ability to learn: on continual ImageNet binary tasks accuracy "dropped from 89% …
on an early task down to 77%, about the level of a linear network, on the 2000th task", with dead
units, growing weights and falling representation rank. Continual backpropagation re-initialises
"a small fraction of less-used units after each example" and keeps learning. [V]
P1 N: an algorithm with no phases. P3 P: units are recycled into new random features and kept if
useful, which is feature search, not concept formation. [I]

**22. The Alberta Plan (Sutton 2022) and Stream-X.** A research plan for agents in which "there
are no special training periods" ("temporal uniformity"), and in which the meta-algorithms that
adapt per-weight step sizes and that create, assess and delete features, subtasks and options run
on every time step. Stream-X is the working piece found here: deep reinforcement learning from
each sample as it arrives, without a replay buffer, on 58 Atari games and 50 control tasks. [V]/[A]
P1 N as a principle, and N in Stream-X. P3 P, P4 P, P5 P as plan only: the steps on feature
creation, on planning with a learned model and on online meta-learning are a programme, and the
document is a plan, not a system. P6 P: no language. [V]/[I]

**23. BGD (Zeno 2018).** Each weight is a Gaussian with mean μ and standard deviation σ; the
posterior after a mini-batch is the prior for the next, with closed-form updates
μ ← μ − ησ²E[∂L/∂θ] and a matching update of σ. "The learning rate of μᵢ is proportional to the
uncertainty in the parameter", and no task boundary is used. [V]
P1 N. P2 P: this is the field's native form of "a weight is a graded belief": content and
certainty are separate numbers, certainty grows with evidence, and a contradiction moves the
content while certainty keeps growing. There is no statement, no rule level, and no question.
[V]/[I]

**24. MESU (Bonnet 2025).** The same Gaussian weights, with an update derived from Bayes' rule
over a window of the last N mini-batches: each mean moves in proportion to that weight's
variance, and extra terms pull mean and variance back to the prior at rate 1/N. The window
exists because pure accumulation "attempts to retain all information indefinitely, eventually
exhausting the capacity of the network. As new tasks are presented, its plasticity collapses"
("catastrophic remembering"). [V]
P1 N: "each new mini-batch can be treated as task t". P2 P: as row 23, plus an uncertainty that
separates in- from out-of-distribution inputs after 200 tasks, so the system has a measure of
what it has not met; and minus permanence, since evidence older than the window leaks away.
P6 P: permuted MNIST and CIFAR. [V]/[I]

**25. LO-FI (Chang 2023).** An extended Kalman filter over all network weights with the posterior
precision kept as diagonal plus low rank, so that one predict–update step per incoming sample
costs time linear in the number of weights. It is deterministic, needs no step size, lets weights
drift (decay γ, noise q), and is tested on Fashion-MNIST streams and a contextual bandit. [V]
P1 N: one sample, one update, no phases. P2 P: as row 23, and it keeps correlations between
weights, which is what lets evidence about a shared feature tighten an instance (§5.2). [V]/[I]

### 2.2 What the table says

- P1: native in 6 rows (18, 21–25), partial in 18, absent in 1. P2: native in 0, partial in 14.
  P3: native in 0, partial in 2. P4: native in 0, partial in 3. P5: native in 0, partial in 14.
  P6: native in 10 (11 with the follow-up of row 2). [I]
- No row is native on both P1 and P6. The six P1-native rows are vision, control or MNIST-scale
  systems; the ten language rows all reset, or freeze most of the model, or have a separate
  training phase. [I]
- The 14 partial P5 codes all mean the same thing: the rule was learned somewhere else and is
  then fixed (or, in row 4, self-modifies without evidence of transfer). [I]
- P2-partial comes in kinds that do not combine: a graded but transient trace (rows 1–4, 10, 14),
  a point estimate nudged by gradient steps (rows 8, 11, 12), a belief with separate content and
  certainty (rows 23–25), a consolidation strength per weight (row 18), and truth probabilities
  read off a language model (row 16). Only the third separates ignorance from a known mixture
  inside the weights (§5.4). [I]/[C]

## 3. Other works read (context)

One line each. Tag = how I know the statement.

Origins and neighbours of fast weights
- Hinton and Plaut 1987, fast weights to "deblur" old memories; Schmidhuber 1992, a slow net that learns to control fast weights; Bengio, Bengio and Cloutier 1991, learning a synaptic learning rule. [M]
- Sandler 2021, Meta-Learning Bidirectional Update Rules (2104.04657): neurons and synapses with several states, a shared low-dimensional "genome" meta-learned, no gradients at run time. [A]
- Munkhdalai 2019, Metalearned Neural Memory (1907.09720): memory is a network whose weights are rewritten in one shot. [A]
- Garcia Rodriguez 2022, Short-Term Plasticity Neurons (2206.14048): a state inside each synapse, trained through time. [A]
- Duan 2023, Hebbian and gradient-based plasticity in RNNs (2302.03235): Hebbian rules suit memory tasks; a gradient rule toward a "self-determined target" suits few-shot regression. [A]
- Money 2026, Hebbian fast weights in vision Transformers (2605.02920): one module after the last stage helps (+0.3 points, 1-shot Omniglot); per-block placement fails. [A]
- Dherin 2025 (2507.16003) and von Oswald 2022/2023 (2212.07677, 2309.05858): a forward pass with context equals a pass without it after a low-rank update of the MLP weights; the update is never executed or kept. Chan 2022 (2205.05055): in-context learning emerges from bursty, long-tailed data and "traded off against more conventional weight-based learning". [A]
- Clark 2022, Meta-Learning Fast Weight Language Models (2212.02475): dynamic evaluation expressed as linear attention so that it can be used in training. [A]
- Rae 2018, Fast Parametric Learning with Activation Memorization (1803.10049): output weights of rare classes written directly from activations. [A]
- Irie 2023, Metalearning Continual Learning Algorithms (2312.00276): self-referential matrices meta-learn an in-context continual learner; "directly scaling … to real-world tasks involving many more classes does not seem straightforward", with "challenges in terms of both domain and length generalization". [V]

Test-time training on language, around the main rows
- Hardt 2023 (2305.18466): fine-tune on 20 retrieved neighbours per test input, one step each. Akyürek 2024 (2411.07279): test-time training on the in-context examples of a task, "temporarily updating model parameters". Wang 2024, Temp-LoRA (2401.11504): a LoRA trained on text generated so far and "discarded post-generation". [A]
- Rannen-Triki 2024 (2403.01518): online adaptation "turns parameters into temporally changing states"; open items named: "automatic detection of reset points", and what weight memory holds as against context. [V]
- Hu 2023, CaMeLS (2305.15076): naive fine-tuning on a document stream is "drowned out by the gradient from inherently noisy tokens"; a meta-trained model re-weights tokens. [A]
- Tack 2024, MAC (2403.04317): gives up weight updates; documents become modulations in a memory bank beside a frozen model. Wang 2024, MemoryLLM (2402.04624): a fixed-size memory pool inside the model, "nearly a million memory updates" without degradation. [A]
- Lee 2026 (2605.26099) and Behrouz 2026 (2606.03979), "sleep": context periodically converted into fast weights by offline passes; distillation of short-term into long-term parameters with self-generated rehearsal data. [A]

Never-stop learners and their evaluation
- Li 2016, Learning without Forgetting (1606.09282) and Kirkpatrick 2016, EWC (1612.00796): keep old behaviour by distillation on new data, or by slowing important weights; both work between tasks, not online. [A]
- Aljundi 2019, MIR (1908.04742): replay the stored samples the next update would hurt most. [A]
- Elsayed 2024, UPGD (2404.00781): small changes to useful units, large perturbations to useless ones, against both forgetting and plasticity loss. Dohare 2021 (2108.06325): the first report of plasticity loss. [A]
- Degris 2024 (2401.17401): IDBD-style meta-gradient step sizes improve where RMSProp and Adam do not, "on simple problems". Ramesh 2026, FADE (2604.27063): per-parameter decay rates adapted online, derived for the linear case and applied to the last layer. Finn 2019, Online Meta-Learning (1902.08438). [A]
- Hammoud 2023 (2305.09275): "blind classifiers, which do not use input images" reach high online accuracy by exploiting label correlations in the stream. Ghunaim 2023 (2302.01047): when the stream does not wait for training, a simple baseline beats the published methods. Mesbahi 2024 (2404.02113): tuning hyper-parameters over the agent's whole lifetime hides which algorithms can learn continually. Bornschein 2022 (2210.07931): prequential (predict, then learn) description length as the objective. [A]
- Prabhu 2023 (2305.09253): a kNN on a fixed pre-trained extractor outperforms online continual learners by over 20% on two large streams. [A]
- Lyle 2023 (2303.01486): plasticity loss is tied to the curvature of the loss landscape and "often occurs in the absence of saturated units". Ash 2019 (1910.08475): a network warm-started on earlier data generalises worse than one trained from fresh weights. [A]

Belief-valued weights
- Aitchison 2021, Synaptic plasticity as Bayesian inference (1410.1029; Nature Neuroscience): synapses keep "error bars" and "more uncertain weights hav[e] higher learning rates". [A]
- Nguyen 2018, VCL (1710.10628); Ebrahimi 2019, UCB (1906.02425); Jones 2024, BONG (2405.19681); Titsias 2023, Kalman filter on the last layer for non-stationary classification (2306.08448). [A]
- Laborieux 2020, metaplastic binarised networks (2003.03533): the hidden real value behind a binary weight acts as consolidation strength. Kaplanis 2018 (1802.07239): Benna–Fusi multi-timescale synapses in RL agents. [A]

Writing knowledge into a deployed language model
- Jang 2021, Continual Knowledge Learning (2110.03215): "parameter expansion is necessary to reliably retain and learn knowledge simultaneously". [A]
- Hartvigsen 2022, GRACE (2211.11031): edits go into a codebook "without altering model weights". Wang 2024, WISE (2405.14768): reliability, generalisation and locality "can not be realized together" when editing in place; edits go to a side memory. [A]
- Gupta 2024 (2401.07453): sequential ROME/MEMIT edits forget gradually, then abruptly. Thede 2025, WikiBigEdit (2503.05683): on 500K real edits ROME, R-ROME and MEMIT "rapidly degrad[e] within the first few hundred updates, leading to model collapse"; retrieval "vastly outperforms" editing, and plain continual fine-tuning with merging beats editing at equal inference cost. Liu 2026 (2602.02543) and Ma 2026 (2605.11836): the collapse is a norm feedback loop; normalisation extends the horizon. [V]/[A]
- O'Neill 2026 (2607.11020): twenty invented facts written one at a time into Qwen3; afterwards 1% of questions on earlier facts are answered when each write used the statement repeated, 46% when it used 24 varied restatements; orthogonalising the updates does not help; "the model can state the fact but fails to use it correctly". [V]
- Onoe 2023 (2305.01651): fine-tuned-in facts show "little propagation"; gains only "when there is lexical overlap". Cohen 2023 (2307.12976): editors "fail to introduce consistent changes"; in-context editing scores best. Berglund 2023 (2309.12288): trained on "A is B", models do not answer "B is A". [A]
- Lampinen 2025 (2505.00661): with the same data, in-context learning generalises to reversals, syllogisms and "category holdouts" better than fine-tuning; adding in-context inferences to the fine-tuning data closes the gap. [V]
- Padmanabhan 2023 (2306.09306): propagate an update by distilling the model-with-definition into the model-without. [A]
- Chen 2025, Self-Questioning Language Models (2508.03682): a proposer generates problems, a solver answers, majority vote stands in for truth. [A]
- Ganguli 2026, TFGN (2605.15053): claims replay-free continual pre-training without forgetting at 8B parameters; single-author preprint, not checked. Atmuri 2026, CMP (2607.26523): local learning with sparse codes forgets less than an EWC Transformer and has "a substantial single-domain accuracy gap". Eugenio 2025 (2503.02057): a Hebbian hierarchy that learns to tokenise. [A]

Lifelong language learners outside weight plasticity
- Mitchell 2015/2018, NELL: runs continuously since 2010, a knowledge base of "confidence-weighted beliefs", learns to "infer new beliefs from old ones" and to extend its ontology. [A] It works in iterations, promotes candidate beliefs by thresholds and agreement of components, starts from a given ontology and suffers drift from early errors. [M] On P2 it is nearer than any neural system in this note; it is not a tensor engine. [I]
- Mazumder 2019, CILK (1907.13295): a dialogue engine that, when it cannot answer, asks the user for a clue or a fact and keeps it in a knowledge base; question types are designed, answers are assumed true. [V]
- Liu 2022, SOLA (2203.08994): a framework in which detected novelty "serves as an intrinsic self-motivation" and triggers characterisation, a question to a human and incremental learning; one implemented agent. [V]
- Misra 2017, Learning by Asking Questions (1712.01238): a visual question-answering learner that "must ask questions it wants answers to" from an oracle and finds an easy-to-hard curriculum on CLEVR. [A]

Positions
- Dupoux, LeCun and Malik 2026 (2603.15381): "AI models, once deployed, learn essentially nothing … learning is outsourced to human experts"; proposes a meta-controller, System M, and "we propose that System M's core routing policy is hardwired"; "we can only offer a blueprint". [V]
- Dorovatas 2026 (2603.01761; 24 authors, ICML position track): in-weight learning has made forgetting "a persistent challenge"; proposes in-context memory for fast accumulation and in-weight learning for slow updates. [A]
- Surveys used for orientation: Shi 2024 (2404.16789), Liang 2023 (2303.15361), Wang 2023 (2302.00487), Bidaki 2025 (2501.04897), Safa 2024 (2407.17305), Zheng 2025 (2501.07278), Parisi 2020 (2003.09114). [A]

## 4. What broke

### 4.1 Stability

- Hebbian memories run away without help: in a fast associative memory "the scalar product of
  two hidden vectors could vanish or explode", cured by layer normalisation; a plain Hebbian trace
  needs decay "to prevent runaway positive feedback"; the follow-up clips traces to [−1, 1] and
  clips gradients, "which greatly improved stability". [V]
- Test-time gradient steps on single tokens "can easily lead to gradient explosion by chance";
  TTT-E2E freezes most layers at test time for stability. [V]
- Unsupervised test-time adaptation collapses to one class under mixed shifts, small batches or
  imbalanced streams; long runs need resets, and deciding when and where to reset is a 2026
  research topic. [V]/[A]
- Online adaptation of LLMs works best with resets "to avoid that the models overfit to data in
  the local context". [V]
- Deep RL from a pure stream "often fail[s] to learn", performing "close to random policies"
  before Stream-X (the "stream barrier"). [A]
- Sequential weight editing has a positive norm feedback loop and collapses abruptly. [A]/[V]

### 4.2 Forgetting

- Dynamic evaluation across documents forgets the past. [V]
- SEAL: "performance on earlier tasks gradually declines as the number of edits increases". [V]
- Twenty one-at-a-time fact writes: 1% of questions on earlier facts answered when each write
  repeated the statement, 46% with varied restatements, "below 50%" even then. [V]
- Nested Learning: forgetting "is not 'solved' in general". [V]
- The working answers move knowledge out of the plastic weights: frozen lower layers (REMIND,
  OML), a second static MLP (TTT-E2E), a codebook or side memory (GRACE, WISE), a memory bank
  beside a frozen model (MAC), or a reset. [V]/[A]
- This is the same wall the owner's repository hit (52 naive writes into one layer, VISION.md
  point 14), now measured at LLM scale by others. [I]

### 4.3 Loss of plasticity, and its Bayesian twin

- Endless backpropagation ends near the level of a linear network; Adam and dropout make it
  worse, L2 regularisation and shrink-and-perturb help, and "continual injection of randomness is
  important to reduce the loss of plasticity". [V]
- Exact accumulation of evidence in Gaussian weights drives variances to zero: FOO-VB's
  "plasticity collapses, capping accuracy at around 70%", and uncertainty estimates lose their
  meaning ("vanishing uncertainty"). [V]
- MESU's cure is to forget on purpose, on a window. A system whose evidence only grows will meet
  the same end; a system that forgets by time deletes without having been contradicted. [V]/[I]

### 4.4 Scale

- Learned learning rules are shown on MNIST, Omniglot, small control tasks; one transfer, MNIST
  to Fashion-MNIST. [V]
- Learning the rule needs an outer loop over whole lifetimes: "a whole life cycle is just one
  data point … extending such techniques to much larger architectures yields severe scalability
  issues". [V]
- TTT-E2E pays for its meta-learning with "gradients of gradients". [V]
- Belief-valued weights are shown on MNIST, Fashion-MNIST and CIFAR; I found no language model
  trained this way. [V]/[I]
- Under a real-time budget, or without a storage limit, the simplest baselines win. [A]

### 4.5 Language

- A fact written into the weights is not thereby usable: no propagation to consequences, no
  reversal, state-but-not-use. The same fact placed in context is used far better. [V]/[A]
- The repairs all have the same shape: generate the consequences in context, then train on them
  (DCT, SEAL's implications, distillation from the model-with-context, augmentation with
  in-context inferences). The weight update does not reason; the forward pass does. [V]/[I]
- Compression into weights loses detail: full attention "dramatically outperforms" TTT-E2E on
  needle retrieval; weight memory is expected to be "better suited to store the style or topic". [V]
- Hebbian or local-rule language models exist as small or qualitative demonstrations. [A]

### 4.6 The rule that learns is itself not learned by living

- Rows 2–7, 9–15: the rule, the gates or the starting point are fitted in an outer loop and then
  fixed. [V]
- Within one life, what the stream itself adapts is a handful of scalars of the rule: step sizes
  and decay rates, shown on simple or last-layer problems [A]; in reinforcement learning also
  discount-type hyper-parameters by online meta-gradients [M]. The form of the rule is not
  learned in-life in anything I found. [I]
- The 2026 blueprint fixes the controller by design, to be found by an evolutionary outer loop. [V]

### 4.7 Learning from one's own output

- Entropy minimisation and pseudo-labels accumulate error and collapse. [A]/[V]
- Training on generated data makes "tails of the original content distribution disappear"
  (Shumailov 2023, 2305.17493); generative replay (Shin 2017, 1705.08690) replaces stored data
  by generated data. [A]
- The two positive cases score the generations first: by held-out questions (SEAL) or by joint
  consistency (DCT). SEAL's authors name the next step: let the model "generate … its own
  evaluation questions". [V]

### 4.8 Evaluation

- Online accuracy can be earned by predicting the stream's label order without looking at the
  input. [A]
- Lifetime hyper-parameter tuning makes "all algorithms equally succeed". [A]
- The train/test split survives in the benchmarks even where the method does not need it. [I]

## 5. The "red" example

### 5.1 The stream and the code

The experiences, as this field's rules can take them: E1 (red, dark); E2 (red, light), the
contradiction; E3, the rule in the only form a weight update accepts, six observations
(blue, dark), (blue, light), (green, dark), (green, light), (yellow, dark), (yellow, light);
E4, 100 unrelated experiences. Input code: one unit per colour word plus one shared unit "is a
colour"; "purple" has a unit and is never observed. The shared unit is given by me; no rule in
this note creates it. All numbers below are [C], from
`tools-online-plasticity/red_example_online_plasticity.py`; I ran the published update equations
on one unit with six relevant weights, not the authors' code.

### 5.2 The closest system: online Bayesian weights (MESU family)

Why this one: it is the family in which each weight carries content and certainty as separate
quantities that can both be read out, is updated by every experience with no phases, and keeps
a contradicted belief instead of replacing it (rows 23–25). Importance-based methods (row 18,
EWC, metaplastic binary weights) keep a resistance to change, which cannot be read as "how well
is this known". MESU is the strongest published form (200 tasks, no boundaries,
out-of-distribution detection); the exact computation it approximates is a Kalman filter. [I]

"Conviction" below = 1 − posterior predictive variance / prior predictive variance. One number
was fitted: the observation noise, so that conviction is 0.230 after E1. [C]

| After | Exact filter: mean(red) | conviction(red) | conviction(colour weight) | conviction(purple, never met) | Owner's value |
|---|---|---|---|---|---|
| E1 "red is dark" | 0.230 toward dark | 0.230 | 0.115 | 0.057 | 0.23 |
| E2 "red is sometimes light" | 0.000 | 0.374 | 0.187 | 0.093 | 0.57 |
| E3 three other colours, dark and light | 0.000 | 0.461 | 0.479 | 0.240 | rule 0.77, red 0.88 |
| E4 100 unrelated | 0.000 | 0.461 | 0.479 | 0.240 | |

What it does, step by step (numbers [C], reading [I]).
1. First hearing: stored weakly. The gain is set by prior and noise variances, not by the event.
   [C]
2. Contradiction: the mean returns to the middle ("as much dark as light") and conviction rises
   from 0.230 to 0.374. Nothing is deleted. This reproduces "contradicting evidence refines" in
   one sense only: the same kind of belief is held more firmly. The belief does not become "dark
   but sometimes light"; there is one number per weight. [C]/[I]
3. Rule: it cannot be told the rule. Given the instances, the shared weight is pinned at the
   middle with conviction 0.479. [C]/[I]
4. Feedback to the instance: red rises from 0.374 to 0.461 although red is not mentioned again,
   and purple, never met, is predicted "as much dark as light" with conviction 0.240. In the
   exact filter this comes from the posterior correlation between the red weight and the colour
   weight (−0.186 at the end). [C]
5. Question: none. The largest remaining uncertainty is on purple (predictive variance 1.521
   against 1.079 for the met colours). The field's analogue of asking is to act on that
   uncertainty: LO-FI is used with Thompson sampling in a bandit, where an action is chosen by
   sampling weights from the posterior [V]. Such a learner can only point at a row it already
   has. [C]/[I]

The same stream through the approximations that scale. [C]
- Mean-field (diagonal) Bayes: red 0.130, 0.230, 0.387; colour weight 0.544; purple 0.272. It is
  under-confident about the instance it met and over-confident about the category and the unseen
  member, because it drops the correlation. [C]
- MESU's printed rule, no forgetting: red 0.144, 0.250, 0.407; colour 0.564; purple 0.282; the
  first mean overshoots (0.299 against 0.230) because the rule assumes small updates. [C]
- MESU with window N = 20: close to that up to E3 (red 0.354), then 0.179 after 20 unrelated steps
  and 0.004 after 100. Mean and variance are back at the prior: the belief is gone without ever
  having been contradicted. (The paper uses N = 300,000 mini-batches [V]; 20 is my choice to make
  the effect visible in a 100-step example.) [C]

Identity worth knowing: after n equal observations a scalar Kalman filter has conviction
n/(n + s²/v₀), the same form as W/(W + k) in the lead design. With the fitted ratio 3.348 it
gives 0.230, 0.374, 0.544, 0.705 for n = 1, 2, 4, 8. The first two steps of the owner's example
are therefore standard Bayesian filtering up to the choice of constants. [C]/[I]

What it lacks for the example.
- The form of a belief cannot change, and "sometimes" is not content: variation in the outcome
  is absorbed by a fixed noise constant. [I]
- The category unit must exist in the input code. [I]
- A rule can only be induced from instances, never told; an exception is not distinguished from
  noise. [I]
- No question, no act. [I]
- Kept exactly, it ends unable to learn (§4.3) [V]; with a window, it forgets by the clock. [C]
- Shown on image benchmarks only. [V]

### 5.3 The strongest working language system: LLM test-time training

TTT-E2E and In-Place TTT are the strongest systems that change weights during inference on
language (3B and 4B parameters, 128K context). What they would do with the example, from their
published mechanics. [V]/[I]
1. The three sentences are a few dozen tokens. TTT-E2E "only takes a gradient step once the …
   tokens have completely filled a TTT mini-batch" of 1K tokens; In-Place TTT updates per chunk
   of 512–1024 tokens [V]. Nothing the model does with these sentences uses a changed weight: they
   are held in the attention window and whatever is concluded from them is concluded in context.
   [I]
2. If the sentences sit far apart in a long document, each falls into a different mini-batch and
   contributes to one averaged gradient step with about a thousand other tokens (TTT-E2E), or to
   one sum of outer products with no error term (In-Place TTT). There is no strength per
   statement and no record that the second statement bears on the first. [I]
3. The rule sentence is more tokens. Whether "red" inherits from "all colours" then depends on
   in-context inference; when facts are written into weights instead, consequences are not drawn
   (§4.5). [I]
4. At the document boundary In-Place TTT resets; TTT-E2E starts the next context from W₀ [V].
   Nothing of the example survives. Without the reset this becomes dynamic evaluation on a
   stream, which forgets (§4.2). [I]
5. No question arises from the mechanism; a pre-trained model may produce one as text. [I]

What a gradient step does to the belief itself, on the same unit (online SGD on a logistic unit,
step size fitted so that p(dark|red) = 0.615 after E1). [C]

| After | p(dark given red) | p(dark given purple, never met) |
|---|---|---|
| birth | 0.500 | 0.500 |
| E1 | 0.615 | 0.558 |
| E2 | 0.473 | 0.487 |
| E3 | 0.447 | 0.461 |
| E4, no decay | 0.447 | 0.461 |
| E4, with decay to the trained weights (λ = 0.02, a separate run) | 0.493 | 0.495 |

### 5.4 Ignorance and a known mixture; the other rules

- Online SGD after "dark" then "light": 0.473 for red, 0.487 for a colour never met, 0.500 at
  birth. The three states differ by less than 0.03 and nothing in the weights says which is
  which. Online Bayes: mean 0.000 in all three,
  conviction 0.374, 0.093 and 0.000. A point-estimate weight cannot hold "I know it is both". [C]
- Hebbian fast weights (row 1 rule, paper defaults): red reads 0.500 after E1, −0.025 after E2
  (the two cancel and the newer wins by the decay margin), 0.000 after E4. With the shared unit,
  purple reads 0.250 after E1 alone: the instance leaks to the whole category at once, by
  overlap, with no evidence about the category. [C]
- Delta rule (row 3, write strength 0.5 assumed): 0.500 then −0.250, a recency-weighted average
  that never accumulates: after any number of "dark" one "light" moves the value half-way. With
  the shared unit, red ends at −0.407 only because the last thing said about yellow was "light".
  [C]
- Titans rule (row 10, gates held constant by me): surprise 1.0 at the first hearing and 1.5 at
  the contradiction, so the contradiction is the largest write; −0.004 after E4. In the real
  model the gates are learned functions and could behave otherwise; the rule has no place for a
  count of evidence. [C]/[I]

### 5.5 The five steps of the example against the field

| Step of the example | Best this field does | Where |
|---|---|---|
| First hearing stored weakly | native in Bayesian weights and in a single gradient step | rows 8, 23–25 |
| Contradiction refines, conviction rises | certainty rises while content moves (Bayesian weights); the form of the belief never changes | rows 23–25 |
| Category rule induced | a shared weight learned from instances, if the shared unit is given; a told rule only as text in a language model | rows 23–25; §4.5 |
| Rule feeds support back to the instance | through shared weights or posterior correlation (small models); in language models only by generating the consequences and training on them | row 25; rows 15, 16 |
| A question arises by itself | not found; nearest are novelty-triggered designed questions (SOLA, CILK) and acting where the posterior is wide | §3 |

All cells [I], resting on the rows cited.

## 6. Unsolved

### 6.1 In the field's own terms

1. Test-time learning that persists. "The acquired in-context knowledge does not persist once
   the current context is removed"; "automatic detection of reset points" is listed as open. [V]
2. Forgetting when knowledge is written into weights: "not 'solved' in general". The 2026
   position paper answers by moving fast learning out of the weights. [V]/[A]
3. Plasticity over an unbounded life, for gradient learners and for Bayesian ones. [V]
4. Stability of unsupervised adaptation without resets. [V]/[A]
5. Learning the learning rule inside one life and at scale. Bilevel search over lifetimes does
   not scale; in-life adaptation covers scalar hyper-parameters only. [V]/[A]
6. Propagation: making a fact learned in weights available to inference, reversal and
   composition without spelling out its consequences as training data. [V]/[A]
7. Choosing what in a stream deserves to be learned: token re-weighting, surprise gates, and
   "TTT on self-generated tokens, which can be a filtered or rephrased version" are open
   proposals. [A]/[V]
8. Belief-valued weights at scale: diagonal posteriors distort (§5.2), full ones are intractable,
   low-rank ones are shown on Fashion-MNIST. [C]/[V]
9. Recall against compression: what belongs in weights and what in context. [V]
10. A reality check for self-generated training data that does not need labelled questions. [V]
11. Evaluation of a learner that has no test phase. [A]
12. A language system with no train/eval split. Stated as principle in the Alberta Plan, Nested
    Learning, TTT-E2E and the 2026 blueprint; every language system found here has a pre-training
    phase and either a per-context reset or measured forgetting. [V]/[I]

### 6.2 Not posed by the field, as far as 33 searches show

All [I]; absence of evidence in my searches, not proof of absence.
- A conviction attached to a statement, not to a weight, that rises when a contradiction
  refines the statement.
- A contradiction that changes the form of a belief ("dark" to "dark but sometimes light")
  instead of moving a number.
- A rule told in language that gives weight-level support to an instance, with exceptions kept.
- A question generated from a category whose members are known to be incomplete.
- Creation of the category unit itself by a plasticity rule (the neighbouring field of growing
  networks is covered in `art-growth.md`).
- A reality score on self-generated constructions that is earned by later experience and feeds a
  learned trust in one's own constructing.
- An operating procedure (when to revise, generalise, feed back, ask) that is learned by the
  same rule in the same life and measured on a domain not seen before.
- Provenance of what is in the weights (told, observed, inferred, imagined).

## 7. Consequences for the design under test

All [I].
- The arithmetic of weak first storage and of conviction rising under contradiction is not new;
  it is scalar Bayesian filtering (§5.2). A claim of novelty has to rest on what §6.2 lists.
- Baselines this field supplies for the acceptance tests, on the identical stream: an online
  Bayesian learner (exact Kalman on small problems, mean-field otherwise), dynamic evaluation
  with and without reset, a delta-rule fast-weight memory, and the "blind" predictor that sees
  only the order of past labels.
- A test that separates the families cheaply: after "A" then "not A", ask for the belief and for
  how well it is known, and compare with a thing never met (§5.4).
- Test use, not statement: application and composition questions, since "can state but cannot
  use" is the measured failure of writing into weights.
- Evidence that only accumulates meets "catastrophic remembering"; forgetting by a time window
  deletes uncontradicted beliefs. The design needs an explicit answer for a world that changes,
  and a test for it.
- Any loop that learns from its own constructions needs the score before the update; the two
  cases that worked in this field both had it, and the ones without it collapsed.
- An outer loop that tunes the operating procedure across simulated lifetimes would repeat the
  field's P5-partial pattern; the distinguishing claim is in-life acquisition and transfer.
- One measured result supports the owner's "a concept is its lived encounters": a fact written
  through 24 varied restatements was still usable after nineteen later writes in 46% of questions,
  against 1% for the same fact repeated verbatim. Varied encounters are what made the knowledge
  last and be usable. ([V] for the numbers.)
- Two 2026 papers and the 2026 position paper move toward an offline consolidation phase and a
  split between fast in-context memory and slow weights. That direction keeps the train/run
  boundary and renames it; a design that claims P1 should say how its idle processing differs.

## 8. Method

- Searches (33; the two marked E were in extended mode): In-Place TTT; differentiable plasticity
  survey; Nested Learning/Hope/Titans; task-free online continual learning; loss of plasticity;
  test-time adaptation failure modes; dynamic evaluation on streams; meta-learned plasticity
  rules; Bayesian synapses and metaplasticity; lifelong model editing collapse; SEAL; TTT-E2E;
  no train/test split and the Alberta Plan; lifelong language learning; in-context learning as an
  implicit weight update; fast weight programmers and self-referential matrices; Hebbian
  Transformers; online Bayesian filtering for networks; never-ending learning; continual learning
  of LLMs surveys; CaMeLS and MAC; complementary learning systems and sleep; evaluation of online
  continual learning; belief revision in weights; Titans mechanism; continual learning in LLM
  agents 2026 (E); "sleep" consolidation for LLMs; one-shot word learning; Hebbian language
  modelling; synaptic plasticity as Bayesian inference; self-initiated open-world learning; LLMs
  that detect their own knowledge gaps and ask (E); IDBD and step-size adaptation.
- Reading: PDFs fetched from arxiv.org and abstract pages from export.arxiv.org with
  `tools-online-plasticity/fetch_arxiv.sh`; text extracted with pdftotext. [V] statements were
  checked against the extracted text.
- Not run: any of the authors' code. The traces in §5 are the published update equations on one
  unit with six relevant weights, with every assumed constant named in the script header.
- Not covered here, by design: growing networks and allocation on mismatch (`art-growth.md`),
  evidence logics (`nars-logic.md`), predictive coding and energy-based learning (the sibling
  note on that field).

## Files

- `/tmp/claude-0/-workspace-cevahir-ai/79909244-c51d-5307-ad00-1a1e5ef77203/scratchpad/newcore/landscape/online-plasticity.md` (this note)
- `/tmp/claude-0/-workspace-cevahir-ai/79909244-c51d-5307-ad00-1a1e5ef77203/scratchpad/newcore/landscape/tools-online-plasticity/red_example_online_plasticity.py` and `red_example_output.txt`
- `/tmp/claude-0/-workspace-cevahir-ai/79909244-c51d-5307-ad00-1a1e5ef77203/scratchpad/newcore/landscape/tools-online-plasticity/source_index.txt` (118 downloaded papers)
- `/tmp/claude-0/-workspace-cevahir-ai/79909244-c51d-5307-ad00-1a1e5ef77203/scratchpad/newcore/landscape/sources-online-plasticity/` (one directory per arXiv id)
