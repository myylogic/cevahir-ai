# Landscape: ART and other "allocate a unit on mismatch" systems, read against P1–P6

Field covered: adaptive resonance theory (ART, fuzzy ART, ARTMAP), growing neural gas and
self-organising incremental networks, dynamically expandable / constructive networks
(cascade-correlation, progressive nets, DEN, neurogenesis-inspired continual learning),
nonparametric Bayesian models (CRP / IBP / HDP and their cognitive-science uses), and
prototype / exemplar memories with novelty-triggered allocation.

Date of this note: 2026-11-14. 30 web searches (2 in extended mode), 34 arXiv papers and 3 ACL
Anthology papers downloaded; the ones marked [V] below were read in the parts cited.

## Summary

Tags are explained in §0; each line below is supported in the section named.

- The field is strong on P1 and P3 and weak everywhere else: of 24 systems in the main table, none
  is native on P4, P5 or P6, and none is native or partial on all six. [I] (§2)
- The systems that learn while running (ART, GWR, HTM, SUSTAIN, Cobweb) hold graded beliefs only
  as counts or synaptic strengths and have no rule level; the one family where an induced category
  rule feeds an instance (hierarchical Bayes) is, in the forms found here, neither online nor a
  network. [I] (§2, §4)
- A first exposure that is stored weakly and must be confirmed exists in only one system reviewed
  here [I]: HTM sequence memory, whose new synapses start at 0.21 against a connection threshold of
  0.5. [V] (§2.1 row 13)
- Closest to the "red" example: the hierarchical Bayesian overhypothesis model with a CRP. It
  reproduces refinement, rule induction and top-down support (red's "sometimes dark, sometimes
  light" belief 0.881 → 0.926 from other colours alone), but its first hearing is strong (0.68),
  one contradiction is nearly decisive, it recomputes a full posterior at every step, and no
  question arises. [C] (§6.1)
- Closest online neural system: ARTMAP-IC. It stores the first hearing at full strength, turns the
  contradiction into a count ratio (1.0 → 0.5 → 0.667), and either keeps instances with no rule or
  merges them into a rule and loses the instance. Standard fuzzy ARTMAP drops the contradicting
  pair altogether. [C] (§6.2)
- What broke: order dependence, category proliferation (about 56 thousand categories for 60,000
  MNIST images), the hand-set allocation threshold, rigidity from freezing or monotone weights,
  forgetting once features are shared (worked around with frozen pre-trained backbones), no
  large-scale evaluation, no fluent language. [V] (§5)
- Unsolved in the field's own terms: order dependence, online vigilance adaptation, structured and
  linguistic representations, distributed codes, uniting match-based with error-based learning,
  bounded growth, online nonparametric inference. [V] (§7.1)
- Not posed by the field at all, as far as 30 searches show: rule-to-instance feedback inside an
  online growing network, a belief strength attached to a statement, refinement that changes the
  form of a belief, allocation pressure turned into a question, a reality measure on
  self-generated constructions, operators as learned objects. [I] (§7.2)

## 0. How to read this note

### Status tags (every statement carries one)

| Tag | Meaning |
|---|---|
| [V] | VERIFIED: I read the source text itself (PDF from arxiv.org or aclanthology.org) for this statement. |
| [V-s] | VERIFIED in a secondary source that I read (mostly the ART survey arXiv:1905.11437, the continual-learning review arXiv:1802.07569, the BNP tutorial arXiv:1106.2697). The original paper was not read. |
| [A] | ABSTRACT: abstract or search-result snippet only. |
| [M] | MEMORY: from my own knowledge, not checked in this session. |
| [C] | COMPUTED here: output of a script in `tools-art-growth/` or hand arithmetic shown in the text. Not a literature claim. |
| [I] | My own inference or judgement from the tagged facts. Not a literature claim; treat as opinion. |

The P1–P6 codes, in the tables and in §2.1, are all [I] (my reading), based on the mechanism
statements next to them. Where a reason given in §2.1 quotes or restates a source, the source's tag
follows it; a reason with no tag is [I].

### Access limits in this session

- WebFetch could not resolve any host (arxiv.org, ar5iv.labs.arxiv.org and open.bu.edu all failed with
  `ENOTFOUND`); those hosts would have to be allowed in the session's network settings for WebFetch
  to work. Full texts were therefore read by downloading PDFs with curl from arxiv.org,
  export.arxiv.org and aclanthology.org only, and converting them with pdftotext.
- Everything not on arXiv or the ACL Anthology (the original ART papers, GNG, GWR, SOINN, RAN,
  cascade-correlation, SUSTAIN, Kemp et al. 2007, the ARTMAP information-fusion paper, Grossberg's
  2020 and 2025 papers, MemoryART) rests on search snippets [A], on a secondary source I did read
  [V-s], or on memory [M]. Two-column PDFs (e.g. arXiv:1612.03770) were read through interleaved text.

### Principles (the owner's, as given)

P1 learning and running are one process, no train/freeze/eval split. P2 weights are graded beliefs:
weak first storage, contradiction refines instead of deleting, a category rule is induced and feeds
support back to the instance. P3 concepts are formed when needed, categories emerge from instances,
no fixed embedding table. P4 imagination: the system observes its own constructions, measures how
real they are, advances by that measure. P5 what is acquired is the way of constructing (revise,
generalise, feed back, ask), improving with experience and transferring across domains. P6 an engine
of tensors and equations, language in and out eventually.

Codes: **N** = natively, **P** = partially, **–** = not at all. **N\*** (used for P3) = categories
emerge natively but over a feature basis that is given, not formed. **N†** (used once, for P2) =
natively, but as posterior probabilities over a hypothesis space fixed by the modeller, not as
network weights.

## 1. What unites the field

Every system here keeps a set of units (categories, prototypes, experts, clusters, columns) and
compares each input with them through some mismatch score; if the score crosses a threshold, a new
unit is allocated, otherwise an existing unit is adjusted. [I]

- ART: no category passes the vigilance test, so "a new one is created to encode the presented input
  sample". [V] (arXiv:1905.11437 §2.1)
- GWR: a neuron is created when network activity is below the insertion threshold a_T and the best
  unit is already habituated. [V] (arXiv:1805.10966 §3.1)
- DP-means: "a new cluster is formed whenever a point is farther than λ away from every existing
  cluster". [V] (arXiv:1111.0352)
- CN-DPM: its inference "adds a new component only when ρ_K+1 is greater than a certain threshold". [V]
  (arXiv:2001.00689 §2.2)
- CURL: "we rely on a threshold to determine when to instantiate a new component". [V] (arXiv:1910.14481 §2.3)
- DEN: expand when the loss stays above τ, split a unit when its drift exceeds σ. [V] (arXiv:1708.01547 Alg. 1)
- HTM sequence memory: a dendritic segment is created "only when there is no match to any existing
  segment". [V] (arXiv:1512.05463, appendix)

The ART vigilance, the GWR insertion threshold, the DP concentration α (DP-means derives λ from α in
a small-variance limit [V]) and the expansion thresholds of deep expandable nets play the same role;
the field's central unsolved parameter is this one threshold (§5.3). [I]

## 2. Main table: 25 works

P-codes are in the order P1 P2 P3 P4 P5 P6.

| # | Title | First author | Year | Venue / arXiv | Status | P1 P2 P3 P4 P5 P6 |
|---|---|---|---|---|---|---|
| 1 | Fuzzy ART: fast stable learning and categorization of analog patterns by an adaptive resonance system | Carpenter | 1991 | Neural Networks 4:759–771 [M] | [V-s] mechanism; [M] bibliography | N – N\* P – P |
| 2 | Fuzzy ARTMAP: a neural network architecture for incremental supervised learning of analog multidimensional maps | Carpenter | 1992 | IEEE Trans. Neural Networks 3(5) [M] | [V-s] mechanism; [A] title | N P N\* P – P |
| 3 | ARTMAP-IC and medical diagnosis: instance counting and inconsistent cases | Carpenter | 1998 | Neural Networks 11(2):323–336 | [A] + [V-s] | N P N\* P – P |
| 4 | Distributed ART / distributed ARTMAP | Carpenter | 1997–98 | Neural Networks [M] | [V-s] | N P N\* P – P |
| 5 | Self-organizing information fusion and hierarchical knowledge discovery: a new framework using ARTMAP neural networks | Carpenter | 2005 | Neural Networks 18(3) [M]; BU tech. report 2004 [A] | [A] | P P N\* P – P |
| 6 | The Bayesian ARTMAP (Bayesian ART) | Vigdor | 2007 | IEEE Trans. Neural Networks [M] | [V-s] | N P N\* P – P |
| 7 | An ART-based topological clustering algorithm with a self-adjusting vigilance parameter (IDAT) | Masuyama | 2025 | arXiv:2511.17983 | [A] | N – N\* P – P |
| 8 | A survey of ART neural network models for engineering applications | Brito da Silva | 2019 | Neural Networks; arXiv:1905.11437 | [V] | n/a (survey) |
| 9 | A growing neural gas network learns topologies | Fritzke | 1995 | NIPS 7 | [A] + [M] | N – N\* – – P |
| 10 | A self-organising network that grows when required (GWR) | Marsland | 2002 | Neural Networks 15:1041–1058 | [A] + [V-s] | N – N\* – – P |
| 11 | Lifelong learning of spatiotemporal representations with dual-memory recurrent self-organization (GDM) | Parisi | 2018 | Front. Neurorobotics; arXiv:1805.10966 | [V] | P P N\* P – P |
| 12 | An incremental network for on-line unsupervised classification and topology learning (SOINN) | Furao | 2006 | Neural Networks 19(1) [M] | [A] + [M] | P – N\* – – P |
| 13 | Continuous online sequence learning with an unsupervised neural network model (HTM sequence memory) | Cui | 2016 | Neural Computation 28(11); arXiv:1512.05463 | [V] | N P P P – P |
| 14 | The cascade-correlation learning architecture | Fahlman | 1990 | NIPS 2; CMU-CS-90-100 | [A] + [M] | – – P – – P |
| 15 | Progressive neural networks | Rusu | 2016 | arXiv:1606.04671 | [V] | – – – – – P |
| 16 | Lifelong learning with dynamically expandable networks (DEN) | Yoon | 2018 | ICLR; arXiv:1708.01547 | [V] | – – P – – P |
| 17 | Neurogenesis deep learning (NDL) | Draelos | 2017 | IJCNN; arXiv:1612.03770 | [V] | – – P P – P |
| 18 | A neural Dirichlet process mixture model for task-free continual learning (CN-DPM) | Lee | 2020 | ICLR; arXiv:2001.00689 | [V] | P P P P – P |
| 19 | Continual unsupervised representation learning (CURL) | Rao | 2019 | NeurIPS; arXiv:1910.14481 | [V] | P – N\* P – P |
| 20 | The adaptive nature of human categorization (rational model = CRP mixture); tutorial on Bayesian nonparametric models | Anderson; Gershman | 1991; 2012 | Psych. Review 98; arXiv:1106.2697 | [V-s]; [V] | P P N\* – – P |
| 21 | A hierarchical Bayesian language model based on Pitman-Yor processes (HPYLM) | Teh | 2006 | ACL, P06-1124 | [V] | – P P – – P |
| 22 | Learning overhypotheses with hierarchical Bayesian models | Kemp | 2007 | Developmental Science 10(3):307–321 | [A] + [M] | – N† P – P P |
| 23 | Contextual dependencies in unsupervised word segmentation (DP / HDP lexicon) | Goldwater | 2006 | ACL, P06-1085 | [V] | – P N – – P |
| 24 | SUSTAIN: a network model of category learning | Love | 2004 | Psych. Review 111(2):309–332 | [A] + [M] | N P N\* – – P |
| 25 | Efficient induction of language models via probabilistic concept formation (Cobweb); Cobweb/4L | MacLellan; Lian | 2022; 2024 | arXiv:2212.11937; arXiv:2409.12440 | [V] | N P N\* P – P |

No row is N on P4, P5 or P6. No row is N or P on all six principles. The rows that are N on P1
(1–4, 6, 7, 9, 10, 13, 24, 25) are at most P on P2; the one row that is N† on P2 (22) is – on
P1 and P4. [I]

### 2.1 Mechanism (two sentences) and P-reading for each row

**1. Fuzzy ART.** Each category is a weight vector w (a hyperbox under complement coding); an input
selects the category maximising |x∧w|/(α+|w|), and the orienting subsystem accepts it only if the
match |x∧w|/|x| reaches the vigilance ρ, otherwise that category is reset and the search continues.
On resonance the winner learns w ← (1−β)w + β(x∧w); if no category passes, a new one is created
from the input. [V-s]
P1 N: one-pass and online; the same match cycle that produces the output gates the weight change. [V-s]
P2 –: the weights are feature bounds whose norm is "monotonically non-increasing during learning since
categories can only expand"; fast learning means the weights "converge to their asymptotic values
directly with each input sample presentation", i.e. a first exposure is stored at full strength. [V-s]
P3 N\*: categories are created from instances on mismatch; the input feature vector is given. [V-s]
P4 P: "A category represents a hypothesis" and a "hypothesis test cycle" checks the top-down template
against the input; nothing is constructed or judged in the absence of input. [V-s] / [I]
P5 –: search, reset and ρ are fixed; ρ is "often set empirically in an ad hoc manner". [V-s]
P6 P: closed-form min / L1 / winner-take-all equations with GPU implementations; no language. [V-s]

**2. Fuzzy ARTMAP.** Two fuzzy ART modules (inputs, labels) are joined by a map field; when the
category chosen for the input predicts the wrong label, match tracking raises the input vigilance
just above the current match (ρ_a = M + ε), which forces a search for, or the creation of, a more
specific category. Map weights are binary under fast learning; a slow-learning rule makes them graded
and gives an estimate of p(label | category) as a normalised map weight. [V-s]
P1 N. P2 P: graded label estimates exist under slow learning, and match tracking keeps the general
category while adding an exception; but an identical input with a conflicting label cannot be learned
under the original rule (see §6.2). [V-s] / [C] P3 N\*. P4 P: the prediction is tested against the
outcome and a mismatch drives the search; the map field "is a critic". [V-s] P5 –. P6 P.

**3. ARTMAP-IC.** Fuzzy ARTMAP plus an instance-counting field (each category counts how often it
was activated) and the MT− rule, which sets vigilance slightly *below* the failed match (ε ≤ 0);
prediction is distributed over the Q most active categories, weighted by count × activation, which
yields class probabilities. [V-s] With MT−, identical inputs with different outcomes can be coded
by separate categories [C] (§6.2), which is what the title calls "inconsistent cases"; the abstract,
as summarised in search results, presents MT− as suited to sparse or inconsistent data and reports
tests on four medical datasets. [A]
P1 N, with a caveat: training is winner-take-all while prediction is distributed, an asymmetry that
Default ARTMAP 2 had to patch with a "distributed-next-input-test". [V-s] P2 P: beliefs are relative
frequencies held as counts and contradictions coexist; there is no rule level. [V-s] / [C] P3 N\*.
P4 P. P5 –. P6 P.

**4. Distributed ART / ARTMAP.** Winner-take-all is replaced by graded activation over several
categories (an "increased-gradient" content-addressable-memory rule), and adaptive thresholds are
learned with distributed instar / outstar laws scaled by each node's activation. It reduces to fuzzy
ART in winner-take-all mode and was built for noise robustness and memory compression. [V-s]
P1 N. P2 P: graded participation in learning, but the thresholds "monotonically increase". [V-s]
P3 N\*. P4 P. P5 –. P6 P. The survey still lists distributed representation as an open problem. [V-s]

**5. ARTMAP information fusion.** An ARTMAP with distributed codes learns from sources whose labels
for the same input are reliable but inconsistent, so that one input comes to predict several output
classes (one-to-many learning). The system then infers multi-level relationships among the output
classes without any supervised labelling of those relationships, producing a self-organised
hierarchy of classes. [A] (paraphrase of the abstract as returned by search, not a quotation) That
the derived relations are rules of the form x ⇒ y carrying confidence values, computed from
predictions over a set of inputs, is from memory. [M]
P1 P: online learning plus a separate derivation pass. [M] P2 P: graded one-to-many beliefs and a
derived rule level, but nothing flows back from the rule to the instance. [A] / [I] P3 N\*: categories
and a class hierarchy emerge. [A] P4 P. P5 –. P6 P.

**6. Bayesian ART(MAP).** Categories are Gaussians with a mean, a covariance and a count-based
prior; the winner is the maximum-posterior category and vigilance bounds its hyper-volume
(det Σ ≤ ρ). Parameters follow sequential maximum-likelihood updates, so "categories can both grow
and shrink". [V-s]
P1 N. P2 P: posterior p(category | x) and counts are graded; no rule level. P3 N\*. P4 P. P5 –. P6 P.

**7. IDAT (ART-based topological clustering, self-adjusting vigilance).** ART node creation on
mismatch is combined with topology learning, and the vigilance threshold and its recalculation
interval are adjusted "through a diversity-driven adaptation mechanism" instead of being set by
hand. It is evaluated on 24 real-world datasets in stationary and class-incremental clustering. [A]
Its ancestors: TopoART (fuzzy ART + SOINN-inspired topology, two modules where the first filters
noise for the second, candidate categories deleted unless they reach a sample count) [V-s], and CAE
(arXiv:2305.01507, vigilance estimated by a determinantal-point-process criterion) [A].
P1 N. P2 –. P3 N\*. P4 P. P5 –: it tunes one of its own thresholds, not its operations. [I] P6 P.

**8. ART survey.** Not a system. It is the verified source for rows 1–7, for the field's definition of
stability ("no prototype vector can take on a previous value after it has changed, and … an infinite
presentation of inputs results in forming a finite number of clusters"), and for the field's own list
of open problems (§7). [V]

**9. Growing Neural Gas.** The winner and its graph neighbours move toward each input, an edge
links winner and runner-up and edges age out; every λ inputs a unit is inserted between the unit
with the largest accumulated error and its worst neighbour. [M] Abstract: an incremental model with
a simple Hebb-like rule and no parameters that change over time, which keeps adding units and
connections until a performance criterion is met. [A]
P1 N: every input updates; growth is on a timer rather than on the mismatch itself. [M] P2 –. P3 N\*:
prototypes and their graph emerge; clusters are connected components. P4 –. P5 –. P6 P.

**10. Grow When Required.** A node is inserted halfway between the best-matching unit and the input
whenever the unit's activity exp(−distance) is below an insertion threshold *and* the unit is already
habituated, i.e. has fired often. Habituation counters also scale each unit's learning rate, so
well-trained units change little. [V-s] (equations as restated in arXiv:1805.10966 §3.1)
P1 N. P2 –: habituation is a plasticity dial, not a belief. [I] P3 N\*. P4 –: a mismatch is measured
but nothing is constructed and judged. [I] P5 –. P6 P.

**11. Growing dual-memory (GDM).** Two recurrent GWR networks: an episodic memory that grows on
prediction mismatch, and a semantic memory that creates a neuron only when its best unit predicts
the wrong label and updates only when the label agrees. Temporal links between episodic units let
the system replay internally generated activation trajectories to both memories "in the absence of
sensory input"; the input features come from a VGG model pre-trained on ILSVRC-2012. [V]
P1 P: replay happens after each learning episode and the feature extractor is pre-trained. [V] P2 P:
each neuron keeps a histogram of the labels it won. [V] P3 N\*: instances then categories, with labels
supplied. P4 P: self-generated replay exists, but no measure of how real the replayed material is. [I]
P5 –. P6 P.

**12. SOINN.** A node is inserted when the input lies outside the adaptive similarity threshold of
the nearest or second-nearest node; otherwise the winner moves, an edge is added, and low-density
nodes are periodically removed as noise. [M] It uses two layers with different thresholds, and
deciding when to stop the first layer is an acknowledged problem that the single-layer ESOINN
(2007) was built to remove. [A]
P1 P for the two-layer original (the table entry), N for the single-layer successor. P2 –. P3 N\*.
P4 –. P5 –. P6 P.

**13. HTM sequence memory.** Columns of cells carry sparse binary codes; a cell whose dendritic
segment matched the previous activity becomes predictive, and "if none of the cells in a winning
column are in a predictive state, all cells in that column become active"; a new segment is created
on a little-used cell of such an unpredicted column, "only when there is no match to any existing
segment". Each synapse has a scalar permanence that starts at 0.21, "below the connection threshold
(0.5)", is raised by 0.1 when its segment is reinforced, and segments that predicted a cell which
then stayed inactive receive "a very small decay" (0.01). [V]
P1 N: "it learns from every new data point"; "continuous online learning". [V] P2 P: a first
exposure is stored below threshold and needs three further reinforcements to become effective
(0.21 + 3 × 0.1 = 0.51) [C]; a failed prediction is weakened slowly, not deleted, and the model
keeps "multiple predictions until there is sufficient disambiguating evidence" [V]; there is no
rule level. [I] P3 P: context-specific codes and segments form on need, but "specialized encoders
are required to encode real-world data". [V] P4 P: every step sets the model's own prediction
against the input, and anomaly detection is a named application [V]; nothing is constructed
offline. [I] P5 –. P6 P: sparse binary matrices; tested on symbol sequences and taxi demand, read
out through a softmax classifier, no language. [V]

**14. Cascade-correlation.** Training starts with no hidden units; when the error stalls, candidate
units are trained to correlate with the residual error, the best is installed with its input-side
weights frozen, and it receives the inputs and all earlier hidden units. The network becomes a deep
cascade of permanent feature detectors. [A] + [M]
P1 –: alternating batch phases and freezing. P2 –. P3 P: features are created when needed, then
frozen. P4 –. P5 –. P6 P.

**15. Progressive neural networks.** A new network column is added for each task; earlier columns
are frozen and feed the new one through lateral connections. The method is "immune to catastrophic
forgetting", but "a downside of the approach is the growth in number of parameters with the number
of tasks" and "choosing which column to use for inference requires knowledge of the task label". [V]
P1 –. P2 –. P3 –. P4 –. P5 –: features transfer, procedures do not. [I] P6 P.

**16. DEN.** For each new task it selectively retrains the sub-network connected to the new output;
if the loss stays above τ it adds k units per layer under group sparsity and drops the useless ones,
and it duplicates any unit whose incoming weights drifted by more than σ. Units are timestamped and
"each task will only use the parameters that were introduced up to stage t". [V]
P1 –: per-task batch optimisation and task identity at inference. P2 –. P3 P. P4 –. P5 –. P6 P.

**17. Neurogenesis deep learning.** The per-level reconstruction error of a stacked autoencoder
flags outlier samples; when too many exceed a user-specified threshold, new neurons are added at
that level and trained on the outliers. Old neurons are stabilised by "intrinsic replay" of samples
generated from stored statistics of earlier classes. [V]
P1 –. P2 –. P3 P. P4 P: the system's own reconstruction of the input is measured and drives growth;
the replayed samples themselves are not measured. [I] P5 –. P6 P.

**18. CN-DPM.** A Dirichlet-process mixture of neural experts, each a classifier plus a density
model; a sample is assigned by responsibility ∝ (expert's count) × likelihood, or to "new" ∝ α ×
prior likelihood. Samples judged new go to a short-term memory, and when it is full "we stop the
data inflow for a while and train a new expert with the data in the STM for multiple epochs" (the
"sleep phase"). [V]
P1 P: sample-wise and task-free, but with sleep phases. P2 P: soft responsibilities and counts.
P3 P: experts emerge per data cluster. P4 P: each expert's generative likelihood decides whether a
sample is explained. P5 –. P6 P.

**19. CURL.** A variational autoencoder with a mixture-of-Gaussians latent whose components act as
concepts; samples whose ELBO is below a threshold are buffered, and when the buffer reaches a set
size a new component is initialised from the closest existing one and tuned on the buffer. Because
encoder and decoder are shared, "learning new tasks interferes with previous ones", which is fought
with mixture generative replay from a snapshot of the model. [V]
P1 P: task-free, but gradient training with snapshots and replay. P2 –. P3 N\*: concept components
emerge; the encoder is shared and learned. P4 P: its own generative fit gates growth, and it trains
on its own samples without checking them. [I] P5 –. P6 P.

**20. CRP mixture / rational model of categorization.** Each new item joins an existing cluster with
probability proportional to the cluster's size (times likelihood) or opens a new cluster with
probability proportional to α, so "the predictive distribution … allows for new data to be assigned
to a previously unseen cluster". The same prior "was independently discovered by Anderson (1991) in
the context of his rational model of categorization". [V] (arXiv:1106.2697 §2.2)
P1 P: the prior is sequential, but "the basic computational problem … is computing the posterior",
which is not available in closed form. [V] P2 P: beliefs are probabilities and counts refine. P3 N\*.
P4 –. P5 –. P6 P.

**21. Hierarchical Pitman-Yor language model.** Each n-gram context has its own Pitman-Yor
"restaurant" whose base distribution is the distribution of the shorter context, so a context's
prediction is its own discounted counts plus probability mass handed down from its parent.
Inference is Gibbs sampling over seating arrangements, and interpolated Kneser-Ney smoothing is
recovered as an approximation of the model. [V]
P1 –: batch Gibbs sampling (125 burn-in + 175 sampling iterations, about 1.5 hours on 14M words). [V]
P2 P: graded counts, and the general level supports the specific level; structurally this is the
"rule feeds the instance" pattern, but for n-gram contexts. [V] / [I] P3 P: unbounded tables, fixed
vocabulary of 17,964 words. [V] P4 –. P5 –. P6 P: a real language model made of counts, trigram
perplexity 114.3 against 113.6 for modified Kneser-Ney. [V]

**22. Overhypothesis model (hierarchical Bayes).** Each category has its own feature distribution
θ_c drawn from a shared Dirichlet with parameters αβ, where β captures the distribution across
categories and α how much the feature varies within each category; both are inferred. [A] Each
setting of (α, β) is an overhypothesis, i.e. a learned rule about what categories in general are
like, which then constrains what is inferred about sparsely observed or new categories. [A] + [M]
P1 –: posterior inference over all data, not incremental weight change. P2 N†: graded beliefs at the
instance and the category level with top-down support, held as posterior probabilities rather than
weights; how strong a first hearing is depends on the hyperprior (§6.1). [C] P3 P: categories are
given in the basic model; the paper's second model groups categories into ontological kinds. [A]
P4 –. P5 P: learning to learn; later categories are learned faster (§6.1), with a fixed inference
procedure. [C] / [I] P6 P: equations, no network, no language.

**23. Bayesian word segmentation.** Words are drawn from a Dirichlet process (the bigram version
from a hierarchical DP) whose base distribution generates novel words phoneme by phoneme, so
segmenting an unsegmented corpus and building the lexicon are a single inference. Inference is
annealed Gibbs sampling over boundary positions for 20,000 iterations. [V]
P1 –: batch. The online particle-filter version reaches lexicon F-scores of 35–47 against 59.7 for
batch MCMC on the unigram model, and its results "vary considerably". [V] (U11-1004, Table 1)
P2 P. P3 N: the lexicon itself is formed when needed; α0 "determines the total probability of
generating any novel word, a probability that decreases as more data is observed, but never
disappears". [V] P4 –. P5 –. P6 P.

**24. SUSTAIN.** The model "initially assumes a simple category structure" [A] and recruits an
additional cluster when it is confronted with a surprising event; "newly recruited clusters are
available to explain future events and can themselves evolve into prototypes/attractors/rules". [A]
(The second quotation was returned by search for the 2003 companion paper by Gureckis and Love; that
the 2004 abstract contains the same sentence is from memory. [M]) Otherwise the winning cluster
moves toward the item and attention to each stimulus dimension is tuned. [M]
P1 N: trial-by-trial learning. [M] P2 P: graded, and an exception gets its own cluster instead of
overwriting; no rule level. P3 N\*. P4 –. P5 –. P6 P.

**25. Cobweb language models.** Each instance (an anchor word plus a context window) is sorted
down a tree of probabilistic concepts whose nodes store attribute-value counts; at each node the
system "simulates each operation" (add to a child, create a child, merge two children, split a
child), "computes the metric" (category utility) and keeps the best. "Learning is interleaved with
performance"; Cobweb/4L predicts by combining several nodes found by best-first search. [V]
P1 N. [V] P2 P: counts are graded beliefs at instance nodes and category nodes, and multi-node
prediction mixes the levels. [V] P3 N\*: the concept hierarchy is formed from instances; word
symbols are given. P4 P: it evaluates its own candidate restructurings before committing, by
predictive utility on stored counts. [V] / [I] P5 –: four fixed operators. P6 P: trees and counts,
not tensors; 0.38 accuracy on the 5-option MSR sentence completion task after about one third of
the Sherlock Holmes training text (BERT trained on all of it: 0.31; chance 0.20). [V]

## 3. Supporting works (read or located, not in the main table)

| Work | Year, id | Status | What it adds to the map |
|---|---|---|---|
| Kulis & Jordan, Revisiting k-means: new algorithms via Bayesian nonparametrics (DP-means) | 2012, arXiv:1111.0352 | [V] | The small-variance limit of a DP mixture is k-means with a hard threshold λ for opening a cluster; ties the Bayesian prior to an ART-style vigilance. |
| Miller & Harrison, A simple example of DP mixture inconsistency for the number of components | 2013, arXiv:1301.2708 | [V] | The DPM posterior on the number of clusters does not converge to the true number; with one true component the posterior probability of one cluster goes to 0. |
| Griffiths & Ghahramani, Indian buffet process; Kessler et al., Hierarchical Indian buffet neural networks for Bayesian continual learning | 2005–11 [V-s]; 2021, arXiv:1912.02290 [A] | mixed | IBP: unbounded number of latent features per object. H-IBP puts that prior on hidden units so layer width grows and shrinks across tasks. |
| Jerfel et al., Reconciling meta-learning and continual learning with online mixtures of tasks | 2019, arXiv:1812.06080 | [V] (parts) | A CRP mixture of hierarchical-Bayes priors over neural-network parameters; a new cluster of tasks is spawned when a threshold is crossed. The closest tensor instantiation of "overhypotheses + CRP", but at the level of tasks, not statements. [I] |
| Pritzel et al., Neural episodic control | 2017, arXiv:1703.01988 | [V] (parts) | Append-only key-value memory (differentiable neural dictionary) read by k-nearest neighbours; at capacity the item that least recently showed up as a neighbour is overwritten. No novelty test. |
| Allen et al., Infinite mixture prototypes | 2019, arXiv:1902.04552 | [V] (parts) | DP-means inside episodic few-shot learning: a new prototype when the distance to all existing ones exceeds λ. |
| Bricken et al., Sparse distributed memory is a continual learner | 2023, arXiv:2303.11934 | [V] (parts) | Top-K activation, no bias, L2-normalised non-negative weights; works on 256-d embeddings from a frozen ConvMixer pre-trained on ImageNet32. |
| Kemker & Kanan, FearNet | 2018, arXiv:1711.10563 | [A] | Recent-memory and long-term networks with pseudo-rehearsal during "sleep" and a module choosing which memory to use. |
| Bendale & Boult, Towards open world recognition | 2015, arXiv:1412.5687 | [A] | Novel categories "must be continuously detected and then added"; Nearest Non-Outlier on ImageNet features. |
| Khandelwal et al., Nearest neighbor language models | 2020, arXiv:1911.00172 | [A] | An exemplar memory for language that can be extended without training, but its keys are the embeddings of a pre-trained LM. |
| Wang et al., SEMA; Yildirim et al., SERENA; Mitchell et al., Self-expanding NNs; Li et al., Learn to grow | 2024–25, arXiv:2403.18886; 2403.14684; 2307.04526; 1904.00310 | [A] | Current expansion methods: adapters added to a frozen ViT when a learned descriptor detects shift (SEMA); frozen "concept cells" inside one over-parameterised net (SERENA); a natural-gradient score for when and where to add neurons (SENN); architecture search over reuse / adapt / new per layer (Learn to grow). |
| Platt, A resource-allocating network for function interpolation (RAN) | 1991, Neural Computation 3(2):213–225 | [A] | A radial-basis network allocates a new unit when an input is both far from every centre and badly predicted, otherwise it takes a gradient step; the distance threshold shrinks over time. One-pass, but its units are basis functions of a mapping, not categories. |
| Tan et al., Fusion ART (and FALCON, its reinforcement-learning use) | 2007 | [V-s]; lineage [A] | Several input channels share one category field, so one category binds patterns across channels [V-s]; search results name a fusion-ART variant as the memory model inside a 2026 ART-plus-LLM system (ARTEM) [A]. |
| Hawkins & Ahmad, Why neurons have thousands of synapses | 2016, arXiv:1511.00083 | [A] | The neuron model behind row 13: most patterns a neuron recognises act as predictions that depolarise it without firing it. |
| Kasabov, evolving connectionist systems (ECOS, EFuNN) | 1998–2003 | [A] | One-pass creation of a rule node when no node is activated above a sensitivity threshold or the error exceeds a threshold; nodes are read as fuzzy IF-THEN rules. |
| Nosofsky, Palmeri & McKinley, RULEX | 1994, Psych. Review 101(1) | [A] | People form simple rules and remember occasional exceptions: the "rule plus exception" shape of the red example, as a psychological model. |
| Carpenter & Tan, Rule extraction (fuzzy ARTMAP); Tan, Cascade ARTMAP | 1995, Connection Science 7(1); 1997, IEEE TNN 8(2) | [A] | Each category node gets a confidence factor from usage and accuracy and low-confidence rules are pruned; Cascade ARTMAP inserts rules and chains them in multi-step inference. |
| Cairns et al., Streaming adversarial robustness in fuzzy ARTMAP | 2026, arXiv:2605.06902 | [V] | The most recent fuzzy ARTMAP evaluation found: USPS, MNIST, Fashion-MNIST, EMNIST-Letters; category counts and attack success rates quoted in §5. |
| Rios & Itti, Lifelong learning without a task oracle | 2020, arXiv:2011.04783 | [A] | Fuzzy ART / ARTMAP used as a task mapper on top of pre-trained feature extractors. |
| Grossberg, A path toward explainable AI and autonomous adaptive intelligence; Grossberg, …how our brains learn large language models and their meanings (ChatSOME) | 2020, Front. Neurorobotics; 2025, Front. Syst. Neurosci. | [A] | The theory's own claims, as summarised in search results: deep learning can forget catastrophically and is not explainable, whereas ART incrementally learns large non-stationary databases using only locally computable quantities; ChatSOME is a proposed ART-based account of learning language meanings. Search results showed descriptions only, no benchmark results; full text not read. |
| Dai et al., MemoryART | 2026, AAAI-26 | [A] | ART-based working / episodic / semantic memory modules wrapped around existing LLMs for long medical dialogues; the LLM does the language. |
| Marinescu, McCoy & Griffiths, Distilling symbolic priors for concept learning into neural networks; critique "Meta-learning neural mechanisms rather than Bayesian priors" | 2024, arXiv:2402.07035; 2025, arXiv:2503.16048 | [A] | A bridge from hierarchical Bayes to networks by offline meta-training on tasks sampled from the prior; disputed as to whether a prior is actually instilled. |
| Parisi et al., Continual lifelong learning with neural networks: a review | 2019, arXiv:1802.07569 | [V] | The outside view of this field from continual learning; source for several failure modes in §5. |

## 4. Reading by principle

**P1 (learning = running).** Native in ART and GWR [V-s], HTM sequence memory and Cobweb [V], RAN
[A] and SUSTAIN [M]: each input either adjusts a unit or allocates one, in one pass. The deep
expandable family is not: it works per task with repeated optimisation (DEN, progressive nets), in
sleep phases (CN-DPM) or with snapshots and replay (CURL). [V] Nonparametric Bayes is sequential in
its prior but batch in its inference (Gibbs sampling in rows 21 and 23). [V]

**P2 (graded beliefs, refine not delete, rule feeds instance).** Four partial forms exist. Counts:
ARTMAP-IC instance counting, GDM label histograms, CRP seating counts, Cobweb tables. [V-s] / [V]
A weak first storage that must be confirmed: HTM's synaptic permanence, which starts below its
connection threshold [V], and nothing else among the systems reviewed [I]. Exceptions kept beside
the general case: ARTMAP match tracking,
SUSTAIN clusters, RULEX. [V-s] / [A] Top-down support from a general level to a specific level:
hierarchical Bayes (row 22) and Pitman-Yor back-off (row 21). [A] / [V] The last form is found only
in systems that are not online, and no system has more than two of the four. [I]

**P3 (concepts formed when needed).** This is the field's home ground: every system row except 15
creates units from data. [I] Two limits recur. The feature basis over which categories form is given in ART,
SUSTAIN, the CRP models and Cobweb. [V-s] / [V] And the recent deep systems form their units on top
of a frozen, batch-trained backbone (GDM: VGG; SDM: ConvMixer; SEMA: ViT; kNN-LM: the LM itself).
[V] / [A] Row 23 is the one place where the vocabulary itself is formed rather than given. [V]

**P4 (imagination with a reality measure).** The nearest element is the ART match cycle: a
top-down expectation is compared with the input and a scalar match decides between learning and
search. [V-s] Second nearest is Cobweb simulating candidate restructurings and scoring them before
committing. [V] Replay (GDM, NDL, CURL [V]; FearNet [A]) produces internal constructions; in the
parts I read, none of them applies a measure of how real the replayed material is. [I] No system
reviewed here generates a construction offline, scores its reality, and uses that score to
advance. [I]

**P5 (acquiring the way of constructing).** The operators are fixed in every system: ART's
search / reset / match tracking, Cobweb's four operations, the CRP seating rule. [V-s] / [V] What can
be learned above the level of content is a threshold (IDAT) [A], a prior over what categories are
like (row 22) [A], or a set of priors or initialisations for clusters of tasks (Jerfel et al.) [V].
None of these is an operator, and none was shown to transfer across domains. [I]

**P6 (tensors and equations, language eventually).** All rows are equations; rows 15–19 are tensor
networks. [I] Language appears only as count models: HPYLM at Kneser-Ney level on 14M words (2006),
Bayesian word segmentation on 9,790 utterances (33,399 words) of child-directed speech, and
Cobweb/4L at 0.38 on a 5-option completion task. [V] ART's own survey lists "grammar inference and
natural language processing" as needing new representations. [V]

## 5. Known failure modes: what broke

### 5.1 Order dependence
- "An important problem faced by all agglomerative clustering or incremental learning algorithms,
  including ART, is order-dependence of data presentation. This is especially true in fast online
  learning mode", and "the presentation order of inputs still remains an open problem". [V]
  (arXiv:1905.11437 §6.1)
- The remedies listed there are pre- and post-processing, voting over several orderings, and
  merge / split / delete operations. [V] Voting over orderings is not available to a system that
  lives through one stream once. [I]
- Cobweb's merge and split exist to "mitigate the effects of training order, which can influence
  the hierarchy structure". [V] Particle-filter learners of CRP models "vary considerably" between
  runs. [V] (U11-1004)

### 5.2 Category proliferation and unbounded growth
- "The winner-take-all category selection process … can sometimes lead to category proliferation
  and is one of the limiting factors of ART's capacity for mapping complex relations." [V]
- Measured in 2026: vanilla fuzzy ARTMAP at ρ_a = 0.9 with fast learning reaches 94.9% clean
  accuracy on MNIST with about 56 thousand categories, 80.7% on Fashion-MNIST with 57 thousand,
  80.8% on EMNIST-Letters with 114 thousand, and 92.3% on USPS with 6 thousand. [V]
  (arXiv:2605.06902, Table 1) The training sets hold 60,000, 60,000, 124,800 and 7,291 images [V],
  so the network keeps roughly 0.93, 0.95, 0.91 and 0.82 categories per training image: on raw
  pixels it is close to storing every example. [C] / [I] Lower-vigilance settings "yield poor clean
  accuracy". [V]
- Match tracking and noisy or overlapping classes are named as causes in the secondary literature. [A]
- The Bayesian counterpart: the DP-mixture posterior on the number of clusters is inconsistent, and
  practitioners "found that ignoring tiny clusters tends to mitigate this issue". [V] (arXiv:1301.2708)
- Progressive nets: parameters grow with the number of tasks and "only a fraction of the new
  capacity is actually utilized". [V] Classifier growth with the number of classes is flagged as a
  scalability issue. [V] (arXiv:1802.07569 §3.3) CN-DPM needs a pruning algorithm for redundant
  experts. [V]

### 5.3 The threshold that nobody can set
- "The vigilance is the single most important parameter in any ART model … it is often set
  empirically in an ad hoc manner … setting the vigilance parameter still remains a challenging
  task … particularly in the online learning mode." [V] "The problem of choosing the number of
  clusters is traded for the problem of choosing the vigilance value." [V]
- The same dial under other names: GWR insertion threshold a_T [V]; DEN's τ and σ [V]; NDL's
  "user-specified threshold" [V]; CURL's expansion threshold, swept by hand, with 12 to 71
  components depending on it [V]; CN-DPM's concentration α [V]; DP-means' λ [V].
- 2023–2025 ART work still targets exactly this (CAE, IDAT). [A]

### 5.4 Stability bought with rigidity
- Under fast learning a fuzzy ART category can only expand, so a category that has become too
  general is never revised; the only repair is an exception category created by match
  tracking. [V-s] / [I]
- Standard fuzzy ARTMAP cannot store two labels for one identical input (§6.2). [C]
- Cascade-correlation freezes input weights [A]; progressive nets freeze columns [V]; SERENA freezes
  each "concept cell" once learned [A]. A frozen unit cannot be a revisable belief. [I]

### 5.5 Local codes and the missing bridge to gradient learning
- Extending ART to distributed codes is open; the difficulty is "to maintain the desirable speed
  and stability of those ART systems in the presence of this distributed representation". [V]
  Whether the split between match-based (Hebbian / ART) and error-based (backpropagation) learning
  is a false one "still lacks a definitive resolution". [V] (arXiv:1905.11437 §6.4–6.5)

### 5.6 Forgetting returns as soon as features are shared
- CURL: the shared encoder is why generative replay is needed. [V] GDM needs replay. [V]
- The workaround in current systems is a frozen pre-trained backbone (GDM, SDM, SEMA, Rios & Itti,
  SOINN on pre-trained CNN features). [V] / [A] The review states the cost: "by relying on fixed
  representations … the discrimination power will be conditioned by the dataset used to train the
  feature extractor". [V] (arXiv:1802.07569)

### 5.7 Scale and evaluation
- As of 2019, for ART "an extensive evaluation with recent lifelong learning benchmarks has not
  been reported". [V] (arXiv:1802.07569) A 2026 fuzzy ARTMAP paper says of its four small image
  benchmarks that they "are not intended to define a large-scale vision leaderboard; rather, they
  serve as controlled streaming testbeds". [V] (arXiv:2605.06902 §6) My searches found no ART result
  on a modern large benchmark without a pre-trained feature extractor. [I]
- CN-DPM and CURL are demonstrated on MNIST, SVHN, CIFAR and Omniglot. [V] / [A]
- Nonparametric Bayes scales through inference cost: batch Gibbs for the language models [V], and
  particle filters "are known to perform rather poorly in high dimensional state spaces". [V]

### 5.8 Language
- No system in this field produces fluent language. [I] The existing language results are listed
  under P6 in §4. [V]
- ART: a grammar variant (GramART, tree templates over a Backus-Naur grammar) [V-s]; models of
  speech and word recognition [M]; the 2023–2025 ChatSOME proposal, for which my searches returned
  conceptual descriptions and no benchmark results [A]; and ART modules used as memory around an
  LLM (MemoryART) [A].

### 5.9 The match score is not a safe reality measure
- A white-box attack aligned with ARTMAP's own competition reaches 89–100% success on vanilla
  fuzzy ARTMAP. [V] After one defence, "adversarial examples can attain higher match scores than
  clean samples, and the corresponding match-threshold detector can degrade below chance". [V]
  (arXiv:2605.06902) Relevant to P4: the field's one built-in self-measure can be inverted. [I]

## 6. The closest existing system on the "red" example

Owner's example: "red is dark" 0.23 → "dark but sometimes light" 0.57 → "all colours are sometimes
dark, sometimes light" 0.77 → red 0.88 → the question "which other colours exist?" arises.

**Answer: the hierarchical Bayesian overhypothesis model (row 22), with a CRP over categories (row
20), is the closest in what it computes.** Hierarchical Bayes is the only family in the field in
which a category-level regularity is induced from instances and then sends graded support back to
an instance (row 21 is the same pattern applied to n-gram contexts). [I] The closest *online,
neural* system is ARTMAP-IC (row 3); it reproduces the graded instance belief and nothing above
it. [C] HTM sequence memory (row 13) is the only system with a sub-threshold first hearing; it is
treated in §6.3. [I] Both traces follow. The scripts are `tools-art-growth/red_example_hbm.py` and
`tools-art-growth/red_example_artmap.py` in this directory; each runs in a few seconds with
`python3 -I <script>` and needs only the standard library.

Event stream used for all traces [C]: E1 hear "red is dark" once; E2 red seen light once; E3 red
seen dark again; E4 blue dark once, light once; E5 green dark once, light twice; E6 yellow dark
twice, light once.

### 6.1 What the hierarchical Bayesian model does, step by step

Model [C], binary case of the bags-of-marbles model: θ_c ~ Beta(αβ, α(1−β)) for each colour c, each
observation of c is dark with probability θ_c, α ~ Exponential(rate λ), β ~ Uniform(0,1). The prior
family is written from memory of the paper [M], not copied from it. Small α means each colour is
almost always dark or almost always light; large α means every colour sits near the common rate β.
Beliefs are read off the posterior: "red is dark" as a generic = P(θ_red ≥ 0.9); "red is sometimes
dark, sometimes light" = P(0.1 < θ_red < 0.9); the category rule "a colour is sometimes dark,
sometimes light" = the same interval probability for a colour not yet seen. The cut-offs 0.1 and
0.9 are my choice. With (0.05, 0.95) or (0.2, 0.8) the numbers move (for λ = 1, red's "sometimes /
sometimes" belief is 0.952 or 0.699 after E3 and 0.976 or 0.768 after E6) but every step below
keeps its direction. [C]

Results for λ = 1 (prior favours homogeneous categories), all [C]:

| After | P(next red is dark) | B["red is dark"] | B[red: sometimes / sometimes] | B[rule: any colour sometimes / sometimes] |
|---|---|---|---|---|
| E1 | 0.865 | 0.678 | 0.314 | 0.314 |
| E2 | 0.500 | 0.060 | 0.881 | 0.492 |
| E3 | 0.631 | 0.113 | 0.881 | 0.492 |
| E4 | 0.621 | 0.094 | 0.901 | 0.595 |
| E5 | 0.603 | 0.074 | 0.921 | 0.665 |
| E6 | 0.608 | 0.069 | 0.926 | 0.714 |
| control: red's own evidence only, no other colours | 0.631 | 0.113 | 0.881 | – |

Results for λ = 0.1 (prior does not favour homogeneous categories), all [C]: B["red is dark"] after
E1 = 0.356; B[red: sometimes / sometimes] = 0.634 → 0.929 (E2, E3) → 0.982 (E6), control 0.929; rule
= 0.634 → 0.805 → 0.943.

What this shows:

1. **First hearing.** The model has no notion of "stored weakly". After one hearing it believes the
   generic "red is dark" at 0.68 (λ = 1) or 0.36 (λ = 0.1); the number is set by the hyperprior, and
   the λ = 1 prior, which is the one I recall the paper using [M], produces a *strong* first belief,
   the opposite of the owner's 0.23. [C]
2. **Contradiction.** One light red collapses the generic to 0.06 and raises "sometimes /
   sometimes" to 0.88. The belief is refined into a new shape, not deleted, but far more abruptly
   than 0.23 → 0.57, because the model treats every report as a reliable observation. [C]
3. **Rule induction.** With each further colour that is seen both ways, the belief that an unseen
   colour is "sometimes / sometimes" rises 0.49 → 0.60 → 0.67 → 0.71. This is the induced category
   rule; in the model it is the posterior over α and β, not a stored statement. [C]
4. **Feedback to the instance.** Red's "sometimes / sometimes" belief rises from 0.881 to 0.926 with
   no new evidence about red (control 0.881). The top-down effect is real and small here (+0.045),
   because red's own evidence had nearly settled the matter. [C]
5. **Where the rule is strong.** A brand-new colour heard once as dark after E6 gets
   B[sometimes / sometimes] = 0.699 and B[generic dark] = 0.292, against 0.314 and 0.678 for the
   very first colour ever heard once. The acquired rule changes how the next instance is learned. [C]
6. **The question.** Nothing arises. With a CRP over colour types the model holds a number, the
   probability that the next token is a colour not yet seen: 0.083 for concentration 1 after 11
   tokens of 4 types. It is a standing expectation that more colours exist, with no mechanism that
   turns it into a query. [C] / [I]

A variant that is *not* the published model [C]: if each report is flipped with probability 0.2 and
λ = 0.1, the trajectory becomes gradual: B["red is dark"] 0.287 after E1; B[red: sometimes /
sometimes] 0.702 after E2–E3; rule 0.794 after E6; red 0.817 after E6 (control 0.702, so +0.115 from
the rule alone). This is the owner's shape (instance, then rule, then instance lifted above the
rule), and it is obtained only by adding a reliability parameter that the model does not have. [C] / [I]

What it lacks against the principles [I]:
- P1: every row of the table is a full posterior recomputation over all data so far. There are no
  weights and no incremental update; online approximations (particle filters) are noisy (§5.1).
- P3: colours as categories and dark / light as the feature are given to it. The two-level form of
  the hypothesis space is fixed by the modeller; it cannot invent the qualifier "sometimes".
- P4: none. P5: the transfer in point 5 is a learned prior, with a fixed inference procedure.
- P6: no tensor network, no language.

### 6.2 What ARTMAP-IC does on the same stream

Minimal implementation from the survey's equations, colours one-hot and complement-coded, fast
learning, map-field vigilance 1. All [C]; this is not the authors' code.

All numbers in this table are P(dark) for the named colour.

| Configuration | Red after E1 | Red after E2 | Red after E3 | End state after E6 | A colour never seen |
|---|---|---|---|---|---|
| High vigilance (0.9), MT−, instance counting | 1.000 | 0.500 | 0.667 | 8 categories, one per (colour, label) pair, with counts; red stays 0.667 | 0.545, the pooled base rate, because all categories tie |
| Low vigilance (0), MT−, instance counting | 1.000 | 0.500 | 0.667 | boxes merge into {red, blue, green}→dark (count 4) and {red, blue, green}→light (count 4), plus two yellow categories; red becomes 0.500 | 0.667 from the nearest categories (yellow's), 0.552 over all nodes |
| Standard fuzzy ARTMAP (MT+) | 1.000 | 1.000 | 1.000 | (red, light) is never stored; 2 of the 11 pairs are dropped, (red, light) and (yellow, light); red stays 1.000 | 0.000 from the nearest box, 0.615 over all nodes |

Reading [C] / [I]:
- **First hearing** is stored at full strength: the category template equals the input and
  P(dark given red) = 1.0. Weakness exists only in the count (1).
- **Contradiction** under MT− creates a second category with the same template mapped to "light";
  the belief is the count ratio. This is "refine, not delete" in its plainest form.
- **Rule.** At high vigilance there is no rule: colours never interact, and an unseen colour gets
  the pooled base rate only because all categories tie. At low vigilance the boxes merge into "these
  colours → dark" and "these colours → light", which is the rule; but red's own 2:1 record is
  dissolved into the 4:4 pool. Generalising erases the instance.
- **Feedback.** There is no configuration in which a category-level unit and an instance-level unit
  both persist and the first sends graded support to the second. Distributed prediction over all
  nodes does pull red from 0.667 to 0.564 at high vigilance, but that is similarity-weighted
  smoothing by the other colours' instance categories, applied at read-out only; no weight changes.
- **Question.** None. An uncommitted node always takes part in the competition [V-s], which is a
  standing readiness for novelty, not a query.
- **Standard fuzzy ARTMAP** ignores the contradiction: match tracking raises vigilance above 1 and
  no node can resonate. [C] That the original algorithm ends the search without learning in this
  case, rather than doing something else, is from memory of the 1992 paper. [M] The survey notes
  complement coding is used to avoid vigilance above 1 [V-s]; it cannot avoid it for an identical
  input with a different label, which is what MT− and "inconsistent cases" in the ARTMAP-IC title
  address [A].
- Slow map-field learning (survey eq. 105) worked by hand [C]: weights start at 1 for every label
  and the unobserved labels decay by (1−β) per event, so P(dark) = 1/(2−β) after the first hearing
  (0.53 for β = 0.1), 0.5 after the contradiction, 1/(2−β) after the third event. That is a graded,
  recency-weighted estimate with a weak first hearing, still without any rule level.

### 6.3 Four other candidates, briefly

- **HTM sequence memory.** Reading the stream as transitions (red, then its tone), from the
  verified rules and parameters and without running the code [I]: the first "red then dark" creates
  a segment whose synapses sit at 0.21, below 0.5, so nothing is predicted yet; three further
  repetitions make it effective. This is the only system here whose first hearing is sub-threshold.
  "Red then light" finds no matching segment and gets a segment of its own; once both are
  connected the model predicts dark and light together after red, each occurrence of one costing the
  other 0.01. That is "dark but sometimes light" as two parallel predictions without stated
  proportions. Colours do not interact, so there is no rule, no feedback and no question.
- **Cobweb.** Category utility on the final 11 instances, with attributes colour and tone, is
  0.259 for splitting the root by tone (dark things / light things) and 0.196 for splitting by
  colour. [C] (`tools-art-growth/cobweb_cu_check.py`, using the formula printed in arXiv:2212.11937)
  The criterion therefore prefers a top level of "dark things" and "light things"; a node for "red"
  holding a graded darkness belief is not guaranteed to form. [I] An incremental run is greedy and
  order-dependent and was not simulated here.
- **SUSTAIN.** The first event creates a cluster (red, dark); the surprising (red, light) recruits
  a second cluster; responses are graded through the clusters' association weights. Attention to
  the colour dimension would fall if it does not predict tone. No rule, no feedback, no question. [M] / [I]
- **ARTMAP information fusion.** The inconsistent labels for red would be learned as a one-to-many
  prediction, and relations among labels would be derived afterwards. It would yield "all colours
  ⇒ sometimes dark" only if "colour" were itself one of the supplied labels, and nothing would
  return to red. [A] / [I]

## 7. What is still unsolved

### 7.1 In the field's own words
Items 1–5 are the five open problems of the ART survey (arXiv:1905.11437, its sections 6.1–6.5).
1. Input-order dependence in online mode. [V] (survey 6.1)
2. Setting or adapting the vigilance online. [V] (survey 6.2) Equivalently the insertion / expansion
   threshold and the DP concentration elsewhere (this note §5.3). [I]
3. Representations and metrics for structured data, explicitly including "grammar inference and
   natural language processing". [V] (survey 6.3)
4. Distributed representations that keep ART's speed and stability. [V] (survey 6.4)
5. A system that does both match-based and error-based learning. [V] (survey 6.5)
6. Evaluation of ART-type learners on modern lifelong-learning benchmarks. [V] (arXiv:1802.07569)
7. Growth that stays bounded: capacity that grows with tasks or classes, redundant units,
   inconsistent cluster counts. [V] (this note §5.2)
8. Representation learning inside the growing system instead of a frozen pre-trained
   backbone. [V] / [I] (this note §5.6)
9. Scalable online inference for nonparametric Bayesian models. [V] (this note §5.7)
10. When, where and how much to grow without a hand-set threshold; a 2023 answer is a
    natural-gradient expansion score. [A] (arXiv:2307.04526)

### 7.2 Not solved because the field has not posed it
These are gaps against P1–P6. For each, "not found" means not found in the 30 searches and the
texts read here; it is absence of evidence, not proof of absence. All [I].

1. **Rule-to-instance feedback inside an online growing network.** The nearest pieces are separate:
   hierarchical Bayes has the feedback but is not online; ARTMAP-IC is online but has no rule level;
   HPYLM back-off has the feedback for n-gram contexts under batch sampling; Cobweb/4L mixes levels
   only at prediction time.
2. **A belief strength attached to a statement, starting low and earned.** ART fast learning stores
   the first exposure fully; the hierarchical Bayesian model's first belief is whatever the
   hyperprior makes it. HTM sequence memory does start a new synapse below threshold and requires
   confirmation, but the strength belongs to a synapse between cells, there is one fixed starting
   value for everything, and nothing above the synapse reads or revises it.
3. **Refinement that changes the form of a belief.** "Dark" becoming "dark but sometimes light"
   appears as a second category (ART, SUSTAIN) or as posterior mass moving inside a fixed
   hypothesis space (hierarchical Bayes). No system constructs the qualified statement as a new
   object.
4. **Allocation pressure turned into a question.** The CRP gives the probability of an unseen
   category, ART keeps an uncommitted node ready, Bayesian word segmentation keeps a never-vanishing
   probability of a novel word. A search for self-generated questions in ARTMAP / GWR systems
   returned nothing relevant.
5. **A reality measure applied to self-generated constructions.** The ART match needs a present
   input; replay systems generate without measuring.
6. **Operators as learned objects (P5).** No system here learns, improves or transfers its own
   revise / generalise / feed-back / ask operations.
7. **Language at scale from a growing, non-pretrained substrate.** The field's language systems
   are count models of 2006–2024 at small scale; everything recent that touches fluent language
   attaches a growing memory to a pre-trained Transformer.

## 8. Vocabulary bridge (the field's names for the owner's ideas)

| Owner's term | Field's terms to search or cite |
|---|---|
| P1 one process | online / incremental / one-pass learning; task-free continual learning; match-based learning; fast learning |
| P2 graded belief | instance counting; confidence factor; slow map-field learning; synaptic permanence and connection threshold (HTM); posterior; hierarchical shrinkage; back-off |
| P2 refine not delete | match tracking; inconsistent cases; rule plus exception; one-to-many learning |
| P2 rule feeds instance | overhypothesis; hierarchical Bayesian model; learning to learn |
| P3 formed when needed | uncommitted node; category commitment; novelty-triggered allocation; nonparametric prior (CRP, IBP); concept formation |
| P4 measuring own construction | resonance; match function; vigilance test; category utility; reconstruction error; pseudo-rehearsal / intrinsic replay (the generating half only) |
| P5 way of constructing | vigilance adaptation; meta-learning; overhypotheses (all weaker than the owner's sense) |

## 9. Sources

### Read in full text or in the cited parts ([V])
- arXiv:1905.11437 Brito da Silva, Elnabarawy, Wunsch — https://arxiv.org/abs/1905.11437
- arXiv:1802.07569 Parisi, Kemker, Part, Kanan, Wermter — https://arxiv.org/abs/1802.07569
- arXiv:1805.10966 Parisi, Tani, Weber, Wermter — https://arxiv.org/abs/1805.10966
- arXiv:1708.01547 Yoon, Yang, Lee, Hwang — https://arxiv.org/abs/1708.01547
- arXiv:1606.04671 Rusu et al. — https://arxiv.org/abs/1606.04671
- arXiv:1612.03770 Draelos et al. — https://arxiv.org/abs/1612.03770
- arXiv:2001.00689 Lee, Ha, Zhang, Kim — https://arxiv.org/abs/2001.00689
- arXiv:1910.14481 Rao et al. — https://arxiv.org/abs/1910.14481
- arXiv:1106.2697 Gershman, Blei — https://arxiv.org/abs/1106.2697
- arXiv:1111.0352 Kulis, Jordan — https://arxiv.org/abs/1111.0352
- arXiv:1301.2708 Miller, Harrison — https://arxiv.org/abs/1301.2708
- arXiv:2212.11937 MacLellan, Matsakis, Langley — https://arxiv.org/abs/2212.11937
- arXiv:2409.12440 Lian, Baglodi, MacLellan — https://arxiv.org/abs/2409.12440
- arXiv:2605.06902 Cairns, Brito da Silva, Petrenko, Wunsch, Liu — https://arxiv.org/abs/2605.06902
- arXiv:1512.05463 Cui, Ahmad, Hawkins — https://arxiv.org/abs/1512.05463 (not from a search:
  recalled, then downloaded and read)
- arXiv:2303.11934 Bricken et al.; arXiv:1703.01988 Pritzel et al.; arXiv:1902.04552 Allen et al.;
  arXiv:1812.06080 Jerfel et al. (parts) — https://arxiv.org/abs/2303.11934,
  https://arxiv.org/abs/1703.01988, https://arxiv.org/abs/1902.04552, https://arxiv.org/abs/1812.06080
- ACL P06-1124 Teh — https://aclanthology.org/P06-1124/
- ACL P06-1085 Goldwater, Griffiths, Johnson — https://aclanthology.org/P06-1085/
- ACL U11-1004 Börschinger, Johnson — https://aclanthology.org/U11-1004/

### Abstract only ([A]), arXiv
arXiv:2511.17983, 2305.01507, 2201.10713 (Masuyama et al.); 2403.03835 (Lian et al.); 2410.10588
(MacLellan et al.); 1912.02290 (Kessler et al.); 1904.00310 (Li et al.); 2307.04526 (Mitchell et
al.); 2403.18886 (Wang et al.); 2403.14684 (Yildirim et al.); 1711.10563 (Kemker, Kanan); 1412.5687
(Bendale, Boult); 2011.04783 (Rios, Itti); 1911.00172 (Khandelwal et al.); 1511.00083 (Hawkins,
Ahmad); 2402.07035 and 2503.16048 (located by search, not downloaded). Each at
https://arxiv.org/abs/<id>.

### Abstract or snippet only ([A]), not on arXiv
- Carpenter, Martens, Ogas, BU technical report record — https://open.bu.edu/items/84f4153e-79da-4295-a755-674e2da8651c
- Carpenter, Markuzon, ARTMAP-IC record — https://open.bu.edu/items/db445a5a-53ed-44a8-acae-a6cf520e6140
- Carpenter, Tan, Rule extraction record — https://open.bu.edu/items/02c191bc-af48-46a5-bbbb-b45cd11a6b2a
- Fritzke, GNG abstract — https://papers.nips.cc/paper/1994/hash/d56b9fc4b0f1be8871f5e1c40c0067e7-Abstract.html
- Fahlman, Lebiere abstract — https://papers.nips.cc/paper/1989/hash/69adc1e107f7f7d035d7baf04342e1ca-Abstract.html
- Love, Medin, Gureckis, SUSTAIN — https://www.gureckislab.org/papers/ref/love2004sustain
- Kemp, Perfors, Tenenbaum 2007 (course copy located by search) — https://sites.socsci.uci.edu/~lpearl/courses/readings/KempPerforsTenenbaum2007_LearningOverhypotheses.pdf
- Platt 1991 (copy located by search) — https://sci2s.ugr.es/keel/pdf/algorithm/articulo/plat1991.pdf
- Marsland, Shapiro, Nehmzow 2002 (copy located by search) — https://vision.unipv.it/AI/aa2013-2014/A%20self-organising%20network%20that%20grows%20when%20required.pdf
- Grossberg 2020 — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7330174/
- Grossberg 2025 — https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2025.1630151/full
- Dai et al., MemoryART — https://ojs.aaai.org/index.php/AAAI/article/view/39205
- Kasabov, ECOS — https://kedri.aut.ac.nz/__data/assets/pdf_file/0018/23328/ecos-iconip98.pdf
- Nosofsky, Palmeri, McKinley 1994, RULEX — located by search, DOI 10.1037/0033-295x.101.1.53

### Search log (30 queries; 19 and 22 in extended mode)
1 ART survey for engineering applications. 2 fuzzy ARTMAP category proliferation, order dependence,
vigilance, noise. 3 GNG, grow-when-required, SOINN comparison. 4 dynamically expandable networks.
5 ARTMAP information fusion and hierarchical knowledge discovery. 6 learning overhypotheses with
hierarchical Bayesian models. 7 SUSTAIN cluster recruitment. 8 neural Dirichlet process mixture,
task-free continual learning. 9 Cobweb probabilistic concept formation language model. 10 ART
topological clustering continual learning. 11 cascade-correlation limitations. 12 resource-allocating
network novelty criterion; ECOS. 13 ART + deep learning hybrids 2024–2025, lifelong benchmarks.
14 ART and large language model memory 2025. 15 SOINN two-layer, noise removal. 16 neurogenesis-
inspired continual learning 2024–2025. 17 ChatSOME. 18 MemoryART. 19 ARTMAP / fuzzy ART with deep
feature extractors on CIFAR / ImageNet 2023–2025. 20 ARTMAP-IC inconsistent cases. 21 DP mixture
inconsistency. 22 ChatSOME simulations / benchmarks. 23 rational approximations to rational models.
24 Grossberg 2020 explainable AI. 25 rule extraction from fuzzy ARTMAP; Cascade ARTMAP. 26 Kasabov
ECOS / EFuNN. 27 ARTMAP / GWR systems that generate their own questions. 28 RULEX. 29 nested CRP /
HDP online concept hierarchies 2023–2025. 30 overhypotheses distilled into neural networks.

### Files in this directory
- `art-growth.md` — this note.
- `tools-art-growth/red_example_hbm.py`, `tools-art-growth/red_example_artmap.py` — the two traces in §6.
- `tools-art-growth/cobweb_cu_check.py` — the category-utility comparison in §6.3.
- `tools-art-growth/fetch_arxiv.sh`, `tools-art-growth/absmeta.py` — download and metadata helpers.
- `sources-art-growth/<id>/` — downloaded PDFs and their text, one directory per paper.
