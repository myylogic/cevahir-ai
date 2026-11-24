# Living Intelligence: notes

Study of 24 November 2026. The design is in [MODEL.md](MODEL.md), the code in [`living/`](../../living), the
measurements in [results/](results). Every number below is in a results file and has a command that reproduces it
(section 7).

## 1. What was built (M1)

| file | what it is |
|---|---|
| `living/evidence.py` | the one rule, unchanged; the novelty event; the speaking line; the identity of the rule with the Krichevsky–Trofimov estimator |
| `living/cortex.py` | the cortex: contexts as origins, the walk with exclusion, novelty judged by the situation's record, growth with m = 1 or 2, exact retraction, the substance digest |
| `living/lexicon.py` | words as paths of letters, numbered by the mind in the order it met them and spelled back from their numbers |
| `living/mind.py` | channels: the mind's own alphabets, a new channel born by the rule, the unfamiliar, structure, watching and speaking |
| `corpora.py` | the four texts, built the same way every time |
| `m1_cortex.py`, `m1_baselines.py`, `m1_transformer.py`, `m1_channels.py`, `m1_retract.py` | the measurements |
| `tests/evolution/test_living_cortex.py` | 13 tests: the rule equals KT and zihin.evidence; distributions are whole and agree with prediction; prequential bits; new symbols; exact retraction; deferred growth; the lexicon; a new channel; noise watched in silence |

## 2. The cortex on four texts

### 2.1 The texts and how they are lived

| corpus | what | documents | characters | distinct symbols |
|---|---|---|---|---|
| en-repo | the repository's English research text at commit 996cc2d: three notes of 14 Nov, the notes of 15, 22 and 23 Nov, ZIHIN_THEORY.md, PAPER.md, README.md | 9 | 943,651 | 141 |
| en-pydoc | the Python 3.13 reference documentation (`pydoc_data.topics`), 81 topics | 81 | 560,058 | 118 |
| tr-lessons | the public Turkish lessons the open mind lived (minds/sources.json) | 36 | 39,188 | 76 |
| tr-private | the author's own Turkish documents, read locally; counts only | 20 | 1,951,426 | 165 |

The SHA-256 of each public stream is in its rows. The documents of a corpus are joined by a blank line into one
stream, and the stream is lived once, in order. Each character is predicted before it is learned, so the bits of the
whole stream are a prequential code length (Dawid 1984), directly comparable to an adaptive compressor's output.

The stream is split at 90 % of its characters, and the last 10 % is measured two ways:

- **living**: the store keeps learning through it;
- **frozen**: the store as it was at the split, not learning (the setting of a model trained and then evaluated).

Unless a row says otherwise, the cortex has depth D = 8, growth m = 1, novelty judged by the situation, exclusion, and
full updates.

### 2.2 Whose record judges novelty

All-stream bits per character, D = 8, m = 1 (`results/m1_cortex.jsonl`):

| novelty judged by | en-pydoc | en-repo | tr-lessons |
|---|---|---|---|
| the context's own record, (t + ½)/(n + 1), with exclusion | 1.969 | 2.264 | 2.525 |
| the context's own record, blending instead of exclusion | 1.674 | 2.019 | 2.062 |
| PPM-C's escape, t/(n + t) (Moffat 1990) | 1.723 | 2.048 | 2.212 |
| PPM-D's escape, t/(2n) (Howard 1993) | 1.723 | 2.041 | 2.269 |
| **the situation's record** (the rule on the record of every context in the same situation) | **1.591** | **1.976** | **1.946** |

The rule applied to a context's own novelty record is the worst of these. Novelty falls as a context lives, so the
past rate overstates the future one (MODEL.md, 2.3). The same rule applied to the record of the situation is the best
of them, ahead of the two classical escape estimators. The rule did not change; the origin whose record it reads did.

**What exactness costs.** PPM normally updates only the contexts from the longest down to the one that knew the
symbol (update exclusion). The counts then depend on the order of life, and exact retraction is lost. Measured with
`--update found`:

| | en-pydoc | en-repo | tr-lessons |
|---|---|---|---|
| full updates (exact) | 1.591 | 1.976 | 1.946 |
| update exclusion | 1.553 | 1.956 | 1.857 |

Exactness costs 0.02–0.09 bits per character. The cortex keeps full updates.

### 2.3 Depth and growth

| corpus | D | all | test living | test frozen | nodes | ties | β nodes | β ties | seconds | memory (Python) |
|---|---|---|---|---|---|---|---|---|---|---|
| en-pydoc | 4 | 1.753 | 1.684 | 1.921 | 55,031 | 126,033 | 0.50 | 0.55 | 2.1 | 75 MB |
| | 6 | 1.638 | 1.578 | 1.875 | 235,457 | 384,767 | 0.59 | 0.63 | 2.9 | 137 MB |
| | 8 | 1.591 | 1.548 | 1.898 | 572,558 | 794,901 | 0.66 | 0.69 | 4.3 | 251 MB |
| | 12 | 1.545 | 1.532 | 1.911 | 1,624,725 | 1,942,139 | 0.74 | 0.75 | 7.7 | 635 MB |
| | 16 | 1.520 | 1.519 | 1.915 | 2,973,206 | 3,335,866 | 0.78 | 0.79 | 11.9 | 1,129 MB |
| en-repo | 4 | 2.043 | 1.967 | 2.174 | 88,512 | 223,243 | 0.46 | 0.54 | 3.3 | 120 MB |
| | 6 | 1.968 | 1.795 | 2.026 | 444,025 | 754,783 | 0.60 | 0.65 | 5.8 | 242 MB |
| | 8 | 1.976 | 1.741 | 1.978 | 1,154,769 | 1,637,427 | 0.70 | 0.73 | 9.4 | 482 MB |
| | 12 | 1.992 | 1.696 | 1.934 | 3,469,071 | 4,173,534 | 0.80 | 0.82 | 17.0 | 1,317 MB |
| | 16 | 1.994 | 1.679 | 1.917 | 6,453,870 | 7,252,704 | 0.86 | 0.87 | 24.0 | 2,401 MB |
| tr-lessons | 4 | 2.094 | 1.925 | 2.870 | 12,887 | 24,684 | 0.73 | 0.82 | 0.1 | |
| | 8 | 1.946 | 1.658 | 2.809 | 76,494 | 97,924 | 0.91 | 0.92 | 0.3 | |
| | 12 | 1.942 | 1.636 | 2.799 | 170,761 | 197,297 | 0.94 | 0.94 | 0.4 | |
| tr-private | 8 | 1.616 | 1.989 | 2.578 | 1,505,465 | 2,219,463 | 0.74 | 0.77 | 17.6 | 799 MB |
| | 12 | 1.661 | 2.006 | 2.598 | 5,174,466 | 6,392,214 | 0.84 | 0.85 | 32.7 | 2,083 MB |

β is the slope of log(nodes) or log(ties) against log(characters) over the last four doublings of the stream. The
times are one processor core. "Memory" is the peak resident memory of the Python process (the Turkish lessons' rows
are dominated by the interpreter and the imports).

- **Learning while living matters.** On every corpus the living test bits are below the frozen ones, by 0.24 (en-repo)
  to 1.15 (tr-lessons) bits per character at D = 8.
- **Depth.** Longer contexts help where the text repeats (en-pydoc, the test part of en-repo) and cost where the
  documents differ (the whole of en-repo, tr-private). D = 8 is the default.
- **Growth.** The store grows sublinearly, and less so as D rises, because long contexts are nearly all unique. At
  D = 8: β = 0.66 (en-pydoc), 0.70 (en-repo), 0.74 (tr-private). The 39,000 characters of the Turkish lessons are too
  few to leave the linear start.
- **The lexicon grows more slowly still**: β = 0.47 (en-pydoc), 0.40 (en-repo), 0.67 (tr-private) for its nodes. The
  lexicon of en-repo holds 6,738 words in 18,459 nodes; 2,276 of the words were met once.

How en-repo's store grew (D = 8):

| characters lived | 8,192 | 65,536 | 262,144 | 524,288 | 943,651 |
|---|---|---|---|---|---|
| cortex nodes | 32,229 | 178,870 | 504,492 | 798,021 | 1,154,769 |
| nodes per character | 3.93 | 2.73 | 1.92 | 1.52 | 1.22 |
| lexicon words | 530 | 2,032 | 3,956 | 5,073 | 6,738 |

**Against Zihin's growth.** en-repo's store holds 1.74 ties per character. In a compact layout (12 bytes per tie,
4 per node) that is about 26 bytes per character. Zihin's growth law (12.3 ties per word at 68 bytes, 5.6 characters
per word in this text) is about 149 bytes per character. MODEL.md 3.4 gives the projection to a trillion words and why
it is only a projection.

### 2.4 Waiting for the second meeting (m = 2)

| corpus | nodes m = 1 | nodes m = 2 | pending marks | ties m = 1 | ties m = 2 | bits m = 1 | bits m = 2 |
|---|---|---|---|---|---|---|---|
| en-pydoc | 572,558 | 182,589 | 118,767 | 794,901 | 379,134 | 1.591 | 1.699 |
| en-repo | 1,154,769 | 299,464 | 240,744 | 1,637,427 | 707,031 | 1.976 | 1.987 |
| tr-lessons | 76,494 | 20,884 | 12,943 | 97,924 | 38,286 | 1.946 | 2.325 |
| tr-private | 1,505,465 | 476,424 | 328,678 | 2,219,463 | 1,102,858 | 1.616 | 1.639 |

Waiting for the second meeting divides the nodes by 3.1–3.9 and the ties by 2.0–2.6. Even counting each pending mark
as a tie, the store is 1.5–1.9 times smaller. It costs 0.01–0.11 bits per character on the larger texts, and 0.38 on
the Turkish lessons, where much is met only once. The order of life then shapes the store, so exact retraction is not
offered with m = 2 (section 4).

### 2.5 Words: seen, unseen, and met again

Bits per character of the words of the test part, by whether the word occurred in the first 90 % (D = 8):

| corpus | seen, living | seen, frozen | unseen, living | unseen, frozen | unseen words |
|---|---|---|---|---|---|
| en-pydoc | 1.648 | 1.952 | 3.069 | 4.135 | 331 of 7,427 |
| en-repo | 1.547 | 1.646 | 3.316 | 4.398 | 1,043 of 16,148 |
| tr-lessons | 1.444 | 1.834 | 2.046 | 3.809 | 283 of 541 |
| tr-private | 1.607 | 1.875 | 2.750 | 3.891 | 6,941 of 25,917 |

A word never seen is not an unknown symbol. It costs 1.4–2.1 times as much as a known one, and living through the test
part lowers that cost by 1.1–1.8 bits per character: by the second time a new word appears there, the store has
already learned part of it.

The same thing over the whole stream: bits per character of a word at its k-th meeting.

| k | 1 | 2 | 3 | 4 | 5 | 6–10 | 11–100 | 101+ |
|---|---|---|---|---|---|---|---|---|
| en-pydoc | 4.00 | 1.99 | 1.85 | 1.71 | 1.70 | 1.59 | 1.38 | 1.38 |
| en-repo | 4.35 | 2.59 | 2.33 | 2.23 | 2.16 | 2.08 | 1.87 | 1.76 |
| tr-lessons | 4.24 | 1.12 | 1.42 | 1.34 | 1.36 | 1.25 | 1.35 | 1.56 |
| tr-private | 2.85 | 1.77 | 1.61 | 1.53 | 1.49 | 1.42 | 1.27 | 1.28 |

Most of what a word costs is paid at its first meeting; the second meeting costs 26–62 % of the first. In the Turkish
lessons, the second meeting is the cheapest of all, because a lesson's answer repeats the words of its question.

**The numbering.** A few words with the numbers en-repo's mind gave them, spelled back from the number alone:

| word | number | spelled back | belief that the path is a whole word |
|---|---|---|---|
| mind | 19 | mind | 0.969 |
| the | 91 | the | 0.885 |
| evidence | 297 | evidence | 0.996 |
| bir | 844 | bir | 0.767 |
| zihin | 2,614 | zihin | 0.991 |
| ve | 659 | ve | 0.119 |

"ve" is mostly the beginning of longer words in this English text ("very", "version"), so the belief that the path
"ve" is a whole word is low. In the Turkish lessons, the same path is a word with belief 0.752.

### 2.6 Against compressors and a Transformer

Bits per character (`results/m1_baselines.jsonl`, `results/m1_transformer.jsonl`). "all" is the whole stream lived
once; "test" is the last 10 %. For the compressors, "test" is size(all) − size(first 90 %); for the offline
Transformer it is frozen.

| | en-repo all | en-repo test | en-pydoc all | en-pydoc test | tr-lessons all | tr-lessons test | tr-private all | tr-private test |
|---|---|---|---|---|---|---|---|---|
| **cortex, D = 8, living** | **1.976** | **1.741** | **1.591** | **1.548** | **1.946** | **1.658** | **1.616** | **1.989** |
| cortex, D = 8, frozen at the split | | 1.978 | | 1.898 | | 2.809 | | 2.578 |
| PPMd, order 8 | 1.793 | 1.573 | 1.429 | 1.392 | 1.763 | 1.531 | 1.474 | 1.781 |
| PPMd, order 16 | 1.751 | 1.444 | 1.312 | 1.307 | 1.733 | 1.486 | 1.448 | 1.744 |
| xz -9e | 2.101 | 1.700 | 1.544 | 1.532 | 2.044 | 1.747 | 1.843 | 2.067 |
| bzip2 -9 | 2.025 | 1.941 | 1.607 | 1.518 | 2.217 | 1.999 | 1.730 | 1.984 |
| gzip -9 | 2.692 | 2.764 | 2.262 | 2.111 | 2.174 | 1.917 | 2.376 | 2.516 |
| Transformer, online (living) | 2.896 | 2.349 | 2.654 | 1.811 | 4.030 | 3.432 | 2.880 | 3.076 |
| Transformer, offline, frozen test | | 2.907 | | 2.047 | | 3.610 | | 3.549 |
| Transformer, offline for 60 minutes, frozen test | | 2.152 | | | | | | |

**What each side had.**

- **The cortex** read each stream once, in order. It took 0.3 s (tr-lessons) to 17.6 s (tr-private) of one processor
  core, learned from nothing, and paid for every new symbol.
- **The offline Transformer** (`m1_transformer.py --mode offline`) trained on the first 90 % minus a held-out 5 %,
  for 20 minutes of four processor cores (8 on the Turkish lessons). That is 834–1,382 steps of 32 × 256 characters.
  The checkpoint with the best held-out bits was then evaluated on the last 10 %, frozen. On the three larger texts,
  its held-out bits were still falling when the time ran out (en-repo: 2.97, 2.84, 2.77, 2.76 at steps 800, 1,000,
  1,200 and 1,232).
- **The online Transformer** lived through the stream in blocks of 1,024 characters. It predicted each block first,
  then trained on everything seen so far, with the steps spread so that the run took 15 minutes (8 on the Turkish
  lessons). That gave 1,609–2,033 steps of 16 × 256 characters. On tr-private, one step per block is the floor, and
  the run took 983 s.
- **Both Transformers** were given the vocabulary of the whole stream in advance.
- **A longer run.** One offline run on en-repo had 60 minutes: 3,990 steps. Its held-out bits were still falling
  slowly (1.948, 1.943 and 1.937 at steps 3,600, 3,800 and 3,990). On the last 10 % it reached 2.152, against 1.978
  for the cortex frozen at the split and 1.741 for the cortex living.
- **A shared machine.** The 20-minute en-repo run shared the four cores with the cortex's single-core runs for a few
  minutes. It made 1,232 steps where the 60-minute run's pace would give about 1,330.

On these texts and with these budgets, the cortex codes every stream and every last tenth better than the
Transformer, offline or online, and it does so in seconds instead of minutes. It is no evidence about large data and
large compute, where Transformers are far ahead of every model here (MODEL.md, section 11). It is evidence about the
regime this study is about, learning while living, where a mind has to make use of what it met once or twice.

**Learning a word while living.** Bits per character of a word at its k-th meeting in the stream: the cortex against
the online Transformer, which lived the same stream in the same order.

| k | 1 | 2 | 3 | 4 | 5 | 6–10 | 11–100 | 101+ |
|---|---|---|---|---|---|---|---|---|
| en-repo, cortex | 4.35 | 2.59 | 2.33 | 2.23 | 2.16 | 2.08 | 1.87 | 1.76 |
| en-repo, Transformer | 4.70 | 4.44 | 4.28 | 4.20 | 4.14 | 4.02 | 3.33 | 2.25 |
| en-pydoc, cortex | 4.00 | 1.99 | 1.85 | 1.71 | 1.70 | 1.59 | 1.38 | 1.38 |
| en-pydoc, Transformer | 4.47 | 4.28 | 4.11 | 3.94 | 3.87 | 3.64 | 2.85 | 1.93 |
| tr-lessons, cortex | 4.24 | 1.12 | 1.42 | 1.34 | 1.36 | 1.25 | 1.35 | 1.56 |
| tr-lessons, Transformer | 4.77 | 4.58 | 4.61 | 4.45 | 4.48 | 4.30 | 3.45 | 1.90 |
| tr-private, cortex | 2.85 | 1.77 | 1.61 | 1.53 | 1.49 | 1.42 | 1.27 | 1.28 |
| tr-private, Transformer | 3.64 | 3.45 | 3.36 | 3.28 | 3.22 | 3.13 | 2.72 | 1.96 |

The cortex pays for a word once. By its second meeting a word costs 26–62 % of its first. The Transformer pays almost
the full price again at the second meeting (94–96 % of the first). After more than a hundred meetings it comes to
1.90–2.25, about where the cortex is at the second meeting (1.12–2.59).

## 3. A new channel, a language outside the world's languages, and noise

`m1_channels.py`, `results/m1_channels.jsonl`. The mind first lives 504,052 characters of the Python documentation
on its declared channel "text". Then three streams arrive without a name, in segments of 256 symbols.

- **A made-up language in an unused script.** About 400 words in six classes, a plural suffix and a verb ending that
  agrees with the subject, written in 22 symbols of Unicode's Private Use Area: 200,000 symbols.
- **Noise.** 22 other private-use symbols, uniform and independent: 100,000 symbols.
- **The same made-up language in Latin letters.** The same words and grammar, other sentences: 100,000 symbols.

| stream | where it went | bits per symbol | said | right when said | first said at | structure at the end |
|---|---|---|---|---|---|---|
| made-up language, unused script | a new channel, born at its first segment (all 200,000 symbols) | 1.424 | 62.5 % | 83.5 % | symbol 302 | 0.935 |
| noise, unused script | another new channel (all 100,000) | 4.613 | 0 | — | never | 0.431 |
| made-up language, Latin letters | the text channel (all 100,000; no new channel) | 1.779 | 62.4 % | 79.2 % | symbol 5 | 0.894 |

On the same streams, bzip2, xz and PPMd (order 8, on the UTF-8 bytes) code the made-up language at 1.443, 2.033 and
1.956 bits per symbol in the unused script, and 1.568, 1.962 and 1.401 in Latin letters. Noise costs at least
log₂ 22 = 4.459 bits per symbol, and the mind pays 4.613 while it learns that there is nothing to learn.

- **A new channel.** The belief that the first segment of the unused script was new to the text channel was above the
  speaking line, so a channel was born. Every later segment went to it. Noise in another unused script was new to
  both channels, so a third channel was born.
- **Watching, then speaking.** In the new channel the mind said nothing for 302 symbols. From then on it said its
  guess before 62.5 % of the symbols, and 83.5 % of what it said was right. The structure belief of the noise channel
  never reached the speaking line (0.431 at the end), so the mind watched 100,000 symbols of noise and never said
  anything.
- **The unfamiliar.** The same language in Latin letters went to the text channel, because it is new in nothing but
  its words. The mind recognised it as unfamiliar: the belief that its symbols were surprising was 0.788 on the first
  segment, above the line. It then fell as the mind learned: 0.239 after 10,000 symbols, 0.123 after 90,000.
- **Nothing old was disturbed by the new channels.** The bits of the held-out tenth of the Python documentation were
  1.898429 before and 1.898429 after the two new channels: identical, because a channel is its own origin. Learning
  the made-up language in Latin letters on the text channel itself moved them to 1.913192 (+0.8 %).

**Speaking a word.** Given 40 symbols of a sentence never seen, ending with the first letter of a word, the mind
continued while its guess reached the speaking line (200 words per channel):

| channel | whole word and the symbol after it | stopped silent before the end, correct so far | said something wrong |
|---|---|---|---|
| text (made-up language in Latin letters) | 56 | 119 | 25 |
| the new channel (unused script) | 56 | 105 | 39 |

Examples, in Latin letters (`before` | what the mind said | what came):

| before | said | was |
|---|---|---|
| `i banetu mi b` | `as ` | `as begfagi bi` |
| `derpe dad. d` | `o b` | `o bane gumulo` |
| `as mari. do d` | `e` | `ada dofelol. ba` |
| `debtu dase g` | (silence) | `a motekpe. te` |

The cortex completes a word it has evidence for and is silent where the first letter leaves several words open.
Choosing among them is the work of the word level and the focus (MODEL.md, sections 5–6), not of the cortex.

## 4. Exact retraction

`m1_retract.py`, `results/m1_retract.jsonl`: en-pydoc's 81 documents, D = 8, each read from an empty history.

| retracted document | characters | substance after retraction = substance of a store that never read it | situation records that differ | probe bits, retracted / never read |
|---|---|---|---|---|
| the first | 1,142 | identical (SHA-256) | 485 of 523 | 3.664145 / 3.663942 |
| the middle (41st) | 1,696 | identical | 374 of 525 | 3.664013 / 3.663715 |
| the last | 816 | identical | 0 of 525 | 3.663918 / 3.663918 |
| the longest | 67,119 | identical | 422 of 525 | 3.680260 / 3.678757 |

The contexts and counts after a retraction are byte for byte those of a store that never read the document. The
situation records are trust learned from the store's own predictions while it lived. They hold the retracted
document's own events exactly, and those are taken back, but the later documents were predicted by a store that had
read it. That difference changes the bits of a probe the stores never read (the paper, 54,397 characters) by at most
0.0015 bits per character.

Retracting the longest document took 0.36 s. With m = 2 the store does not offer retraction: the order of life shapes
it.

## 5. What M1 does not show

- **The cortex alone is a PPM-class model.** At the same order it is 0.14–0.18 bits per character behind PPMd on the
  whole stream; PPMd has two decades of engineering. Information inheritance, count scaling and richer secondary
  estimation are left out because they cost exactness or add constants.
- **Against the other compressors.** On the whole stream the cortex is ahead of bzip2, xz and gzip on every text
  except xz on the Python documentation. On the last 10 % alone, xz or bzip2 is ahead on three of the four texts: xz's
  long matches and bzip2's block sorting reach further back than eight characters, which is the episode expert's work
  (M2).
- **No expert above the cortex yet.** There is no word level, no episode memory and no focus, so nothing chooses the
  next word. Section 3's silences at word starts show the gap the harmony and the PFC have to fill (M2, M3).
- **Small texts.** The corpora are 39 thousand to 2 million characters. The growth exponents are measured over at most
  four doublings, and the projection in MODEL.md is an extrapolation. A standard benchmark (enwik8, 100 MB, about
  36 MB compressed, from mattmahoney.net) needs the owner's permission to download.
- **Memory in Python** is about 420 bytes per node (en-repo, D = 8), against a compact layout's 16–26. Scale needs a
  compact implementation.
- **The Transformer's budget** is 20 minutes (offline) or 15 minutes (online) of four processor cores, with no GPU, and
  its size was not tuned beyond one choice per text size. On large data and with large compute, Transformers are far
  ahead of every model here (MODEL.md, section 11).
- **The made-up language** is generated by a small grammar. Real unknown languages have more structure and less
  regularity.

## 6. M2: the harmony

The owner's measure for the study is that it must pass the Transformer in every area. Section 2.6 shows the cortex
alone ahead of a character Transformer on these texts, but behind PPMd, and without a large text. M2 builds the
harmony of MODEL.md section 5 and adds a fifth, larger corpus.

### 6.1 What was built

- **`living/_core.c`, `living/fast.py`.** The cortex in C. Its bits are bit-identical to the Python cortex's,
  symbol for symbol, in both update modes; the tests check equality of the floating-point totals. It is about 12
  times faster: 0.34 s instead of 4.1 s for 504,052 characters.
- **The harmony** on top of it, as specified in MODEL.md section 5:
  - the origins: the cortex, every context d = 0 … 8, two word contexts, the episode, the line episode and the line
    context (the latest line, and the counts of every line, that began as the current line began), and a short-term
    memory of contexts 1–6 decaying by 1 − 1/F;
  - each origin's belief per bit by the rule with the coarser origin as prior;
  - two gates learned by the gradient of the code length;
  - the calibration by the rule, and a last gate;
  - two ways to speak without learning: saying (each symbol the most probable, while it reaches the speaking line
    or always), and recalling (locking onto the lived line that began as the cue begins, MODEL.md 6 and 8).
- **`code-pystdlib`, a fifth corpus.** The source of Python 3.13's standard library as installed: 537 files,
  10,302,143 characters, 252 symbols. It is the largest public text on the machine, ten times en-repo. The standard
  benchmark enwik8 could not be downloaded (section 6.6).
- **Two errors were found and fixed by the protocol itself.** Both are now tests
  (`tests/evolution/test_living_fast.py`):
  - A pass that does not learn wrote its text into the episode memory. The living pass that followed found the test
    text among its episodes, and its bits fell to a sham 0.002 per character.
  - The same pass changed the previous symbol's code, the word state and slots of the short-term memory.

  A pass that does not learn now leaves the mind exactly as it was. The rows below are from the fixed code.

### 6.2 What each part adds

`m2_ablation.py`, `results/m2_ablation.jsonl`. The parts are added one at a time. Whole stream / last tenth living:

| step | tr-lessons | en-pydoc | en-repo | tr-private |
|---|---|---|---|---|
| A, the cortex | 1.946 / 1.658 | 1.591 / 1.548 | 1.976 / 1.741 | 1.616 / 1.989 |
| B, + the episode, verdict by the rule | 1.933 / 1.628 | 1.445 / 1.427 | 1.930 / 1.615 | 1.596 / 1.964 |
| C, the harmony (every context and the episode, the gates, the calibration) | 1.807 / 1.518 | 1.311 / 1.281 | 1.749 / 1.458 | 1.457 / 1.774 |
| D, + the word contexts | 1.777 / 1.494 | 1.289 / 1.265 | 1.729 / 1.442 | 1.438 / 1.756 |
| E, + a second input per origin, the coarser origin as prior | 1.779 / 1.486 | 1.284 / 1.260 | 1.718 / 1.428 | 1.425 / 1.739 |
| F, + the short-term memory | 1.756 / 1.467 | 1.248 / 1.209 | 1.679 / 1.383 | 1.364 / 1.622 |
| **G, + the line episode and the line context (M2)** | **1.754 / 1.466** | **1.245 / 1.205** | **1.678 / 1.383** | **1.361 / 1.620** |

- **The gate is what pays most.** It gains 0.13–0.18 bits per character over the episode with only the verdict by
  the rule (B to C). This is the measure by which the gate, the one borrowed method, earns its place (MODEL.md 5.3).
- **The short-term memory** adds 0.02–0.06, most on the texts whose topics change (en-repo, tr-private).
- **The line episode and line context** add little to the bits (0.001–0.003), but they are what recall rests on
  (section 6.8).
- **From the cortex alone to M2,** the whole streams fall by 0.19 to 0.35 bits per character.

On code-pystdlib, the cortex alone holds 1.431 / 1.482, with the episode 1.287 / 1.365, and the full harmony
1.044 / 1.115: 0.39 bits per character less than the cortex alone.

### 6.3 Against compressors and a Transformer

`m2_harmony.py` (`results/m2_harmony.jsonl`), `m1_baselines.py`, `m1_transformer.py`. Whole stream / last tenth:

| | en-repo | en-pydoc | tr-lessons | tr-private | code-pystdlib |
|---|---|---|---|---|---|
| **harmony (M2), living** | **1.678 / 1.383** | **1.245 / 1.205** | **1.754 / 1.466** | **1.361 / 1.620** | **1.044 / 1.115** |
| harmony, frozen at the split | 1.669 | 1.678 | 2.705 | 2.218 | 1.874 |
| cortex alone (M1), living | 1.976 / 1.741 | 1.591 / 1.548 | 1.946 / 1.658 | 1.616 / 1.989 | 1.431 / 1.482 |
| PPMd, order 16 | 1.751 / 1.444 | 1.312 / 1.307 | 1.733 / 1.486 | 1.448 / 1.744 | 1.185 / 1.266 |
| PPMd, order 8 | 1.793 / 1.573 | 1.429 / 1.392 | 1.763 / 1.531 | 1.474 / 1.781 | 1.281 / 1.349 |
| xz -9e | 2.101 / 1.700 | 1.544 / 1.532 | 2.044 / 1.747 | 1.843 / 2.067 | 1.316 / 1.431 |
| bzip2 -9 | 2.025 / 1.941 | 1.607 / 1.518 | 2.217 / 1.999 | 1.730 / 1.984 | 1.460 / 1.619 |
| gzip -9 | 2.692 / 2.764 | 2.262 / 2.111 | 2.174 / 1.917 | 2.376 / 2.516 | 1.832 / 1.939 |
| Transformer, online (living) | 2.896 / 2.349 | 2.654 / 1.811 | 4.030 / 3.432 | 2.880 / 3.076 | (running) |
| Transformer, offline, frozen test | 2.907 | 2.047 | 3.610 | 3.549 | (running) |

The Transformer's rows for code-pystdlib (60 minutes offline and 60 minutes online, four cores) are running.

- **Against PPMd at order 16,** the strongest compressor measured, the harmony is ahead on nine of the ten cells. It
  leads by 0.07–0.14 bits per character on the whole streams of the four larger texts, and on every last tenth. The
  one cell lost is the whole stream of tr-lessons (section 6.6).
- **Against the Transformer:**
  - living through the stream, the harmony is ahead of the online Transformer by 1.2–2.3 bits per character on the
    whole streams and by 0.6–2.0 on the last tenths;
  - frozen at the split, it is ahead of the Transformer trained offline by 0.37–1.33 on the last tenths;
  - the Transformer trained offline for 60 minutes on en-repo reached 2.152 on the last tenth (section 2.6), and the
    harmony, frozen at the split, holds 1.669 there.

### 6.4 Words

The harmony's bits per character on the words of the last tenth, and by a word's k-th meeting in the stream:

| corpus | seen, living | unseen, living | unseen, frozen | k = 1 | k = 2 | k = 3 | 6–10 | 101+ |
|---|---|---|---|---|---|---|---|---|
| en-pydoc | 1.284 | 2.619 | 3.691 | 3.67 | 1.74 | 1.57 | 1.25 | 0.98 |
| en-repo | 1.207 | 2.757 | 3.841 | 3.99 | 2.40 | 2.12 | 1.82 | 1.42 |
| tr-lessons | 1.182 | 1.884 | 3.752 | 3.95 | 1.05 | 1.27 | 1.04 | 1.10 |
| tr-private | 1.304 | 2.278 | 3.422 | 2.50 | 1.55 | 1.39 | 1.18 | 1.04 |
| code-pystdlib | 1.271 | 1.544 | 4.116 | 3.44 | 1.90 | 1.67 | 1.33 | 1.06 |

Against the cortex alone (section 2.5), every column falls. A word met for the first time costs 0.29–0.36 bits per
character less, because the word contexts and the gate guess more of a new word from its beginning and its
neighbours.

### 6.5 Cost

| corpus | characters | seconds to live the first 90 % (one core) | nodes (with the word and line contexts) | peak memory |
|---|---|---|---|---|
| tr-lessons | 39,188 | 0.5 | 131,808 | (interpreter) |
| en-pydoc | 560,058 | 8.7 | 1,056,429 | 300 MB |
| en-repo | 943,651 | 17.9 | 2,268,053 | 392 MB |
| tr-private | 1,951,426 | 32.6 | 4,268,276 | 707 MB |
| code-pystdlib | 10,302,143 | 246.0 | 11,916,461 | 1,367 MB |

The harmony lives about 40,000–60,000 characters a second on one core, against about 1.5 million for the cortex
alone. code-pystdlib's run shared the machine with the Transformer's. A hundred million characters would take 30–40
minutes and, at this pace of growth, a few gigabytes.

### 6.7 Going on with a text never seen

`m2_say.py` (`results/m2_say.jsonl`); the Transformer's side is `m1_transformer.py --say 500`
(`results/m2_say_transformer.jsonl`). Both minds are frozen at the split. Each is given the same 500 places in the
last tenth, with the 256 characters before each, and goes on for up to 40 characters.

- **free**: always the most probable symbol; counted are the characters right before the first mistake;
- **speaking line**: silent from the first symbol below e*; counted are the characters said and those right.

| corpus | free: right before the first mistake | free: 10 or more right | line: said per place | line: right per place | line: right of said |
|---|---|---|---|---|---|
| en-repo | 6.36 | 18.8 % | 6.15 | 4.13 | 67.1 % |
| en-pydoc | 4.40 | 12.2 % | 6.22 | 3.38 | 54.4 % |
| tr-lessons | 1.55 | 3.0 % | 1.64 | 0.84 | 51.4 % |
| tr-private | 3.71 | 8.6 % | 4.77 | 3.09 | 64.9 % |
| code-pystdlib | 2.97 | 6.8 % | 2.93 | 1.97 | 67.1 % |

The Transformer's runs on the same places (trained offline for 20 minutes, 8 on tr-lessons) are running; their row follows.

Under the line, 51–67 % of what the harmony says is right. Its belief is at least e* ~ 0.583 at each symbol, and
the share right is near that belief, a little below it on the texts most unlike what it lived (the new lessons of
tr-lessons, the new topics of en-pydoc).

### 6.8 Recall after one reading: the race of the earlier study

`m2_race.py` (`results/m2_race.jsonl`) asks exactly what research/living_mind_2026_11_22/t_race.py asked (its
section 40), on the author's documents (`education/ozel-dokumanlar`; counts only):

- the program's own split into sentences: 22,601 sentences, 1,907,184 characters;
- 200 sentences of at least five words drawn with `random.Random(0)`, each cued by its first two words;
- scoring: exact (the rest of that sentence), valid (the rest of some sentence that begins with the same two
  words), silent, words in place, word F1.

The harmony reads the sentences once, one a line, in 32 seconds of one core, and answers in three ways:

| way | exact | valid | silent | words in place | word F1 | unique starts exact |
|---|---|---|---|---|---|---|
| free (always its most probable symbol) | 9 | 23 | 0 | 0.116 | 0.124 | 7 / 66 |
| under the speaking line | 1 | 9 | 136 | 0.032 | 0.050 | 0 / 66 |
| **recall (locked onto the lived line that began so)** | **73** | **190** | 0 | 0.410 | 0.447 | 57 / 66 |

- **Predicting is not recalling.** In free speech the harmony says what usually follows the cue's words in these
  documents, not the sentence it read. Lines that begin alike part soon in this text, and the harmony learned that.
- **Recall needs a goal.** With the goal of recalling, the prefrontal lock of MODEL.md section 6 holds the lived
  line that began as the cue begins (found and verified by the line episode, MODEL.md section 8) and says it to its
  end. That gives 190 valid answers of 200, and 57 of the 66 cues whose start is unique say exactly their own
  sentence. It locked on 186 cues; the other 14 went on freely.
- On a cue shared by several sentences, the lock takes the latest one: that is valid, but exact only by chance.
  This is why exact is 14 of 41 and 2 of 93 for starts shared by 2–3 and 4 or more sentences.

The word-level Transformer of t_race.py (60 minutes, four cores, asked at 2 to 60 minutes) and Zihin (t_race.py `mind`) on the same folder are running; their rows follow.

### 6.6 What M2 does not show

- **No standard large benchmark yet.** enwik8, which the owner approved, could not be downloaded: the session's proxy
  refuses mattmahoney.net (HTTP 403), although the environment allows it. That points to a policy above the
  environment. Wikipedia dumps on an allowed host are an alternative that needs the owner's choice.
- **Against Transformers trained offline with large compute on large data,** M2 has not been measured. The published
  enwik8 numbers (0.99 bits per character, a 277-million-parameter Transformer-XL on GPUs) remain the reference for
  that regime.
- **One cell is still lost to PPMd:** the whole stream of the smallest text (tr-lessons, 39,188 characters: 1.754
  against 1.733). PPMd's built-in priors win the first few thousand characters; the harmony wins the last tenth.
- **Prediction is one area.** Answering, holding a dialogue and reasoning through the new architecture come with the
  prefrontal focus and the word-level cortex (M3).
- **Constants.** The gate's learning rates, the 48 weight sets and the 24 calibration buckets are constants of the
  borrowed method and of the calibration, chosen once and not fitted per text. The short-term memory's decay is the
  theory's (F = 7).

## 7. Reproduce

From this folder (Python 3.13, `pip install pyppmd`):

```
python corpora.py                                   # the texts and their digests
python m1_cortex.py en-pydoc --depth 8              # one row of section 2 (see results/m1_cortex.jsonl for all)
python m1_cortex.py en-repo --depth 8 --update found
python m1_baselines.py                              # section 2.6, compressors
python m1_transformer.py en-repo --mode offline --minutes 20
python m1_transformer.py en-repo --mode online --minutes 15
python m1_channels.py                               # section 3
python m1_retract.py                                # section 4
python m2_harmony.py en-repo                        # section 6 (the harmony; a C compiler is needed)
python m2_ablation.py                               # section 6.2
python m1_transformer.py code-pystdlib --mode offline --minutes 60
python -m pytest ../../tests/evolution/test_living_cortex.py ../../tests/evolution/test_living_fast.py
```

The rows of `results/m1_cortex.jsonl` written before the `--update` option existed have no `update` field; they are
full updates, and the current code reproduces them (en-pydoc, D = 8: 1.5907).

Dawid, A. P. (1984). Present position and potential developments: some personal views. Statistical theory: the
prequential approach. *Journal of the Royal Statistical Society A*, 147(2), 278–292.
