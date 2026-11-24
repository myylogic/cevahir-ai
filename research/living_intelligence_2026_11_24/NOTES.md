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

The Transformer runs are in progress; their rows and this table are added when they finish.

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

## 6. Next: M2, the harmony

1. The word level: a cortex over word numbers, with the lexicon turning word candidates into letter predictions.
2. The episode expert: the longest earlier occurrence beyond the cortex's depth.
3. The harmony's verdict by the rule, against a gradient-learned gate.

Measure each part's share, on the same four texts and against the same baselines. The target is the cortex's
prequential bits below PPMd's while keeping the substance exact.

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
python -m pytest ../../tests/evolution/test_living_cortex.py
```

The rows of `results/m1_cortex.jsonl` written before the `--update` option existed have no `update` field; they are
full updates, and the current code reproduces them (en-pydoc, D = 8: 1.5907).

Dawid, A. P. (1984). Present position and potential developments: some personal views. Statistical theory: the
prequential approach. *Journal of the Royal Statistical Society A*, 147(2), 278–292.
