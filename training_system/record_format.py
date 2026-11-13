"""Next-token record layout shared by every writer of the formatted training cache.

Layout version: ``training_system.cache_identity.CACHE_FORMAT_VERSION`` (2). Each formatted record is
the one-position shift of a single token sequence::

    sequence = [BOS if the record starts its document] + ids + [EOS if the record ends its document]
    input    = sequence[:-1] + PAD...      target = sequence[1:] + PAD...      (target[i] == input[i + 1])

A document that fits in one record keeps the classic ``[BOS, t1..tN]`` / ``[t1..tN, EOS]`` pair. When a
long document is split into overlapping chunks, only its first chunk starts with BOS and only its last
chunk ends with EOS: a continuation chunk does not teach "the text ends here" while the text goes on,
and its first token is context only (the overlap with the previous chunk already trained it).
"""


def record_boundaries(records):
    """``[(starts_document, ends_document), ...]`` for records in the order they were produced.

    ``TokenizerCore.load_training_data`` emits every source (a TXT/DOCX file, or one question-answer
    pair) as a run of consecutive records in text order; the overlapping chunks of a long text share
    the source's id. Only the first record of a run starts the document and only the last one ends it.
    A record without a source id (2-tuple) is a complete document. If an id reappears after another
    source, the boundaries are unknown and a ValueError is raised instead of inventing a false EOS.
    """
    sources = [item[2] if len(item) == 3 else None for item in records]
    seen, flags = set(), []
    for i, source in enumerate(sources):
        if source is None:
            flags.append((True, True))
            continue
        starts = i == 0 or sources[i - 1] != source
        if starts and source in seen:
            raise ValueError(f"records of source_id={source!r} are not consecutive; document boundaries are unknown")
        seen.add(source)
        flags.append((starts, i == len(sources) - 1 or sources[i + 1] != source))
    return flags


def format_autoregressive_record(ids, *, starts_document, ends_document, max_seq_length,
                                 bos_id, eos_id, pad_id, vocab_size, unk_id):
    """``(input, target)`` of one record, both right-padded with PAD to ``max_seq_length``.

    Out-of-vocabulary ids become UNK. A record that does not fit raises ValueError instead of being
    cut: cutting it and keeping EOS would move the document's end to a false place, and cutting it
    without EOS would silently drop its tail. Chunk the text to the same length first (the tokenizer
    splits records longer than ``max_seq_length - 2`` tokens).
    """
    ids = [i if 0 <= i < vocab_size else unk_id for i in ids]
    sequence = ([bos_id] if starts_document else []) + ids + ([eos_id] if ends_document else [])
    if len(sequence) < 2:
        raise ValueError("record has no token to predict")
    positions = len(sequence) - 1
    if positions > max_seq_length:
        raise ValueError(f"record needs {positions} positions but max_seq_length={max_seq_length}; chunk it first")
    padding = [pad_id] * (max_seq_length - positions)
    return sequence[:-1] + padding, sequence[1:] + padding
