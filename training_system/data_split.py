"""One source/content split contract for legacy and current training entry points."""
import random
import torch


def split_training_records(data, train_ratio=.8, seed=42, pad_token_id=0):
    if not data or not 0 < train_ratio < 1:
        raise ValueError("Split requires nonempty data and train_val_split between 0 and 1")
    if any(len(item) not in (2, 3) for item in data):
        raise ValueError("Invalid cache record structure")
    has_ids = [len(item) == 3 and item[2] is not None for item in data]
    if any(has_ids) and not all(has_ids):
        raise ValueError("Cache has mixed or missing source IDs")
    if not any(has_ids):
        data = [(item[0], item[1], i) for i, item in enumerate(data)]
    # source_id → indeksler
    source_to_indices = {}
    for i, item in enumerate(data):
        sid = item[2] if len(item) == 3 else None
        if sid not in source_to_indices:
            source_to_indices[sid] = []
        source_to_indices[sid].append(i)

    # Identical examples connect their source groups. Keep the transitive
    # component together so duplicates cannot cross train/validation.
    import hashlib
    parents = {sid: sid for sid in source_to_indices}
    def find(sid):
        while parents[sid] != sid:
            parents[sid] = parents[parents[sid]]
            sid = parents[sid]
        return sid
    first_source = {}
    pad_id = int(pad_token_id)
    for item in data:
        x, y = (part.tolist() if isinstance(part, torch.Tensor) else list(part) for part in item[:2])
        while x and y and x[-1] == y[-1] == pad_id:
            x.pop(); y.pop()
        signature = hashlib.sha256(repr((x, y)).encode("utf-8")).digest()
        sid = item[2]
        if signature in first_source:
            parents[find(sid)] = find(first_source[signature])
        else:
            first_source[signature] = sid
    grouped = {}
    for sid, indices in source_to_indices.items():
        grouped.setdefault(find(sid), []).extend(indices)
    source_to_indices = grouped

    # Source groups are shuffled deterministically.
    rng = random.Random(seed)
    source_ids = list(source_to_indices.keys())
    if len(source_ids) < 2:
        raise ValueError("At least two independent sources or content groups are needed for train/validation splitting")
    rng.shuffle(source_ids)

    # Train/val source_id split
    train_size = max(1, min(len(source_ids) - 1, int(train_ratio * len(source_ids))))
    train_source_ids = source_ids[:train_size]
    val_source_ids = source_ids[train_size:]

    # Chunk'ları topla
    train_indices = []
    for sid in train_source_ids:
        train_indices.extend(source_to_indices[sid])

    val_indices = []
    for sid in val_source_ids:
        val_indices.extend(source_to_indices[sid])


    def tensors(indices):
        result = []
        for i in indices:
            x, y = (torch.as_tensor(part, dtype=torch.long, device="cpu") for part in data[i][:2])
            if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or not x.numel():
                raise ValueError("Training records require nonempty, equally sized token vectors")
            result.append((x, y))
        return result
    return tensors(train_indices), tensors(val_indices)
