"""Dimension rules shared by the decoder and its configuration schema.

These describe the existing implementation, not a numbered architecture variant.
"""


def resolve_ffn_dim(embed_dim, ffn_dim=None, *, gated=True):
    if not isinstance(embed_dim, int) or isinstance(embed_dim, bool) or embed_dim <= 0:
        raise ValueError("embed_dim must be a positive integer")
    if ffn_dim is not None:
        if not isinstance(ffn_dim, int) or isinstance(ffn_dim, bool) or ffn_dim <= 0:
            raise ValueError("ffn_dim must be a positive integer")
        return ffn_dim
    if not gated:
        return 4 * embed_dim
    # Preserve the decoder's existing checkpoint geometry.
    raw = int(2 / 3 * 4 * embed_dim)
    return ((raw + 255) // 256) * 256
