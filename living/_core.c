/* The cortex of living/cortex.py in C, for long lives (research/living_intelligence_2026_11_24, M2).
 *
 * The same store and the same arithmetic, operation for operation, so that the bits of every symbol are the same
 * floating-point numbers as the Python cortex's (escape "situation", mixing "exclude", growth m = 1, update "all" or
 * "found"). The contexts form a suffix tree: a context of length d is a child of its suffix of length d - 1, keyed
 * by (parent, the symbol one step further back). Symbols are numbered in the order the store met them.
 *
 * Above the cortex, the episode expert (MODEL.md, section 5) and the harmony's verdict by the rule:
 * - the episode: the longest earlier occurrence of the last MINLEN symbols (found through a hash of them, verified
 *   symbol by symbol) predicts the symbol that followed it;
 * - the verdict: when the episode predicts y, its belief is the rule on the record of the situation
 *   (match-length bucket, bucket of the cortex's own probability of y): P(y) = (R + 1/2) / (A + 1), and every other
 *   symbol shares 1 - P(y) in proportion to the cortex's distribution without y.
 *
 * Build: cc -O2 -shared -fPIC -o _core.so _core.c -lm (living/fast.py does it).
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NONE 0xFFFFFFFFu
#define UNICODE_SPACE 1114112.0

typedef struct {
    /* nodes: struct of arrays; node 0 is the root */
    uint32_t *n, *t, *head, *tail;
    uint64_t nodes, cap_nodes;
    /* entries: (symbol, count, next) linked per node in insertion order; index 0 is null */
    uint32_t *esym, *ecnt, *enext;
    uint64_t entries, cap_entries;
    /* child table: key (parent << 21 | symbol) + 1 -> node */
    uint64_t *ckey;
    uint32_t *cval;
    uint64_t cmask, cused;
    /* alphabet: code point -> number */
    uint64_t *akey;
    uint32_t *aval;
    uint64_t amask, aused;
    uint32_t *acode;           /* number -> code point */
    uint64_t asize, acap;
    /* situations */
    uint64_t *sit_new, *sit_total;
    uint64_t nsit;
    /* history: the last `depth` symbols (numbers, NONE for a symbol never met) */
    int depth;
    uint32_t *hist;
    int hlen;
    /* exclusion stamps per symbol number */
    uint32_t *stamp;
    uint64_t stamp_cap;
    uint32_t cur_stamp;
    int update_found;
    int found;
    uint64_t lived;
    /* episode expert */
    int episode;
    int minlen;
    uint32_t *text;            /* every symbol lived, as numbers */
    uint64_t text_len, text_cap;
    uint32_t *ep_table;        /* hash of the last minlen symbols -> position after them + 1 */
    int lines_on;              /* the line episode: the latest line that began as the current line began */
    uint64_t line_h; uint32_t line_len;
    uint32_t nl_num;           /* the number of the line end, NONE until met */
    uint32_t *ln_table; uint64_t ln_mask;
    uint64_t ln_r[8], ln_a[8];
    int ep_n;                  /* several lengths (longest first), 0 for the one length minlen */
    int ep_L[8];
    uint32_t *ep_tabs[8];
    uint64_t ep_mask;
    uint64_t ep_ptr;           /* position of the predicted symbol in text, valid if ep_len > 0 */
    uint32_t ep_len;           /* length of the current match */
    uint64_t *v_right, *v_asked; /* verdict records per situation */
    uint64_t verdict_cells;
    /* the harmony: binary context mixing (core_stream_mix) */
    int mix_on, nin, nsets;
    double *w;                 /* nsets * nin weights of the gate */
    double lr;
    double *dist;              /* the cortex's distribution over symbol numbers */
    uint64_t dist_cap;
    uint32_t *lv_code, *lv_cnt, *lv_len; /* per level: entries filtered by the code prefix decided so far */
    uint64_t lv_cap;
    uint64_t ep_r[8], ep_a[8]; /* the episode's record per match-length bucket */
    uint64_t *ext_n1, *ext_n;  /* numbers 254 and above: order-0 record per bit-prefix node */
    double mix_bits_new, mix_bits_ext;
    /* the word expert: contexts of the current word's beginning, alone and after the previous word */
    int words_on;
    uint64_t *wkey; uint32_t *wval; uint64_t wmask, wused;
    uint64_t cur_word, prev_word;
    /* the final calibration by the rule: (previous symbol's code, bit prefix) x 24 buckets of the gate's belief */
    uint32_t *apm_n1, *apm_n;
    double *w2;                /* the last gate: the two gates and the calibration, per bit position */
    double *wg;                /* a second gate with one weight set per bit position: it learns fast */
    int two_inputs;            /* each origin also gives its belief minus 1/2 */
    int tree_prior;            /* an origin with little record leans on the coarser origin: (W+ + p0) / (W + 1) */
    int recent;                /* short-term memory: contexts of length 1..recent with records decaying by 1 - 1/F */
    float *rm_n1, *rm_n;       /* bounded table: (context, bit prefix) -> decayed record */
    uint32_t *rm_tag;
    uint64_t rm_mask;
    uint64_t *set_updates;     /* updates per weight set, for the learning rate */
    uint32_t prev_code;
    double lr_min;
    /* the word voice (MODEL.md 12): candidates for the next token (its gap and its letters, as lived), given by a
     * word-level mind with weights and with how much that mind knows; one more origin of the harmony */
    int voice_on;
    uint32_t *vc_code;         /* the candidates' forms, concatenated (code points) */
    uint32_t *vc_off, *vc_len;
    double *vc_w;
    uint8_t *vc_live;          /* still consistent with what was lived since the voice was set */
    uint32_t vc_n, vc_ncap;
    uint64_t vc_ccap;
    uint32_t vc_pos;           /* symbols lived since the voice was set */
    double vc_know;            /* how much the voice knows, in [0, 1]: 2 (e_best - 1/2) */
    uint64_t vc_r[12], vc_a[12]; /* its record per (knowing in sixths, first symbol or not): top symbol right, asked */
    uint32_t *vt_code, *vt_cp; double *vt_w; /* per symbol: the consistent candidates' next symbol and weight */
    int i_voice;               /* index of its first input */
} Core;

static uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

static void *grow(void *p, uint64_t newcap, size_t size) {
    void *q = realloc(p, newcap * size);
    if (!q) abort();
    return q;
}

/* ---------------------------------------------------------------------------------------------------------------- */
static int TB[8] = {0, 1, 2, 3, 4, 4, 4, 4};

static int situation(int d, int first, uint32_t t, uint32_t n) {
    int tb = t < 8 ? TB[t] : 5;
    int nb;
    if (n < 5) nb = (int)n;          /* 0..4 (n >= 1 in use) */
    else if (n < 8) nb = 5;
    else if (n < 16) nb = 6;
    else if (n < 32) nb = 7;
    else if (n < 64) nb = 8;
    else nb = 9;
    return (((d << 1) | first) * 6 + tb) * 10 + nb;
}

static double novelty(Core *c, int key) {
    if ((uint64_t)key >= c->nsit || c->sit_total[key] == 0) return 0.5;
    return ((double)c->sit_new[key] + 0.5) / ((double)c->sit_total[key] + 1.0);
}

/* ---------------------------------------------------------------------------------------------------------------- */
static uint32_t new_node(Core *c) {
    if (c->nodes == c->cap_nodes) {
        uint64_t cap = c->cap_nodes * 2;
        c->n = grow(c->n, cap, 4); c->t = grow(c->t, cap, 4);
        c->head = grow(c->head, cap, 4); c->tail = grow(c->tail, cap, 4);
        c->cap_nodes = cap;
    }
    uint32_t v = (uint32_t)c->nodes++;
    c->n[v] = 0; c->t[v] = 0; c->head[v] = 0; c->tail[v] = 0;
    return v;
}

static void add_entry(Core *c, uint32_t v, uint32_t sym) {
    if (c->entries == c->cap_entries) {
        uint64_t cap = c->cap_entries * 2;
        c->esym = grow(c->esym, cap, 4); c->ecnt = grow(c->ecnt, cap, 4); c->enext = grow(c->enext, cap, 4);
        c->cap_entries = cap;
    }
    uint32_t e = (uint32_t)c->entries++;
    c->esym[e] = sym; c->ecnt[e] = 1; c->enext[e] = 0;
    if (c->tail[v]) c->enext[c->tail[v]] = e; else c->head[v] = e;
    c->tail[v] = e;
    c->t[v] += 1;
}

static uint32_t count_of(Core *c, uint32_t v, uint32_t sym) {
    for (uint32_t e = c->head[v]; e; e = c->enext[e])
        if (c->esym[e] == sym) return c->ecnt[e];
    return 0;
}

static void ctable_grow(Core *c) {
    uint64_t oldcap = c->cmask + 1, cap = oldcap * 2;
    uint64_t *ok = c->ckey; uint32_t *ov = c->cval;
    c->ckey = calloc(cap, 8); c->cval = malloc(cap * 4);
    if (!c->ckey || !c->cval) abort();
    c->cmask = cap - 1;
    for (uint64_t i = 0; i < oldcap; i++) if (ok[i]) {
        uint64_t h = mix64(ok[i]) & c->cmask;
        while (c->ckey[h]) h = (h + 1) & c->cmask;
        c->ckey[h] = ok[i]; c->cval[h] = ov[i];
    }
    free(ok); free(ov);
}

static uint32_t child(Core *c, uint32_t parent, uint32_t sym) {
    if (sym == NONE) return NONE;
    uint64_t key = (((uint64_t)parent << 21) | sym) + 1;
    uint64_t h = mix64(key) & c->cmask;
    while (c->ckey[h]) {
        if (c->ckey[h] == key) return c->cval[h];
        h = (h + 1) & c->cmask;
    }
    return NONE;
}

static void set_child(Core *c, uint32_t parent, uint32_t sym, uint32_t v) {
    if ((c->cused + 1) * 5 > (c->cmask + 1) * 3) ctable_grow(c);
    uint64_t key = (((uint64_t)parent << 21) | sym) + 1;
    uint64_t h = mix64(key) & c->cmask;
    while (c->ckey[h]) h = (h + 1) & c->cmask;
    c->ckey[h] = key; c->cval[h] = v; c->cused++;
}

static void atable_grow(Core *c) {
    uint64_t oldcap = c->amask + 1, cap = oldcap * 2;
    uint64_t *ok = c->akey; uint32_t *ov = c->aval;
    c->akey = calloc(cap, 8); c->aval = malloc(cap * 4);
    if (!c->akey || !c->aval) abort();
    c->amask = cap - 1;
    for (uint64_t i = 0; i < oldcap; i++) if (ok[i]) {
        uint64_t h = mix64(ok[i]) & c->amask;
        while (c->akey[h]) h = (h + 1) & c->amask;
        c->akey[h] = ok[i]; c->aval[h] = ov[i];
    }
    free(ok); free(ov);
}

static uint32_t symbol_number(Core *c, uint32_t code) {
    uint64_t key = (uint64_t)code + 1;
    uint64_t h = mix64(key) & c->amask;
    while (c->akey[h]) {
        if (c->akey[h] == key) return c->aval[h];
        h = (h + 1) & c->amask;
    }
    return NONE;
}

static uint32_t add_symbol(Core *c, uint32_t code) {
    if ((c->aused + 1) * 2 > c->amask + 1) atable_grow(c);
    uint64_t key = (uint64_t)code + 1;
    uint64_t h = mix64(key) & c->amask;
    while (c->akey[h]) h = (h + 1) & c->amask;
    uint32_t num = (uint32_t)c->asize;
    c->akey[h] = key; c->aval[h] = num; c->aused++;
    if (code == 10) c->nl_num = num;
    if (c->asize == c->acap) { c->acap *= 2; c->acode = grow(c->acode, c->acap, 4); }
    c->acode[c->asize++] = code;
    if (c->asize > c->stamp_cap) {
        uint64_t cap = c->stamp_cap * 2;
        c->stamp = grow(c->stamp, cap, 4);
        memset(c->stamp + c->stamp_cap, 0, (cap - c->stamp_cap) * 4);
        c->stamp_cap = cap;
    }
    return num;
}

/* ---------------------------------------------------------------------------------------------------------------- */
Core *core_new(int depth, int update_found, int episode, int minlen, int ep_bits) {
    Core *c = calloc(1, sizeof(Core));
    c->depth = depth;
    c->update_found = update_found;
    c->cap_nodes = 1 << 16; c->nodes = 0;
    c->n = malloc(c->cap_nodes * 4); c->t = malloc(c->cap_nodes * 4);
    c->head = malloc(c->cap_nodes * 4); c->tail = malloc(c->cap_nodes * 4);
    c->cap_entries = 1 << 16; c->entries = 1;              /* entry 0 is null */
    c->esym = malloc(c->cap_entries * 4); c->ecnt = malloc(c->cap_entries * 4); c->enext = malloc(c->cap_entries * 4);
    c->cmask = (1 << 16) - 1; c->ckey = calloc(c->cmask + 1, 8); c->cval = malloc((c->cmask + 1) * 4);
    c->amask = (1 << 10) - 1; c->akey = calloc(c->amask + 1, 8); c->aval = malloc((c->amask + 1) * 4);
    c->acap = 1024; c->acode = malloc(c->acap * 4);
    c->stamp_cap = 1024; c->stamp = calloc(c->stamp_cap, 4);
    c->nsit = (uint64_t)(((depth << 1) | 1) * 6 + 5) * 10 + 10;
    c->sit_new = calloc(c->nsit, 8); c->sit_total = calloc(c->nsit, 8);
    c->hist = calloc(depth > 0 ? depth : 1, 4);
    c->nl_num = NONE;
    new_node(c);                                            /* the root, n = 0 until the first symbol */
    c->episode = episode;
    c->minlen = minlen > 0 ? minlen : 12;
    c->text_cap = 1 << 16; c->text = malloc(c->text_cap * 4);
    c->ep_mask = (1ull << (ep_bits > 0 ? ep_bits : 22)) - 1; c->ep_table = calloc(c->ep_mask + 1, 4);
    c->verdict_cells = 8 * 16;
    c->v_right = calloc(c->verdict_cells, 8); c->v_asked = calloc(c->verdict_cells, 8);
    return c;
}

void core_free(Core *c) {
    free(c->n); free(c->t); free(c->head); free(c->tail);
    free(c->esym); free(c->ecnt); free(c->enext);
    free(c->ckey); free(c->cval); free(c->akey); free(c->aval); free(c->acode);
    free(c->stamp); free(c->sit_new); free(c->sit_total); free(c->hist);
    free(c->text); free(c->ep_table); free(c->v_right); free(c->v_asked);
    for (int k = 0; k < c->ep_n; k++) free(c->ep_tabs[k]);
    free(c->w); free(c->dist); free(c->lv_code); free(c->lv_cnt); free(c->lv_len); free(c->ext_n1); free(c->ext_n);
    free(c->wkey); free(c->wval); free(c->apm_n1); free(c->apm_n); free(c->w2); free(c->set_updates); free(c->wg);
    free(c->rm_n1); free(c->rm_n); free(c->rm_tag);
    free(c->vc_code); free(c->vc_off); free(c->vc_len); free(c->vc_w); free(c->vc_live);
    free(c->vt_code); free(c->vt_cp); free(c->vt_w);
    free(c);
}

void core_reset_history(Core *c) { c->hlen = 0; c->ep_len = 0; }

/* continue from a given history (its last `depth` code points; symbols never met cannot be in a context) */
static int is_word_char(uint32_t code);

/* continue from a given history: its last `depth` code points form the contexts (symbols never met cannot be in a
 * context); the whole of it gives the word state and the previous symbol */
void core_set_history(Core *c, const uint32_t *codes, uint64_t n) {
    c->hlen = 0;
    c->ep_len = 0;
    uint64_t from = n > (uint64_t)c->depth ? n - c->depth : 0;
    for (uint64_t i = from; i < n; i++) c->hist[c->hlen++] = symbol_number(c, codes[i]);
    c->cur_word = 0; c->prev_word = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (is_word_char(codes[i])) c->cur_word = mix64(c->cur_word ^ (codes[i] + 0x100000000ull));
        else if (c->cur_word) { c->prev_word = c->cur_word; c->cur_word = 0; }
    }
    uint32_t num = n ? symbol_number(c, codes[n - 1]) : NONE;
    c->prev_code = n ? (num == NONE ? 254u : (num < 254 ? num : 255u)) : 0;
    c->line_h = 0x6C696E65ull; c->line_len = 0;          /* the line so far: what follows the last line end */
    uint64_t start = 0;
    for (uint64_t i = n; i > 0; i--) if (codes[i - 1] == 10) { start = i; break; }
    for (uint64_t i = start; i < n; i++) { c->line_h = mix64(c->line_h ^ codes[i]); c->line_len++; }
}

/* the nodes on the path of the history: root first */
static int chain_of(Core *c, uint32_t *chain) {
    if (c->n[0] == 0) return 0;
    int len = 0;
    chain[len++] = 0;
    int top = c->hlen < c->depth ? c->hlen : c->depth;
    uint32_t v = 0;
    for (int d = 1; d <= top; d++) {
        v = child(c, v, c->hist[c->hlen - d]);
        if (v == NONE) break;
        chain[len++] = v;
    }
    return len;
}

/* probability of symbol number x (NONE: never met) with exclusion; events (situation, new) recorded if ev != 0 */
static double p_excl(Core *c, uint32_t *chain, int len, uint32_t x, int *ev, int *nev) {
    double mass = 1.0;
    int first = 1;
    uint32_t st = ++c->cur_stamp;
    if (st == 0) { memset(c->stamp, 0, c->stamp_cap * 4); st = c->cur_stamp = 1; }
    for (int d = len - 1; d >= 0; d--) {
        uint32_t v = chain[d];
        if (first) {
            uint32_t t = c->t[v], n = c->n[v];
            int key = situation(d, 1, t, n);
            double e = novelty(c, key);
            uint32_t cx = x == NONE ? 0 : count_of(c, v, x);
            if (ev) { ev[2 * *nev] = key; ev[2 * *nev + 1] = cx ? 0 : 1; (*nev)++; }
            if (cx) { c->found = d; return mass * (1.0 - e) * (double)cx / (double)n; }
            mass *= e;
            for (uint32_t q = c->head[v]; q; q = c->enext[q]) c->stamp[c->esym[q]] = st;
            first = 0;
            continue;
        }
        uint64_t nn = 0; uint32_t tt = 0;
        for (uint32_t q = c->head[v]; q; q = c->enext[q])
            if (c->stamp[c->esym[q]] != st) { nn += c->ecnt[q]; tt++; }
        if (tt == 0) continue;
        int key = situation(d, 0, tt, (uint32_t)nn);
        double e = novelty(c, key);
        uint32_t cx = x == NONE ? 0 : count_of(c, v, x);
        if (ev) { ev[2 * *nev] = key; ev[2 * *nev + 1] = cx ? 0 : 1; (*nev)++; }
        if (cx) { c->found = d; return mass * (1.0 - e) * (double)cx / (double)nn; }
        mass *= e;
        for (uint32_t q = c->head[v]; q; q = c->enext[q]) c->stamp[c->esym[q]] = st;
    }
    c->found = 0;
    return mass / (UNICODE_SPACE - (double)c->asize);
}

static void learn(Core *c, uint32_t *chain, int len, uint32_t x) {
    int start = c->update_found ? c->found : 0;
    for (int i = start; i < len; i++) {
        uint32_t v = chain[i];
        c->n[v] += 1;
        uint32_t e;
        for (e = c->head[v]; e; e = c->enext[e]) if (c->esym[e] == x) { c->ecnt[e] += 1; break; }
        if (!e) add_entry(c, v, x);
    }
    int top = c->hlen < c->depth ? c->hlen : c->depth;
    int d = len;
    uint32_t parent;
    if (d == 0) { c->n[0] = 1; add_entry(c, 0, x); d = 1; parent = 0; }
    else parent = chain[len - 1];
    while (d <= top) {
        uint32_t sym = c->hist[c->hlen - d];
        if (sym == NONE) break;                      /* a context holding a symbol never met cannot exist */
        uint32_t v = new_node(c);
        c->n[v] = 1; add_entry(c, v, x);
        set_child(c, parent, sym, v);
        parent = v;
        d++;
    }
    c->lived++;
}

static void push_history(Core *c, uint32_t x) {
    if (c->depth == 0) return;
    if (c->hlen == c->depth) { memmove(c->hist, c->hist + 1, (c->depth - 1) * 4); c->hlen--; }
    c->hist[c->hlen++] = x;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* episode expert */
static uint64_t ep_hash(Core *c, uint64_t end) {   /* hash of text[end - minlen, end) */
    uint64_t h = 0;
    for (uint64_t i = end - c->minlen; i < end; i++) h = mix64(h ^ c->text[i]);
    return h;
}

static uint64_t ep_hash_len(Core *c, uint64_t end, int L) {   /* hash of text[end - L, end), salted by L */
    uint64_t h = (uint64_t)L * 0x9E37ull;
    for (uint64_t i = end - L; i < end; i++) h = mix64(h ^ c->text[i]);
    return h;
}

/* the episode looks for its match at several lengths, the longest first */
void core_set_episode_lengths(Core *c, const int *lens, int n) {
    for (int k = 0; k < c->ep_n; k++) free(c->ep_tabs[k]);
    c->ep_n = n > 8 ? 8 : n;
    for (int k = 0; k < c->ep_n; k++) {
        c->ep_L[k] = lens[k];
        c->ep_tabs[k] = calloc(c->ep_mask + 1, 4);
        if (!c->ep_tabs[k]) abort();
    }
}

static void text_push(Core *c, uint32_t x) {
    if (c->text_len == c->text_cap) { c->text_cap *= 2; c->text = grow(c->text, c->text_cap, 4); }
    c->text[c->text_len++] = x;
}

/* after a symbol was appended to text: update the match and (when learning) index the new position */
static void ep_after(Core *c, int learning) {
    uint64_t L = c->text_len;
    if (c->ep_len > 0) {
        if (c->text[c->ep_ptr] == c->text[L - 1]) { c->ep_ptr++; c->ep_len++; }
        else c->ep_len = 0;
    }
    if (c->ep_n) {
        uint64_t hs[8];
        for (int k = 0; k < c->ep_n; k++) hs[k] = L >= (uint64_t)c->ep_L[k] ? ep_hash_len(c, L, c->ep_L[k]) & c->ep_mask : 0;
        if (c->ep_len == 0) {
            for (int k = 0; k < c->ep_n; k++) {
                if (L < (uint64_t)c->ep_L[k]) continue;
                uint32_t cand = c->ep_tabs[k][hs[k]];
                if (!cand) continue;
                uint64_t p = cand - 1;
                uint32_t len = 0;
                while (len < 400 && len < p && c->text[p - 1 - len] == c->text[L - 1 - len]) len++;
                if (len >= (uint32_t)c->ep_L[k]) { c->ep_ptr = p; c->ep_len = len; break; }
            }
        }
        if (learning)
            for (int k = 0; k < c->ep_n; k++) if (L >= (uint64_t)c->ep_L[k]) c->ep_tabs[k][hs[k]] = (uint32_t)L + 1;
        return;
    }
    if (L >= (uint64_t)c->minlen) {
        uint64_t h = ep_hash(c, L) & c->ep_mask;
        if (c->ep_len == 0) {
            uint32_t cand = c->ep_table[h];
            if (cand) {
                uint64_t p = cand - 1;               /* text[p] followed an earlier occurrence */
                uint32_t len = 0;
                while (len < 400 && len < p && c->text[p - 1 - len] == c->text[L - 1 - len]) len++;
                if (len >= (uint32_t)c->minlen) { c->ep_ptr = p; c->ep_len = len; }
            }
        }
        if (learning) c->ep_table[h] = (uint32_t)L + 1;   /* a newer episode takes the place of an older one */
    }
}

static int len_bucket_abs(uint32_t len) {          /* several lengths: by the length itself */
    if (len < 8) return 0;
    if (len < 12) return 1;
    if (len < 16) return 2;
    if (len < 24) return 3;
    if (len < 32) return 4;
    if (len < 64) return 5;
    if (len < 128) return 6;
    return 7;
}

static int len_bucket(uint32_t len, int minlen) {
    uint32_t k = len - minlen;
    if (k < 4) return 0;
    if (k < 8) return 1;
    if (k < 16) return 2;
    if (k < 32) return 3;
    if (k < 64) return 4;
    if (k < 128) return 5;
    if (k < 256) return 6;
    return 7;
}

static int p_bucket(double p) {                     /* 16 buckets of the cortex's probability, finer near 1 */
    double s = log(p / (1.0 - p + 1e-12) + 1e-12);   /* stretch */
    int b = (int)floor((s + 8.0) / 1.0);
    if (b < 0) b = 0;
    if (b > 15) b = 15;
    return b;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* live a text: predict each symbol, then learn it. codes: Unicode code points. Returns total bits. */
double core_stream(Core *c, const uint32_t *codes, uint64_t n, int learning, double *bits_out) {
    uint32_t *chain = malloc((c->depth + 2) * 4);
    int *ev = malloc((c->depth + 2) * 2 * sizeof(int));
    double total = 0.0;
    uint64_t kept = c->text_len;          /* a pass that does not learn leaves no episode behind */
    for (uint64_t i = 0; i < n; i++) {
        uint32_t x = symbol_number(c, codes[i]);
        int len = chain_of(c, chain);
        int nev = 0;
        double p = p_excl(c, chain, len, x, learning ? ev : 0, &nev);
        int found = c->found;
        if (c->episode && c->ep_len > 0) {
            uint32_t y = c->text[c->ep_ptr];
            double py = p_excl(c, chain, len, y, 0, 0);      /* the cortex's own belief in the episode's symbol */
            int cell = (c->ep_n ? len_bucket_abs(c->ep_len) : len_bucket(c->ep_len, c->minlen)) * 16 + p_bucket(py);
            double q = ((double)c->v_right[cell] + 0.5) / ((double)c->v_asked[cell] + 1.0);
            if (x == y) p = q;
            else p = (1.0 - q) * p / (1.0 - py);
            if (learning) { c->v_asked[cell]++; if (x == y) c->v_right[cell]++; }
        }
        c->found = found;
        if (learning)
            for (int k = 0; k < nev; k++) { c->sit_total[ev[2 * k]]++; c->sit_new[ev[2 * k]] += ev[2 * k + 1]; }
        double b = -log2(p);
        total += b;
        if (bits_out) bits_out[i] = b;
        if (learning) {
            if (x == NONE) x = add_symbol(c, codes[i]);
            learn(c, chain, len, x);
        }
        push_history(c, x);
        if (c->episode) { text_push(c, x); ep_after(c, learning); }
    }
    if (c->episode && !learning) { c->text_len = kept; c->ep_len = 0; }
    free(chain); free(ev);
    return total;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* The harmony (MODEL.md, section 5), as binary context mixing.
 *
 * A symbol is decided in 8 binary steps of its code: its number if below 254; 254 for a symbol never met (then its
 * code point, at the root's price log2(1114112 - |A|)); 255 for numbers 254 and above (then the number, 16 bits, by
 * an order-0 record). At each step every origin gives its belief that the next bit is 1:
 *   - the cortex: its own distribution (the walk with exclusion), summed over the codes with each prefix;
 *   - each context the history has (d = 0 .. D): the rule on its record restricted to the codes with the prefix
 *     decided so far, (n1 + 1/2) / (n + 1); no belief if no code with the prefix was met there;
 *   - the episode: the bit of its predicted symbol, with the rule on its record per match length;
 *   - a constant.
 * The gate adds the beliefs' log-odds with learned weights (a set per bit position, episode state and depth) and
 * learns them online by the gradient of the code length: the one method of a network the harmony borrows.
 */
/* log-odds and their inverse through tables with linear interpolation (deterministic, error below 1e-4) */
#define ST_N 4096
#define SQ_N 4096
#define SQ_R 16.0
static double st_tab[ST_N + 1], sq_tab[SQ_N + 1];
static int tabs_ready = 0;
static void make_tables(void) {
    for (int i = 0; i <= ST_N; i++) {
        double p = (double)i / ST_N;
        if (p < 1e-6) p = 1e-6;
        if (p > 1 - 1e-6) p = 1 - 1e-6;
        st_tab[i] = log(p / (1.0 - p));
    }
    for (int i = 0; i <= SQ_N; i++) {
        double x = -SQ_R + 2.0 * SQ_R * i / SQ_N;
        sq_tab[i] = 1.0 / (1.0 + exp(-x));
    }
    tabs_ready = 1;
}
static inline double stretch(double p) {
    if (p <= 1.0 / ST_N || p >= 1.0 - 1.0 / ST_N) return log(p / (1.0 - p));   /* the tails exactly */
    double f = p * ST_N;
    int i = (int)f;
    double r = f - i;
    return st_tab[i] + r * (st_tab[i + 1] - st_tab[i]);
}
static inline double squash(double x) {
    if (x >= SQ_R) return 1.0 / (1.0 + exp(-x));
    if (x <= -SQ_R) return 1.0 / (1.0 + exp(-x));
    double f = (x + SQ_R) * (SQ_N / (2.0 * SQ_R));
    int i = (int)f;
    if (i >= SQ_N) i = SQ_N - 1;
    double r = f - i;
    return sq_tab[i] + r * (sq_tab[i + 1] - sq_tab[i]);
}

void core_enable_mix(Core *c, double lr, double lr_min, int words, int two_inputs, int tree_prior, int recent,
                     int voice) {
    if (!tabs_ready) make_tables();
    c->mix_on = 1;
    c->voice_on = voice;
    c->tree_prior = tree_prior;
    c->recent = recent;
    if (recent) {
        c->rm_mask = (1u << 22) - 1;
        c->rm_n1 = calloc(c->rm_mask + 1, sizeof(float)); c->rm_n = calloc(c->rm_mask + 1, sizeof(float));
        c->rm_tag = calloc(c->rm_mask + 1, 4);
    }
    c->lr = lr;
    c->lr_min = lr_min;
    c->words_on = words;
    c->two_inputs = two_inputs;
    int norig = (c->depth + 1) + (words ? 2 : 0);
    int nv = voice ? (two_inputs ? 2 : 1) : 0;                              /* the word voice's inputs */
    c->nin = 1 + (norig + 1) * (two_inputs ? 2 : 1) + recent + nv + 1 + 1 + 1;   /* a slot for the line context;
                                                                           ..., line episode, episode, constant */
    c->i_voice = c->nin - 3 - nv;
    c->nsets = 8 * 2 * 3 * (voice ? 3 : 1);                                /* x the voice: absent, below, at e* */
    if (voice) {
        c->vc_ncap = 64; c->vc_ccap = 1024;
        c->vc_code = malloc(c->vc_ccap * 4); c->vc_off = malloc(c->vc_ncap * 4); c->vc_len = malloc(c->vc_ncap * 4);
        c->vc_w = malloc(c->vc_ncap * sizeof(double)); c->vc_live = calloc(c->vc_ncap, 1);
        c->vt_code = malloc(c->vc_ncap * 4); c->vt_cp = malloc(c->vc_ncap * 4);
        c->vt_w = malloc(c->vc_ncap * sizeof(double));
    }
    c->w = calloc((size_t)c->nsets * c->nin, sizeof(double));
    for (int k = 0; k < c->nsets; k++) c->w[(size_t)k * c->nin] = 1.0;     /* trust the cortex first */
    c->set_updates = calloc(c->nsets, 8);
    c->dist_cap = 1024; c->dist = calloc(c->dist_cap, sizeof(double));
    c->lv_cap = 4096;
    c->lv_code = malloc(c->lv_cap * 4); c->lv_cnt = malloc(c->lv_cap * 4);
    c->lv_len = calloc(c->depth + 4, 4);
    c->ext_n1 = calloc(1 << 17, 8); c->ext_n = calloc(1 << 17, 8);
    c->wmask = (1 << 16) - 1; c->wkey = calloc(c->wmask + 1, 8); c->wval = malloc((c->wmask + 1) * 4);
    c->apm_n1 = calloc((size_t)65536 * 24, 4); c->apm_n = calloc((size_t)65536 * 24, 4);
    c->w2 = calloc(8 * 4, sizeof(double));
    for (int k = 0; k < 8; k++) { c->w2[k * 4] = 0.5; c->w2[k * 4 + 1] = 0.5; }
    c->wg = calloc((size_t)8 * c->nin, sizeof(double));
    for (int k = 0; k < 8; k++) c->wg[(size_t)k * c->nin] = 1.0;
}

/* the word voice: n candidates for the next token, each a form (its gap and its letters as lived: lens[i] code
 * points of `codes`, one after another) with a weight; `know` is how much the voice knows. Replaces what it held. */
void core_set_voice(Core *c, const uint32_t *codes, const uint32_t *lens, const double *weights, uint32_t n,
                    double know) {
    if (!c->voice_on) return;
    uint64_t total = 0;
    for (uint32_t i = 0; i < n; i++) total += lens[i];
    if (n > c->vc_ncap) {
        uint32_t cap = c->vc_ncap;
        while (n > cap) cap *= 2;
        c->vc_off = grow(c->vc_off, cap, 4); c->vc_len = grow(c->vc_len, cap, 4);
        c->vc_w = grow(c->vc_w, cap, sizeof(double)); c->vc_live = grow(c->vc_live, cap, 1);
        c->vt_code = grow(c->vt_code, cap, 4); c->vt_cp = grow(c->vt_cp, cap, 4);
        c->vt_w = grow(c->vt_w, cap, sizeof(double));
        c->vc_ncap = cap;
    }
    if (total > c->vc_ccap) {
        uint64_t cap = c->vc_ccap;
        while (total > cap) cap *= 2;
        c->vc_code = grow(c->vc_code, cap, 4); c->vc_ccap = cap;
    }
    uint64_t off = 0;
    for (uint32_t i = 0; i < n; i++) {
        c->vc_off[i] = (uint32_t)off; c->vc_len[i] = lens[i]; c->vc_w[i] = weights[i];
        c->vc_live[i] = lens[i] > 0 && weights[i] > 0;
        for (uint32_t j = 0; j < lens[i]; j++) c->vc_code[off + j] = codes[off + j];
        off += lens[i];
    }
    c->vc_n = n;
    c->vc_pos = 0;
    c->vc_know = know < 0 ? 0 : (know > 1 ? 1 : know);
}

/* the voice's state: out[0] candidates still consistent and unfinished, out[1] consistent and finished (a whole form
 * was lived), out[2] symbols lived since it was set */
void core_voice_info(Core *c, uint32_t *out) {
    out[0] = out[1] = 0;
    out[2] = c->vc_pos;
    if (!c->voice_on) return;
    for (uint32_t i = 0; i < c->vc_n; i++) if (c->vc_live[i]) {
        if (c->vc_pos < c->vc_len[i]) out[0]++;
        else if (c->vc_pos == c->vc_len[i]) out[1]++;
    }
}

void core_enable_lines(Core *c, int bits) {
    c->lines_on = 1;
    c->ln_mask = (1ull << (bits > 0 ? bits : 22)) - 1;
    c->ln_table = calloc(c->ln_mask + 1, 4);
    c->line_h = 0x6C696E65ull; c->line_len = 0;
    c->nl_num = symbol_number(c, 10);
}

static int line_bucket(uint32_t n) {
    if (n < 4) return 0;
    if (n < 8) return 1;
    if (n < 16) return 2;
    if (n < 32) return 3;
    if (n < 64) return 4;
    return 5;
}

/* word contexts: a separate table from 64-bit keys (hashes of exact strings) to nodes */
static void wtable_grow(Core *c) {
    uint64_t oldcap = c->wmask + 1, cap = oldcap * 2;
    uint64_t *ok = c->wkey; uint32_t *ov = c->wval;
    c->wkey = calloc(cap, 8); c->wval = malloc(cap * 4);
    if (!c->wkey || !c->wval) abort();
    c->wmask = cap - 1;
    for (uint64_t i = 0; i < oldcap; i++) if (ok[i]) {
        uint64_t h = mix64(ok[i]) & c->wmask;
        while (c->wkey[h]) h = (h + 1) & c->wmask;
        c->wkey[h] = ok[i]; c->wval[h] = ov[i];
    }
    free(ok); free(ov);
}

static uint32_t word_node(Core *c, uint64_t key, int create) {
    if (key == 0) key = 1;
    uint64_t h = mix64(key) & c->wmask;
    while (c->wkey[h]) {
        if (c->wkey[h] == key) return c->wval[h];
        h = (h + 1) & c->wmask;
    }
    if (!create) return NONE;
    if ((c->wused + 1) * 5 > (c->wmask + 1) * 3) { wtable_grow(c); return word_node(c, key, 1); }
    uint32_t v = new_node(c);
    c->wkey[h] = key; c->wval[h] = v; c->wused++;
    return v;
}

static int is_word_char(uint32_t code) {
    if (code < 128) return (code >= '0' && code <= '9') || (code >= 'A' && code <= 'Z') || (code >= 'a' && code <= 'z');
    return code >= 0xC0 && code != 0xD7 && code != 0xF7;
}

static inline uint32_t code8(uint32_t num) { return num == NONE ? 254u : (num < 254 ? num : 255u); }

/* the cortex's full distribution with exclusion (dist[number]); returns the mass of symbols never met */
static double cortex_dist(Core *c, uint32_t *chain, int len) {
    if (c->asize + 1 > c->dist_cap) {
        uint64_t cap = c->dist_cap;
        while (c->asize + 1 > cap) cap *= 2;
        c->dist = grow(c->dist, cap, sizeof(double)); c->dist_cap = cap;
    }
    memset(c->dist, 0, c->asize * sizeof(double));
    double mass = 1.0;
    int first = 1;
    uint32_t st = ++c->cur_stamp;
    if (st == 0) { memset(c->stamp, 0, c->stamp_cap * 4); st = c->cur_stamp = 1; }
    for (int d = len - 1; d >= 0; d--) {
        uint32_t v = chain[d];
        uint64_t nn = 0; uint32_t tt = 0;
        for (uint32_t q = c->head[v]; q; q = c->enext[q])
            if (c->stamp[c->esym[q]] != st) { nn += c->ecnt[q]; tt++; }
        if (tt == 0) continue;
        double e = novelty(c, situation(d, first, tt, (uint32_t)nn));
        first = 0;
        for (uint32_t q = c->head[v]; q; q = c->enext[q])
            if (c->stamp[c->esym[q]] != st) c->dist[c->esym[q]] = mass * (1.0 - e) * (double)c->ecnt[q] / (double)nn;
        mass *= e;
        for (uint32_t q = c->head[v]; q; q = c->enext[q]) c->stamp[c->esym[q]] = st;
    }
    return mass;
}

/* gather a node's entries as (code, count) at offset off; returns how many */
static uint32_t gather(Core *c, uint32_t v, uint64_t off) {
    uint32_t m = 0;
    for (uint32_t q = c->head[v]; q; q = c->enext[q]) {
        c->lv_code[off + m] = code8(c->esym[q]); c->lv_cnt[off + m] = c->ecnt[q]; m++;
    }
    return m;
}

/* one symbol through the harmony. Scoring (say = 0): the bits of symbol number x (NONE if never met, with its code
 * point), learning it if asked. Saying (say = 1): each bit is chosen as the more probable one; *code_out gets the
 * code, *prob_out the probability of the whole path, and the symbol is lived without learning. */
static double mix_one(Core *c, uint32_t x, uint32_t code_point, int learning, int say, uint32_t *code_out,
                      double *prob_out, uint32_t *chain, uint32_t *orig, int *ev, double *s, uint64_t *lv_off) {
    int nlev = c->depth + 1 + (c->words_on ? 2 : 0) + (c->lines_on ? 1 : 0);
    double tree[512];
    double path = 1.0;
        int len = chain_of(c, chain);
        int nev = 0;
        p_excl(c, chain, len, x, learning ? ev : 0, &nev);      /* the cortex's walk: its situation events */
        int found = c->found;
        double unmet = len ? cortex_dist(c, chain, len) : 1.0;
        /* the cortex's mass per code in a tree of sums: leaves 256..511, node (1 << L) | prefix */
        for (int k = 0; k < 256; k++) tree[256 + k] = 0.0;
        for (uint32_t num = 0; num < c->asize; num++) tree[256 + code8(num)] += c->dist[num];
        tree[256 + 254] += unmet;
        for (int k = 255; k >= 1; k--) tree[k] = tree[2 * k] + tree[2 * k + 1];
        /* origins: every context of the history, then the word contexts */
        for (int d = 0; d <= c->depth; d++) orig[d] = d < len ? chain[d] : NONE;
        uint64_t wk1 = 0, wk2 = 0;
        if (c->words_on) {
            wk1 = mix64(0x5751 ^ c->cur_word);
            wk2 = mix64(0x5752 ^ mix64(c->prev_word) ^ c->cur_word);
            orig[c->depth + 1] = word_node(c, wk1, 0);
            orig[c->depth + 2] = word_node(c, wk2, 0);
        }
        uint64_t lk = 0;
        if (c->lines_on) {                           /* the line context: the exact beginning of the current line */
            lk = mix64(0x4C494E45ull ^ c->line_h ^ ((uint64_t)c->line_len << 40));
            orig[nlev - 1] = word_node(c, lk, 0);
        }
        uint64_t tot = 0;
        for (int o = 0; o < nlev; o++) if (orig[o] != NONE) tot += c->t[orig[o]];
        if (tot > c->lv_cap) {
            while (tot > c->lv_cap) c->lv_cap *= 2;
            c->lv_code = grow(c->lv_code, c->lv_cap, 4); c->lv_cnt = grow(c->lv_cnt, c->lv_cap, 4);
        }
        uint64_t off = 0;
        for (int o = 0; o < nlev; o++) {
            lv_off[o] = off;
            c->lv_len[o] = orig[o] != NONE ? gather(c, orig[o], off) : 0;
            off += c->lv_len[o];
        }
        /* the episode */
        int ep_active = c->episode && c->ep_len > 0;
        uint32_t ycode = 0; double eq = 0.5; int eb = 0;
        if (ep_active) {
            ycode = code8(c->text[c->ep_ptr]);
            eb = c->ep_n ? len_bucket_abs(c->ep_len) : len_bucket(c->ep_len, c->minlen);
            eq = ((double)c->ep_r[eb] + 0.5) / ((double)c->ep_a[eb] + 1.0);
        }
        /* the line episode */
        int ln_active = 0; uint32_t lcode = 0; double lq = 0.5; int lb = 0; uint64_t ln_slot = 0;
        if (c->lines_on) {
            ln_slot = mix64(c->line_h ^ c->line_len) & c->ln_mask;
            uint32_t cand = c->ln_table[ln_slot];
            int ok = 0;
            uint64_t p = cand ? cand - 1 : 0;
            if (cand && p < c->text_len && p >= c->line_len) {
                /* the same beginning: the symbols before p are the current line, and a line end (or the start) before them */
                ok = p == c->line_len || c->text[p - c->line_len - 1] == c->nl_num;
                uint32_t m = c->line_len < 64 ? c->line_len : 64;
                for (uint32_t q = 1; ok && q <= m; q++)
                    if (c->text[p - q] != c->text[c->text_len - q]) ok = 0;
            }
            if (ok) {
                ln_active = 1;
                lcode = code8(c->text[p]);
                lb = line_bucket(c->line_len);
                lq = ((double)c->ln_r[lb] + 0.5) / ((double)c->ln_a[lb] + 1.0);
                if (lq < 1e-6) lq = 1e-6;
                if (lq > 1 - 1e-6) lq = 1 - 1e-6;
            }
        }
        /* the word voice: the next symbol of every candidate still consistent, and the voice's trust by its record */
        uint32_t vn = 0; int vcell = 0, vstate = 0; double vq = 0.5; uint32_t vtop = NONE;
        if (c->voice_on && c->vc_n) {
            for (uint32_t i = 0; i < c->vc_n; i++) {
                if (!c->vc_live[i] || c->vc_pos >= c->vc_len[i]) continue;
                uint32_t cp = c->vc_code[c->vc_off[i] + c->vc_pos];
                c->vt_code[vn] = code8(symbol_number(c, cp)); c->vt_cp[vn] = cp; c->vt_w[vn] = c->vc_w[i]; vn++;
            }
            if (vn) {
                double best = -1.0;                        /* the top symbol: the largest weight in sum */
                for (uint32_t j = 0; j < vn; j++) {
                    double sum = 0.0;
                    for (uint32_t q = 0; q < vn; q++) if (c->vt_cp[q] == c->vt_cp[j]) sum += c->vt_w[q];
                    if (sum > best) { best = sum; vtop = c->vt_cp[j]; }
                }
                int kb = (int)(c->vc_know * 6.0);
                if (kb > 5) kb = 5;
                vcell = kb * 2 + (c->vc_pos > 0 ? 1 : 0);
                vq = ((double)c->vc_r[vcell] + 0.5) / ((double)c->vc_a[vcell] + 1.0);
                if (vq < 1e-6) vq = 1e-6;
                if (vq > 1 - 1e-6) vq = 1 - 1e-6;
                vstate = kb == 0 ? 1 : 2;                  /* below the speaking line, or at it: 2 (e - 1/2) >= 1/6 */
            }
        }
        uint32_t xcode = code8(x);
        int dbucket = len <= 3 ? 0 : (len <= 6 ? 1 : 2);
        double b = 0.0;
        uint32_t node = 1;                                 /* bit-prefix node of the decision */
        uint64_t rh[8] = {0};                              /* short-term contexts of length 1..recent */
        for (int r = 1; r <= c->recent && r <= 7; r++) {
            uint64_t h = 0x7265636Eull + r;
            for (int q = 1; q <= r; q++) h = mix64(h ^ (q <= c->hlen ? c->hist[c->hlen - q] : 0xFFFFFFFEu));
            rh[r] = h;
        }
        int nrec = c->recent < 7 ? c->recent : 7;
        for (int k = 7; k >= 0; k--) {
            uint32_t P = node - (1u << (7 - k));               /* the prefix decided so far */
            int yk = say ? -1 : (int)((xcode >> k) & 1);
            memset(s, 0, c->nin * sizeof(double));             /* origins absent here give nothing */
            double sp = tree[node], sp1 = tree[2 * node + 1];
            double pc = sp > 0 ? sp1 / sp : 0.5;
            if (pc < 1e-6) pc = 1e-6;
            if (pc > 1 - 1e-6) pc = 1 - 1e-6;
            s[0] = stretch(pc);
            double prior = 0.5, root_e = 0.5, word_prior = 0.5;
            for (int o = 0; o < nlev; o++) {
                uint64_t ofs = lv_off[o];
                uint64_t np = 0, np1 = 0;
                uint32_t m = c->lv_len[o];
                for (uint32_t j = 0; j < m; j++) {        /* entries already filtered to the prefix */
                    np += c->lv_cnt[ofs + j];
                    if ((c->lv_code[ofs + j] >> k) & 1) np1 += c->lv_cnt[ofs + j];
                }
                double p0 = 0.5;
                if (c->tree_prior) {
                    if (o <= c->depth) p0 = prior;                       /* the next shorter context */
                    else if (c->lines_on && o == nlev - 1) p0 = root_e;  /* the line's beginning */
                    else p0 = (o == c->depth + 1) ? root_e : word_prior; /* word beginning, then with the word before */
                }
                int has = np > 0 || (c->tree_prior && o <= c->depth && o < len);
                double e = ((double)np1 + p0) / ((double)np + 1.0);
                if (e < 1e-6) e = 1e-6;                   /* keep the log-odds finite */
                if (e > 1 - 1e-6) e = 1 - 1e-6;
                if (c->tree_prior) {
                    if (o <= c->depth && o < len) prior = e;
                    if (o == 0) root_e = e;
                    if (o == c->depth + 1) word_prior = np ? e : root_e;
                }
                if (c->two_inputs) {
                    s[1 + 2 * o] = has ? stretch(e) : 0.0;
                    s[2 + 2 * o] = has ? 2.0 * e - 1.0 : 0.0;
                } else s[1 + o] = has ? stretch(e) : 0.0;
            }
            uint64_t rslot[8];
            for (int r = 1; r <= nrec; r++) {
                uint64_t h = mix64(rh[r] ^ node);
                uint64_t slot = h & c->rm_mask;
                uint32_t tag = (uint32_t)(h >> 40) | 1u;
                int mine = c->rm_tag[slot] == tag;
                if (!mine && learning) { c->rm_tag[slot] = tag; c->rm_n1[slot] = 0; c->rm_n[slot] = 0; mine = 1; }
                rslot[r] = slot;
                double e = mine ? (c->rm_n1[slot] + 0.5) / (c->rm_n[slot] + 1.0) : 0.5;
                s[1 + (nlev) * (c->two_inputs ? 2 : 1) + (r - 1)] = (mine && c->rm_n[slot] > 0) ? stretch(e) : 0.0;
            }
            int ep_here = ep_active && (ycode >> (k + 1)) == P;
            if (eq < 1e-6) eq = 1e-6;
            if (eq > 1 - 1e-6) eq = 1 - 1e-6;
            s[c->nin - 2] = ep_here ? (((ycode >> k) & 1) ? stretch(eq) : -stretch(eq)) : 0.0;
            s[c->nin - 1] = 1.0;
            int ln_here = ln_active && (lcode >> (k + 1)) == P;
            s[c->nin - 3] = ln_here ? (((lcode >> k) & 1) ? stretch(lq) : -stretch(lq)) : 0.0;
            int v_here = 0;
            if (vn) {                                    /* the voice: its candidates' share, trusted by its record */
                double W = 0.0, W1 = 0.0;
                for (uint32_t j = 0; j < vn; j++) if ((c->vt_code[j] >> (k + 1)) == P) {
                    W += c->vt_w[j];
                    if ((c->vt_code[j] >> k) & 1) W1 += c->vt_w[j];
                }
                if (W > 0) {
                    double p1 = vq * (W1 / W) + (1.0 - vq) * 0.5;
                    if (p1 < 1e-6) p1 = 1e-6;
                    if (p1 > 1 - 1e-6) p1 = 1 - 1e-6;
                    s[c->i_voice] = stretch(p1);
                    if (c->two_inputs) s[c->i_voice + 1] = 2.0 * p1 - 1.0;
                    v_here = vstate;
                }
            }
            int set = k + 8 * (ep_here ? 1 : 0) + 16 * dbucket + 48 * v_here;
            double *w = c->w + (size_t)set * c->nin;
            double *wg = c->wg + (size_t)k * c->nin;
            double dot = 0.0, dotg = 0.0;
            for (int j = 0; j < c->nin; j++) { dot += w[j] * s[j]; dotg += wg[j] * s[j]; }
            double pm = squash(dot), pg = squash(dotg);
            if (!(pm == pm)) pm = 0.5;                        /* never index with a number that is not one */
            if (!(pg == pg)) pg = 0.5;
            if (pm < 1e-6) pm = 1e-6;
            if (pm > 1 - 1e-6) pm = 1 - 1e-6;
            if (pg < 1e-6) pg = 1e-6;
            if (pg > 1 - 1e-6) pg = 1 - 1e-6;
            /* the calibration by the rule: (previous code, prefix node) x bucket of the gate's log-odds */
            double st = stretch(pm);
            double pos = (st + 12.0) * (23.0 / 24.0);
            if (pos < 0) pos = 0;
            if (pos > 23) pos = 23;
            int lo = (int)pos; if (lo > 22) lo = 22;
            double fr = pos - lo;
            size_t base = ((size_t)c->prev_code * 256 + node) * 24;
            double e_lo = ((double)c->apm_n1[base + lo] + 0.5) / ((double)c->apm_n[base + lo] + 1.0);
            double e_hi = ((double)c->apm_n1[base + lo + 1] + 0.5) / ((double)c->apm_n[base + lo + 1] + 1.0);
            double pa = (1.0 - fr) * e_lo + fr * e_hi;
            if (pa < 1e-6) pa = 1e-6;
            if (pa > 1 - 1e-6) pa = 1 - 1e-6;
            double s2a = st, s2b = stretch(pa), s2c = stretch(pg);
            double *w2 = c->w2 + k * 4;
            double p = squash(w2[0] * s2a + w2[1] * s2b + w2[2] * s2c + w2[3]);
            if (p < 1.0 / 4096) p = 1.0 / 4096;
            if (p > 1 - 1.0 / 4096) p = 1 - 1.0 / 4096;
            if (say) { yk = p >= 0.5 ? 1 : 0; path *= yk ? p : 1.0 - p; }
            b += -log2(yk ? p : 1.0 - p);
            if (learning) {
                uint64_t u = ++c->set_updates[set];
                double lr = c->lr / sqrt(1.0 + (double)u / 1024.0);
                if (lr < c->lr_min) lr = c->lr_min;
                double err = (double)yk - pm;
                for (int j = 0; j < c->nin; j++) w[j] += lr * err * s[j];
                double errg = (double)yk - pg;
                double lrg = c->lr / sqrt(1.0 + (double)u / 128.0);
                if (lrg < c->lr_min) lrg = c->lr_min;
                for (int j = 0; j < c->nin; j++) wg[j] += lrg * errg * s[j];
                double err2 = (double)yk - p;
                w2[0] += c->lr_min * err2 * s2a; w2[1] += c->lr_min * err2 * s2b;
                w2[2] += c->lr_min * err2 * s2c; w2[3] += c->lr_min * err2;
                int near = fr < 0.5 ? lo : lo + 1;
                c->apm_n[base + near]++; c->apm_n1[base + near] += yk;
                for (int r = 1; r <= nrec; r++) {               /* recent evidence: decay by 1 - 1/F, F = 7 */
                    uint64_t slot = rslot[r];
                    c->rm_n1[slot] = c->rm_n1[slot] * (float)(1.0 - 1.0 / 7.0) + (float)yk;
                    c->rm_n[slot] = c->rm_n[slot] * (float)(1.0 - 1.0 / 7.0) + 1.0f;
                }
            }
            /* keep only the entries with the decided bit */
            for (int o = 0; o < nlev; o++) {
                uint64_t ofs = lv_off[o];
                uint32_t m = c->lv_len[o], keep = 0;
                for (uint32_t j = 0; j < m; j++)
                    if (((c->lv_code[ofs + j] >> k) & 1) == (uint32_t)yk) {
                        c->lv_code[ofs + keep] = c->lv_code[ofs + j]; c->lv_cnt[ofs + keep] = c->lv_cnt[ofs + j]; keep++;
                    }
                c->lv_len[o] = keep;
            }
            node = 2 * node + yk;
        }
        if (!say && xcode == 254) {
            double cost = log2(UNICODE_SPACE - (double)c->asize);
            b += cost; c->mix_bits_new += cost;
        } else if (!say && xcode == 255) {
            uint32_t r = x - 254, nd = 1;
            double cost = 0.0;
            for (int k = 15; k >= 0; k--) {
                int bit = (r >> k) & 1;
                double e = ((double)c->ext_n1[nd] + 0.5) / ((double)c->ext_n[nd] + 1.0);
                cost += -log2(bit ? e : 1.0 - e);
                if (learning) { c->ext_n[nd]++; c->ext_n1[nd] += bit; }
                nd = nd * 2 + bit;
                if (nd >= (1u << 17)) nd = 1;
            }
            b += cost; c->mix_bits_ext += cost;
        }
        c->found = found;
        if (learning) {
            for (int k = 0; k < nev; k++) { c->sit_total[ev[2 * k]]++; c->sit_new[ev[2 * k]] += ev[2 * k + 1]; }
            if (ep_active) { c->ep_a[eb]++; if (x != NONE && x == c->text[c->ep_ptr]) c->ep_r[eb]++; }
            if (ln_active) { c->ln_a[lb]++; if (x != NONE && code8(x) == lcode) c->ln_r[lb]++; }
            if (vn) { c->vc_a[vcell]++; if (code_point == vtop) c->vc_r[vcell]++; }
            if (c->lines_on && c->episode) c->ln_table[ln_slot] = (uint32_t)c->text_len + 1;   /* where x will be */
            if (x == NONE) x = add_symbol(c, code_point);
            learn(c, chain, len, x);
            if (c->lines_on) {
                uint32_t v = word_node(c, lk, 1);
                c->n[v] += 1;
                uint32_t e;
                for (e = c->head[v]; e; e = c->enext[e]) if (c->esym[e] == x) { c->ecnt[e] += 1; break; }
                if (!e) add_entry(c, v, x);
            }
            if (c->words_on) {
                uint64_t keys[2] = {wk1, wk2};
                for (int j = 0; j < 2; j++) {
                    uint32_t v = word_node(c, keys[j], 1);
                    c->n[v] += 1;
                    uint32_t e;
                    for (e = c->head[v]; e; e = c->enext[e]) if (c->esym[e] == x) { c->ecnt[e] += 1; break; }
                    if (!e) add_entry(c, v, x);
                }
            }
        }
        if (say) {                                   /* the symbol the mind would say */
            *code_out = node - 256;
            *prob_out = path;
            if (*code_out >= 254) return b;          /* never says a symbol it has not met, nor an escape */
            x = *code_out;
            code_point = c->acode[x];
        }
        if (c->voice_on && c->vc_n) {                /* the voice follows what was lived */
            for (uint32_t i = 0; i < c->vc_n; i++)
                if (c->vc_live[i] && (c->vc_pos >= c->vc_len[i] || c->vc_code[c->vc_off[i] + c->vc_pos] != code_point))
                    c->vc_live[i] = 0;
            c->vc_pos++;
        }
        if (c->words_on) {
            if (is_word_char(code_point)) c->cur_word = mix64(c->cur_word ^ (code_point + 0x100000000ull));
            else if (c->cur_word) { c->prev_word = c->cur_word; c->cur_word = 0; }
        }
        c->prev_code = code8(x);
        if (c->lines_on) {
            if (code_point == 10) { c->line_h = 0x6C696E65ull; c->line_len = 0; }
            else { c->line_h = mix64(c->line_h ^ code_point); c->line_len++; }
        }
        push_history(c, x);
        if (c->episode) { text_push(c, x); ep_after(c, learning); }
    return b;
}

double core_stream_mix(Core *c, const uint32_t *codes, uint64_t n, int learning, double *bits_out) {
    int nlev = c->depth + 1 + (c->words_on ? 2 : 0) + (c->lines_on ? 1 : 0);   /* origins with records */
    uint32_t *chain = malloc((c->depth + 2) * 4);
    uint32_t *orig = malloc((nlev + 1) * 4);           /* node of each origin, NONE if absent */
    int *ev = malloc((c->depth + 2) * 2 * sizeof(int));
    double *s = malloc(c->nin * sizeof(double));
    uint64_t *lv_off = malloc((nlev + 1) * 8);
    double total = 0.0;
    uint64_t kept = c->text_len;
    uint32_t code; double prob;
    for (uint64_t i = 0; i < n; i++) {
        double b = mix_one(c, symbol_number(c, codes[i]), codes[i], learning, 0, &code, &prob, chain, orig, ev, s,
                           lv_off);
        total += b;
        if (bits_out) bits_out[i] = b;
    }
    if (c->episode && !learning) { c->text_len = kept; c->ep_len = 0; }
    free(chain); free(orig); free(ev); free(s); free(lv_off);
    return total;
}

/* continue a history: say the most probable symbol while its probability reaches `line` (0: always), at most `most`
 * symbols; nothing is learned and the mind is left as it was. Returns how many symbols were said. */
uint64_t core_say(Core *c, const uint32_t *hist, uint64_t nh, uint64_t most, double line, uint32_t *out, double *probs) {
    int nlev = c->depth + 1 + (c->words_on ? 2 : 0) + (c->lines_on ? 1 : 0);
    uint32_t *chain = malloc((c->depth + 2) * 4);
    uint32_t *orig = malloc((nlev + 1) * 4);
    int *ev = malloc((c->depth + 2) * 2 * sizeof(int));
    double *s = malloc(c->nin * sizeof(double));
    uint64_t *lv_off = malloc((nlev + 1) * 8);
    uint32_t *save_hist = malloc((c->depth + 1) * 4);
    memcpy(save_hist, c->hist, c->hlen * 4);
    int save_hlen = c->hlen;
    uint64_t save_cw = c->cur_word, save_pw = c->prev_word, save_text = c->text_len, save_ptr = c->ep_ptr;
    uint32_t save_pc = c->prev_code, save_len = c->ep_len;
    uint64_t save_lh = c->line_h; uint32_t save_ll = c->line_len;
    core_set_history(c, hist, nh);
    if (c->episode) {                                    /* the history is the recent episode */
        uint64_t from = nh > 64 ? nh - 64 : 0;
        for (uint64_t i = from; i < nh; i++) { uint32_t num = symbol_number(c, hist[i]); text_push(c, num == NONE ? 0xFFFFFFFEu : num); }
        if (nh) ep_after(c, 0);
    }
    uint64_t said = 0;
    for (uint64_t i = 0; i < most; i++) {
        uint32_t code; double prob;
        mix_one(c, NONE, 0, 0, 1, &code, &prob, chain, orig, ev, s, lv_off);
        if (code >= 254 || (line > 0 && prob < line)) { if (probs) probs[said] = prob; break; }
        out[said] = c->acode[code];
        if (probs) probs[said] = prob;
        said++;
    }
    memcpy(c->hist, save_hist, save_hlen * 4); c->hlen = save_hlen;
    c->cur_word = save_cw; c->prev_word = save_pw; c->prev_code = save_pc;
    c->text_len = save_text; c->ep_ptr = save_ptr; c->ep_len = save_len;
    c->line_h = save_lh; c->line_len = save_ll;
    free(chain); free(orig); free(ev); free(s); free(lv_off); free(save_hist);
    return said;
}

/* recall (MODEL.md 6, the lock; 8, completion): the goal is to recall what began as the current line began. If a
 * lived line began so (the line episode, verified), the mind locks onto it and says it to its end; otherwise it goes
 * on as core_say does. Returns how many symbols were said; *locked tells which way it went. Nothing is learned. */
uint64_t core_recall(Core *c, const uint32_t *hist, uint64_t nh, uint64_t most, double line, uint32_t *out,
                     int *locked) {
    *locked = 0;
    if (c->lines_on && c->episode && nh) {
        uint64_t save_lh = c->line_h; uint32_t save_ll = c->line_len, save_pc = c->prev_code;
        uint64_t save_cw = c->cur_word, save_pw = c->prev_word;
        uint32_t *save_hist = malloc((c->depth + 1) * 4);
        memcpy(save_hist, c->hist, c->hlen * 4);
        int save_hlen = c->hlen;
        uint64_t save_text = c->text_len;
        core_set_history(c, hist, nh);
        uint64_t from = nh > c->line_len ? nh - c->line_len : 0;   /* the current line, as the episode's tail */
        for (uint64_t i = from; i < nh; i++) { uint32_t num = symbol_number(c, hist[i]); text_push(c, num == NONE ? 0xFFFFFFFEu : num); }
        uint64_t slot = mix64(c->line_h ^ c->line_len) & c->ln_mask;
        uint32_t cand = c->ln_table[slot];
        uint64_t p = cand ? cand - 1 : 0;
        int ok = 0;
        if (cand && p < save_text && p >= c->line_len) {
            ok = p == c->line_len || c->text[p - c->line_len - 1] == c->nl_num;
            uint32_t m = c->line_len < 64 ? c->line_len : 64;
            for (uint32_t q = 1; ok && q <= m; q++)
                if (c->text[p - q] != c->text[c->text_len - q]) ok = 0;
        }
        uint64_t said = 0;
        if (ok) {
            *locked = 1;
            while (said < most && p < save_text && c->text[p] != c->nl_num && c->text[p] < c->asize)
                out[said++] = c->acode[c->text[p++]];
        }
        c->text_len = save_text;
        memcpy(c->hist, save_hist, save_hlen * 4); c->hlen = save_hlen;
        c->line_h = save_lh; c->line_len = save_ll; c->prev_code = save_pc;
        c->cur_word = save_cw; c->prev_word = save_pw;
        free(save_hist);
        if (ok) return said;
    }
    return core_say(c, hist, nh, most, line, out, 0);
}

void core_stats(Core *c, uint64_t *out) {
    out[0] = c->nodes - (c->n[0] == 0 ? 1 : 0);
    out[1] = c->entries - 1;
    out[2] = c->asize;
    out[3] = c->lived;
    uint64_t s = 0;
    for (uint64_t k = 0; k < c->nsit; k++) if (c->sit_total[k]) s++;
    out[4] = s;
    out[5] = c->text_len;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* Keeping a core: every count, record, weight and the history, written to a file and read back into a new core that
 * goes on exactly as the one saved would have (tested: the bits after a load are the bits without one). What is
 * scratch (the distribution, the filtered entries, the exclusion stamps) and the voice's current candidates (set
 * again before every token) are not kept. */
#include <stdio.h>

typedef struct { FILE *f; int writing; int ok; } Ser;

static void ser_bytes(Ser *s, void *p, size_t n) {
    if (!s->ok || n == 0) return;
    if (s->writing) { if (fwrite(p, 1, n, s->f) != n) s->ok = 0; }
    else { if (fread(p, 1, n, s->f) != n) s->ok = 0; }
}
#define SV(s, x) ser_bytes((s), &(x), sizeof(x))

/* an array of `cap` elements of which the first `count` are kept; allocated (zeroed) when reading */
static void ser_arr(Ser *s, void **pp, uint64_t count, uint64_t cap, size_t size) {
    if (!s->ok) return;
    if (!s->writing) {
        free(*pp);
        *pp = calloc(cap ? cap : 1, size);
        if (!*pp) { s->ok = 0; return; }
    }
    ser_bytes(s, *pp, count * size);
}
#define SA(s, p, count, cap) ser_arr((s), (void **)&(p), (count), (cap), sizeof(*(p)))
#define SZ(s, p, cap) do { if (!(s)->writing) { free(p); (p) = calloc((cap) ? (cap) : 1, sizeof(*(p))); \
                           if (!(p)) (s)->ok = 0; } } while (0)

static void ser_core(Ser *s, Core *c) {
    uint64_t magic = 0x31455243564C4ull, version = 1;          /* "LVCRE1" */
    uint64_t m = magic, v = version;
    SV(s, m); SV(s, v);
    if (m != magic || v != version) { s->ok = 0; return; }
    SV(s, c->nodes); SV(s, c->cap_nodes); SV(s, c->entries); SV(s, c->cap_entries);
    SV(s, c->cmask); SV(s, c->cused); SV(s, c->amask); SV(s, c->aused); SV(s, c->asize); SV(s, c->acap);
    SV(s, c->nsit); SV(s, c->depth); SV(s, c->hlen); SV(s, c->stamp_cap);
    SV(s, c->update_found); SV(s, c->found); SV(s, c->lived);
    SV(s, c->episode); SV(s, c->minlen); SV(s, c->text_len); SV(s, c->text_cap); SV(s, c->ep_mask);
    SV(s, c->lines_on); SV(s, c->line_h); SV(s, c->line_len); SV(s, c->nl_num); SV(s, c->ln_mask);
    SV(s, c->ln_r); SV(s, c->ln_a); SV(s, c->ep_n); SV(s, c->ep_L); SV(s, c->ep_ptr); SV(s, c->ep_len);
    SV(s, c->verdict_cells);
    SV(s, c->mix_on); SV(s, c->nin); SV(s, c->nsets); SV(s, c->lr); SV(s, c->lr_min);
    SV(s, c->dist_cap); SV(s, c->lv_cap); SV(s, c->ep_r); SV(s, c->ep_a); SV(s, c->mix_bits_new);
    SV(s, c->mix_bits_ext); SV(s, c->words_on); SV(s, c->wmask); SV(s, c->wused); SV(s, c->cur_word);
    SV(s, c->prev_word); SV(s, c->two_inputs); SV(s, c->tree_prior); SV(s, c->recent); SV(s, c->rm_mask);
    SV(s, c->prev_code); SV(s, c->voice_on); SV(s, c->vc_ncap); SV(s, c->vc_ccap); SV(s, c->vc_r); SV(s, c->vc_a);
    SV(s, c->i_voice);
    if (!s->ok) return;
    SA(s, c->n, c->nodes, c->cap_nodes); SA(s, c->t, c->nodes, c->cap_nodes);
    SA(s, c->head, c->nodes, c->cap_nodes); SA(s, c->tail, c->nodes, c->cap_nodes);
    SA(s, c->esym, c->entries, c->cap_entries); SA(s, c->ecnt, c->entries, c->cap_entries);
    SA(s, c->enext, c->entries, c->cap_entries);
    SA(s, c->ckey, c->cmask + 1, c->cmask + 1); SA(s, c->cval, c->cmask + 1, c->cmask + 1);
    SA(s, c->akey, c->amask + 1, c->amask + 1); SA(s, c->aval, c->amask + 1, c->amask + 1);
    SA(s, c->acode, c->asize, c->acap);
    SA(s, c->sit_new, c->nsit, c->nsit); SA(s, c->sit_total, c->nsit, c->nsit);
    SA(s, c->hist, (uint64_t)c->hlen, (uint64_t)(c->depth > 0 ? c->depth : 1));
    SZ(s, c->stamp, c->stamp_cap);
    if (!s->writing) c->cur_stamp = 0;
    SA(s, c->text, c->text_len, c->text_cap);
    SA(s, c->ep_table, c->ep_mask + 1, c->ep_mask + 1);
    if (c->lines_on) SA(s, c->ln_table, c->ln_mask + 1, c->ln_mask + 1);
    for (int k = 0; k < c->ep_n; k++) SA(s, c->ep_tabs[k], c->ep_mask + 1, c->ep_mask + 1);
    SA(s, c->v_right, c->verdict_cells, c->verdict_cells); SA(s, c->v_asked, c->verdict_cells, c->verdict_cells);
    if (c->mix_on) {
        if (!s->writing && !tabs_ready) make_tables();
        SA(s, c->w, (uint64_t)c->nsets * c->nin, (uint64_t)c->nsets * c->nin);
        SA(s, c->set_updates, (uint64_t)c->nsets, (uint64_t)c->nsets);
        SZ(s, c->dist, c->dist_cap);
        SZ(s, c->lv_code, c->lv_cap); SZ(s, c->lv_cnt, c->lv_cap);
        SZ(s, c->lv_len, (uint64_t)c->depth + 4);
        SA(s, c->ext_n1, 1u << 17, 1u << 17); SA(s, c->ext_n, 1u << 17, 1u << 17);
        SA(s, c->wkey, c->wmask + 1, c->wmask + 1); SA(s, c->wval, c->wmask + 1, c->wmask + 1);
        SA(s, c->apm_n1, (uint64_t)65536 * 24, (uint64_t)65536 * 24);
        SA(s, c->apm_n, (uint64_t)65536 * 24, (uint64_t)65536 * 24);
        SA(s, c->w2, 32, 32);
        SA(s, c->wg, (uint64_t)8 * c->nin, (uint64_t)8 * c->nin);
        if (c->recent) {
            SA(s, c->rm_n1, c->rm_mask + 1, c->rm_mask + 1); SA(s, c->rm_n, c->rm_mask + 1, c->rm_mask + 1);
            SA(s, c->rm_tag, c->rm_mask + 1, c->rm_mask + 1);
        }
        if (c->voice_on) {
            SZ(s, c->vc_code, c->vc_ccap); SZ(s, c->vc_off, c->vc_ncap); SZ(s, c->vc_len, c->vc_ncap);
            SZ(s, c->vc_w, c->vc_ncap); SZ(s, c->vc_live, c->vc_ncap);
            SZ(s, c->vt_code, c->vc_ncap); SZ(s, c->vt_cp, c->vc_ncap); SZ(s, c->vt_w, c->vc_ncap);
            if (!s->writing) { c->vc_n = 0; c->vc_pos = 0; c->vc_know = 0.0; }
        }
    }
}

int core_save(Core *c, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    Ser s = {f, 1, 1};
    ser_core(&s, c);
    if (fclose(f) != 0) s.ok = 0;
    return s.ok;
}

Core *core_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    Core *c = calloc(1, sizeof(Core));
    Ser s = {f, 0, 1};
    ser_core(&s, c);
    int extra = fgetc(f);                                    /* the file must end where the core ends */
    fclose(f);
    if (!s.ok || extra != EOF) { core_free(c); return 0; }
    return c;
}

/* what a loaded core is: depth, update "found", episode, harmony, voice, lines */
void core_config(Core *c, int *out) {
    out[0] = c->depth; out[1] = c->update_found; out[2] = c->episode; out[3] = c->mix_on; out[4] = c->voice_on;
    out[5] = c->lines_on;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* Sleep (MODEL.md 8): every context met fewer than `keep_min` times is forgotten, with the longer contexts that
 * extend it (a longer context is never met more often than its suffix), and so are the word and line contexts met
 * fewer times. What recurred stays exactly as it was: its counts, its entries in their order. The situations, the
 * episodes, the history and every learned weight stay. The store is then compacted. Returns how many nodes were
 * forgotten. */
uint64_t core_sleep(Core *c, uint32_t keep_min) {
    uint64_t N = c->nodes;
    if (N <= 1 || keep_min <= 1) return 0;
    uint8_t *keep = malloc(N);
    for (uint64_t v = 0; v < N; v++) keep[v] = v == 0 || c->n[v] >= keep_min;
    for (int changed = 1; changed;) {
        changed = 0;
        for (uint64_t i = 0; i <= c->cmask; i++) if (c->ckey[i]) {
            uint32_t parent = (uint32_t)((c->ckey[i] - 1) >> 21), v = c->cval[i];
            if (keep[v] && !keep[parent]) { keep[v] = 0; changed = 1; }
        }
    }
    uint32_t *nid = malloc(N * 4);
    uint64_t m = 0, ecount = 1;
    for (uint64_t v = 0; v < N; v++) {
        nid[v] = keep[v] ? (uint32_t)m++ : NONE;
        if (keep[v]) ecount += c->t[v];
    }
    uint64_t cap_n = 1 << 16, cap_e = 1 << 16;
    while (cap_n < m) cap_n *= 2;
    while (cap_e < ecount) cap_e *= 2;
    uint32_t *n2 = malloc(cap_n * 4), *t2 = malloc(cap_n * 4), *h2 = malloc(cap_n * 4), *tl2 = malloc(cap_n * 4);
    uint32_t *es2 = malloc(cap_e * 4), *ec2 = malloc(cap_e * 4), *en2 = malloc(cap_e * 4);
    es2[0] = ec2[0] = en2[0] = 0;
    uint64_t e2 = 1;
    for (uint64_t v = 0; v < N; v++) if (keep[v]) {
        uint32_t w = nid[v];
        n2[w] = c->n[v]; t2[w] = c->t[v]; h2[w] = tl2[w] = 0;
        for (uint32_t q = c->head[v]; q; q = c->enext[q]) {
            es2[e2] = c->esym[q]; ec2[e2] = c->ecnt[q]; en2[e2] = 0;
            if (tl2[w]) en2[tl2[w]] = (uint32_t)e2; else h2[w] = (uint32_t)e2;
            tl2[w] = (uint32_t)e2;
            e2++;
        }
    }
    uint64_t edges = 0;
    for (uint64_t i = 0; i <= c->cmask; i++) if (c->ckey[i] && keep[c->cval[i]]) edges++;
    uint64_t ccap = 1 << 16;
    while ((edges + 1) * 5 > ccap * 3) ccap *= 2;
    uint64_t *ck2 = calloc(ccap, 8); uint32_t *cv2 = malloc(ccap * 4);
    for (uint64_t i = 0; i <= c->cmask; i++) if (c->ckey[i] && keep[c->cval[i]]) {
        uint64_t old = c->ckey[i] - 1;
        uint32_t parent = (uint32_t)(old >> 21), sym = (uint32_t)(old & ((1u << 21) - 1));
        uint64_t key = (((uint64_t)nid[parent] << 21) | sym) + 1;
        uint64_t h = mix64(key) & (ccap - 1);
        while (ck2[h]) h = (h + 1) & (ccap - 1);
        ck2[h] = key; cv2[h] = nid[c->cval[i]];
    }
    if (c->wkey) {                                      /* word and line contexts */
        uint64_t wcap = c->wmask + 1, wused = 0;
        uint64_t *wk2 = calloc(wcap, 8); uint32_t *wv2 = malloc(wcap * 4);
        for (uint64_t i = 0; i < wcap; i++) if (c->wkey[i] && keep[c->wval[i]]) {
            uint64_t h = mix64(c->wkey[i]) & c->wmask;
            while (wk2[h]) h = (h + 1) & c->wmask;
            wk2[h] = c->wkey[i]; wv2[h] = nid[c->wval[i]]; wused++;
        }
        free(c->wkey); free(c->wval);
        c->wkey = wk2; c->wval = wv2; c->wused = wused;
    }
    free(c->n); free(c->t); free(c->head); free(c->tail); free(c->esym); free(c->ecnt); free(c->enext);
    free(c->ckey); free(c->cval);
    c->n = n2; c->t = t2; c->head = h2; c->tail = tl2; c->nodes = m; c->cap_nodes = cap_n;
    c->esym = es2; c->ecnt = ec2; c->enext = en2; c->entries = e2; c->cap_entries = cap_e;
    c->ckey = ck2; c->cval = cv2; c->cmask = ccap - 1; c->cused = edges;
    free(keep); free(nid);
    return N - m;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* Retraction (MODEL.md 2.5, for the core): take back exactly the counts that living `codes` after the history `hist`
 * added, in the cortex (growth m = 1, update "all") and in the word and line contexts; and overwrite the episode text
 * of that stretch (from position `text_from`), so that it can never be recalled or completed again. What depends on
 * the order of life (the situations' records, the gate's weights, the calibration, the short-term memory) is not
 * taken back. A symbol first met in that stretch keeps its number, with no count. Returns 1, or 0 if the core cannot
 * retract exactly (update "found"). */
static void table_delete(uint64_t *keys, uint32_t *vals, uint64_t mask, uint64_t slot) {
    uint64_t i = slot;
    keys[i] = 0;
    for (uint64_t j = (i + 1) & mask; keys[j]; j = (j + 1) & mask) {      /* backward shift of the probe run */
        uint64_t home = mix64(keys[j]) & mask;
        int between = i <= j ? (home > i && home <= j) : (home > i || home <= j);
        if (!between) { keys[i] = keys[j]; vals[i] = vals[j]; keys[j] = 0; i = j; }
    }
}

static uint64_t table_find(uint64_t *keys, uint64_t mask, uint64_t key) {
    uint64_t h = mix64(key) & mask;
    while (keys[h]) { if (keys[h] == key) return h; h = (h + 1) & mask; }
    return (uint64_t)-1;
}

static void untake(Core *c, uint32_t v, uint32_t x) {    /* one count of x less at node v (never below zero) */
    uint32_t prev = 0;
    for (uint32_t e = c->head[v]; e; prev = e, e = c->enext[e]) if (c->esym[e] == x) {
        if (c->n[v] > 0) c->n[v] -= 1;
        if (--c->ecnt[e] == 0) {
            if (prev) c->enext[prev] = c->enext[e]; else c->head[v] = c->enext[e];
            if (c->tail[v] == e) c->tail[v] = prev;
            c->t[v] -= 1;
        }
        return;
    }
}

int core_retract(Core *c, const uint32_t *codes, uint64_t n, const uint32_t *hist, uint64_t nh, uint64_t text_from) {
    if (c->update_found) return 0;
    uint32_t *save_hist = malloc((c->depth + 1) * 4);
    memcpy(save_hist, c->hist, c->hlen * 4);
    int save_hlen = c->hlen;
    uint64_t save_cw = c->cur_word, save_pw = c->prev_word, save_lh = c->line_h;
    uint32_t save_ll = c->line_len, save_pc = c->prev_code;
    core_set_history(c, hist, nh);
    uint32_t *chain = malloc((c->depth + 2) * 4);
    for (uint64_t i = 0; i < n; i++) {
        uint32_t x = symbol_number(c, codes[i]);
        if (x == NONE) continue;                         /* never lived: nothing to take back */
        int len = chain_of(c, chain);
        for (int d = len - 1; d >= 0; d--) {             /* the deepest first, so a node left empty can go */
            uint32_t v = chain[d];
            untake(c, v, x);
            if (d > 0 && c->n[v] == 0) {
                uint64_t key = (((uint64_t)chain[d - 1] << 21) | c->hist[c->hlen - d]) + 1;
                uint64_t slot = table_find(c->ckey, c->cmask, key);
                if (slot != (uint64_t)-1) { table_delete(c->ckey, c->cval, c->cmask, slot); c->cused--; }
            }
        }
        if (c->mix_on && c->wkey) {
            uint64_t keys[3]; int nk = 0;
            if (c->words_on) {
                keys[nk++] = mix64(0x5751 ^ c->cur_word);
                keys[nk++] = mix64(0x5752 ^ mix64(c->prev_word) ^ c->cur_word);
            }
            if (c->lines_on) keys[nk++] = mix64(0x4C494E45ull ^ c->line_h ^ ((uint64_t)c->line_len << 40));
            for (int k = 0; k < nk; k++) {
                uint64_t key = keys[k] ? keys[k] : 1;
                uint64_t slot = table_find(c->wkey, c->wmask, key);
                if (slot == (uint64_t)-1) continue;
                uint32_t v = c->wval[slot];
                untake(c, v, x);
                if (c->n[v] == 0) { table_delete(c->wkey, c->wval, c->wmask, slot); c->wused--; }
            }
        }
        c->lived -= 1;
        uint32_t cp = codes[i];
        if (c->words_on) {
            if (is_word_char(cp)) c->cur_word = mix64(c->cur_word ^ (cp + 0x100000000ull));
            else if (c->cur_word) { c->prev_word = c->cur_word; c->cur_word = 0; }
        }
        if (c->lines_on) {
            if (cp == 10) { c->line_h = 0x6C696E65ull; c->line_len = 0; }
            else { c->line_h = mix64(c->line_h ^ cp); c->line_len++; }
        }
        push_history(c, x);
    }
    if (c->episode)                                       /* the stretch can no longer be recalled */
        for (uint64_t p = text_from; p < text_from + n && p < c->text_len; p++) c->text[p] = 0xFFFFFFFEu;
    memcpy(c->hist, save_hist, save_hlen * 4); c->hlen = save_hlen;
    c->cur_word = save_cw; c->prev_word = save_pw; c->line_h = save_lh; c->line_len = save_ll; c->prev_code = save_pc;
    free(save_hist); free(chain);
    return 1;
}

/* a digest of the substance: every context reachable from the root with its count and its entries (sorted), and every
 * word and line context with its count and entries; independent of node numbers and of the order entries were made */
static uint64_t node_digest(Core *c, uint32_t v) {
    uint64_t acc = mix64(0x6E6F6465ull ^ c->n[v]);
    uint64_t sum = 0;                                    /* entries in any order: a commutative sum of their hashes */
    for (uint32_t e = c->head[v]; e; e = c->enext[e]) sum += mix64(((uint64_t)c->acode[c->esym[e]] << 32) | c->ecnt[e]);
    return mix64(acc ^ sum);
}

uint64_t core_digest(Core *c) {
    /* contexts: each child edge is hashed with its path from the root, so a context is identified by its symbols */
    uint64_t total = node_digest(c, 0);
    uint64_t N = c->nodes;
    uint64_t *path = calloc(N ? N : 1, 8);               /* the hash of each node's context */
    uint8_t *done = calloc(N ? N : 1, 1);
    done[0] = 1; path[0] = 0x726F6F74ull;
    for (int changed = 1; changed;) {                    /* parents before children */
        changed = 0;
        for (uint64_t i = 0; i <= c->cmask; i++) if (c->ckey[i]) {
            uint64_t k = c->ckey[i] - 1;
            uint32_t parent = (uint32_t)(k >> 21), sym = (uint32_t)(k & ((1u << 21) - 1)), v = c->cval[i];
            if (done[parent] && !done[v]) {
                path[v] = mix64(path[parent] ^ (0x5000000000ull + c->acode[sym]));
                done[v] = 1; changed = 1;
                total += mix64(path[v] ^ node_digest(c, v));
            }
        }
    }
    if (c->wkey)
        for (uint64_t i = 0; i <= c->wmask; i++) if (c->wkey[i])
            total += mix64(c->wkey[i] ^ node_digest(c, c->wval[i]) ^ 0x776F7264ull);
    free(path); free(done);
    return total;
}
