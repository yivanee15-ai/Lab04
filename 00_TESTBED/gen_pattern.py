#!/usr/bin/env python3
"""Generate input / golden data files for Lab04 PATTERN.v (Flash Attention).

Usage:  python3 gen_pattern.py [--n 1000] [--seed 1] [--out DAT] [--no-transpose-w] [--golden online|f64]

Files written to <out>/ (one word per line, hex, $readmemh format):
    K.dat Q.dat V.dat   n*64 words of float32  (token-major raster: 16 tokens x 4 dims)
    W.dat               n*16 words of float32  (4x4 raster)
    GOLD.dat            n*64 words of float64  (16 hex digits) pass/fail reference: float32 online model
                        (default) or float64 truth with --golden f64
    REF.dat             n*64 words of float64  float64 truth, only used to print the error vs it
    GOLD_F32.dat        n*64 words of float32  (GOLD.dat rounded to float32, for waveform debug)

Assumptions taken from the lab PDF (confirm with the TA where marked ?):
  * head1 = dims 0,1 ; head2 = dims 2,3 ; head_dim = 2
  * Score = Q_head x K_head^T (16x16), divided by sqrt(2), softmax over each row, no mask
  * Head_out = softmax(Score) x V_head ; Concat = [head1 | head2]  (16x4)
  * Final = Concat x Out_weight^T  (?) -- W is delivered in raster order as the matrix W[c][k],
    so Final[t][c] = sum_k Concat[t][k] * W[c][k].  Use --no-transpose-w for Concat x W instead.
  * Ranges: |K| in [0.5,255]  |Q| in [0,0.5]  |V| in [0,0.5]  |W| in [0,0.5], random sign
"""
import argparse
import os
import numpy as np

F32 = np.float32
N_TOK, DIM, HEADS, HD, BLK = 16, 4, 2, 2, 2

K_LO, K_HI = 0.5, 255.0
Q_LO, Q_HI = 0.0, 0.5
V_LO, V_HI = 0.0, 0.5
W_LO, W_HI = 0.0, 0.5


# ----------------------------------------------------------------------------
# golden models
# ----------------------------------------------------------------------------
def golden_f64(Q, K, V, W, transpose_w=True):
    """Exact reference in float64 (standard softmax, no tiling)."""
    Q, K, V, W = (a.astype(np.float64) for a in (Q, K, V, W))
    heads = []
    for h in range(HEADS):
        sl = slice(HD * h, HD * h + HD)
        s = (Q[:, sl] @ K[:, sl].T) / np.sqrt(2.0)
        s = s - s.max(axis=1, keepdims=True)
        p = np.exp(s)
        p = p / p.sum(axis=1, keepdims=True)
        heads.append(p @ V[:, sl])
    concat = np.concatenate(heads, axis=1)
    return concat @ (W.T if transpose_w else W)


def row_max_scores(Q, K):
    """max_j S_ij / sqrt(2) for every head/row (to study the exp() input range)."""
    out = []
    for h in range(HEADS):
        sl = slice(HD * h, HD * h + HD)
        s = (Q[:, sl].astype(np.float64) @ K[:, sl].T.astype(np.float64)) / np.sqrt(2.0)
        out.append(s.max(axis=1))
    return np.concatenate(out)


def online_f32(Q, K, V, W, transpose_w=True):
    """float32 model of the tiled online-softmax datapath described in the PDF.

    Every operation is a separate IEEE float32 op (like one DW_fp_mult / DW_fp_add /
    DW_fp_div / DW_fp_exp each), in this order:
        S   = (q0*k0 + q1*k1) / sqrt2          (sqrt2 = the constant given in the PDF)
        m'  = max(m, rowmax(S))     alpha = exp(m - m')     p = exp(S - m')
        l   = alpha*l + (p0 + p1)
        O   = alpha*O + (p0*v0 + p1*v1)        (per output dim)
        out = O / l ;  Final[t][c] = ((c0*w0 + c1*w1) + c2*w2) + c3*w3
    float32 results depend on this order (e.g. dividing by sqrt2 vs multiplying by 1/sqrt2
    changes ~58% of the patterns by more than 1e-7), so keep the RTL in the same order.
    """
    sqrt2 = F32(np.sqrt(2.0))
    heads = []
    for h in range(HEADS):
        sl = slice(HD * h, HD * h + HD)
        q, k, v = Q[:, sl], K[:, sl], V[:, sl]
        out = np.zeros((N_TOK, HD), F32)
        for bi in range(N_TOK // BLK):
            qb = q[BLK * bi:BLK * bi + BLK]
            m = np.full(BLK, -np.inf, F32)
            l = np.zeros(BLK, F32)
            acc = np.zeros((BLK, HD), F32)
            for bj in range(N_TOK // BLK):
                kb = k[BLK * bj:BLK * bj + BLK]
                vb = v[BLK * bj:BLK * bj + BLK]
                s = (qb[:, 0:1] * kb[:, 0][None, :] + qb[:, 1:2] * kb[:, 1][None, :]) / sqrt2
                m_new = np.maximum(m, np.maximum(s[:, 0], s[:, 1]))
                alpha = np.exp(m - m_new)
                p = np.exp(s - m_new[:, None])
                l = alpha * l + (p[:, 0] + p[:, 1])
                pv = p[:, 0:1] * vb[0][None, :] + p[:, 1:2] * vb[1][None, :]
                acc = alpha[:, None] * acc + pv
                m = m_new
            out[BLK * bi:BLK * bi + BLK] = acc / l[:, None]
        heads.append(out)
    c = np.concatenate(heads, axis=1)
    if transpose_w:
        res = [((c[:, 0] * W[j, 0] + c[:, 1] * W[j, 1]) + c[:, 2] * W[j, 2]) + c[:, 3] * W[j, 3] for j in range(DIM)]
    else:
        res = [((c[:, 0] * W[0, j] + c[:, 1] * W[1, j]) + c[:, 2] * W[2, j]) + c[:, 3] * W[3, j] for j in range(DIM)]
    return np.stack(res, axis=1).astype(F32)


# ----------------------------------------------------------------------------
# stimulus
# ----------------------------------------------------------------------------
def rand_sign(rng, shape):
    return np.where(rng.random(shape) < 0.5, -1.0, 1.0)


def rand_mag(rng, shape, lo, hi, mode):
    if mode == "uniform":
        m = rng.uniform(lo, hi, shape)
    elif mode == "log":
        m = np.exp(rng.uniform(np.log(max(lo, 1e-3)), np.log(hi), shape))
        if lo == 0.0:                       # let some values hit exactly 0
            m = np.where(rng.random(shape) < 0.05, 0.0, m)
    else:                                   # "extreme": mostly the range limits
        pick = rng.integers(0, 3, shape)
        m = np.where(pick == 0, lo, np.where(pick == 1, hi, rng.uniform(lo, hi, shape)))
    return m


def make_matrix(rng, shape, lo, hi, mode):
    return (rand_sign(rng, shape) * rand_mag(rng, shape, lo, hi, mode)).astype(F32)


def directed_cases():
    """Corner cases that stress exp() range, softmax max tracking and zero data."""
    cases = []
    full = lambda x, s=(N_TOK, DIM): np.full(s, x, F32)
    rnd_w = lambda rng: make_matrix(rng, (DIM, DIM), W_LO, W_HI, "uniform")

    # all scores equal and maximal positive / maximal negative (|S/sqrt2| ~ 180)
    cases.append(("max_pos_equal", lambda rng: (full(Q_HI), full(K_HI), make_matrix(rng, (N_TOK, DIM), V_LO, V_HI, "uniform"), rnd_w(rng))))
    cases.append(("max_neg_equal", lambda rng: (full(Q_HI), full(-K_HI), make_matrix(rng, (N_TOK, DIM), V_LO, V_HI, "uniform"), rnd_w(rng))))
    # all scores zero
    cases.append(("q_zero", lambda rng: (full(0.0), make_matrix(rng, (N_TOK, DIM), K_LO, K_HI, "uniform"), make_matrix(rng, (N_TOK, DIM), V_LO, V_HI, "uniform"), rnd_w(rng))))
    cases.append(("v_zero", lambda rng: (make_matrix(rng, (N_TOK, DIM), Q_LO, Q_HI, "uniform"), make_matrix(rng, (N_TOK, DIM), K_LO, K_HI, "uniform"), full(0.0), rnd_w(rng))))
    cases.append(("w_zero", lambda rng: (make_matrix(rng, (N_TOK, DIM), Q_LO, Q_HI, "uniform"), make_matrix(rng, (N_TOK, DIM), K_LO, K_HI, "uniform"), make_matrix(rng, (N_TOK, DIM), V_LO, V_HI, "uniform"), full(0.0, (DIM, DIM)))))
    # smallest keys: scores tiny, softmax almost uniform
    cases.append(("k_min", lambda rng: (make_matrix(rng, (N_TOK, DIM), Q_LO, Q_HI, "uniform"), full(K_LO) * rand_sign(rng, (N_TOK, DIM)).astype(F32), make_matrix(rng, (N_TOK, DIM), V_LO, V_HI, "uniform"), rnd_w(rng))))

    # one dominant key token t (everything else tiny): softmax ~ one-hot, tests running max
    def dominant(t):
        def f(rng):
            Q = full(Q_HI)
            K = full(K_LO)
            K[t, :] = K_HI
            V = make_matrix(rng, (N_TOK, DIM), V_LO, V_HI, "uniform")
            return Q, K, V, rnd_w(rng)
        return f
    for t in (0, 1, 7, 8, 14, 15):
        cases.append((f"dominant_{t}", dominant(t)))

    # every score negative (row max itself is negative): a fixed positive offset would underflow
    def all_negative(rng):
        Q = full(Q_HI)
        K = -np.abs(make_matrix(rng, (N_TOK, DIM), K_LO, K_HI, "uniform"))
        return Q, K, make_matrix(rng, (N_TOK, DIM), V_LO, V_HI, "uniform"), rnd_w(rng)
    cases.append(("all_negative", all_negative))
    return cases


def gen_patterns(n, seed):
    rng = np.random.default_rng(seed)
    Qs, Ks, Vs, Ws, names = [], [], [], [], []
    for name, fn in directed_cases():
        if len(Qs) >= n:
            break
        Q, K, V, W = fn(rng)
        Qs.append(Q); Ks.append(K); Vs.append(V); Ws.append(W); names.append(name)
    modes = ["uniform"] * 6 + ["log"] * 2 + ["extreme"] * 2
    while len(Qs) < n:
        mode = modes[len(Qs) % len(modes)]
        Qs.append(make_matrix(rng, (N_TOK, DIM), Q_LO, Q_HI, mode))
        Ks.append(make_matrix(rng, (N_TOK, DIM), K_LO, K_HI, mode))
        Vs.append(make_matrix(rng, (N_TOK, DIM), V_LO, V_HI, mode))
        Ws.append(make_matrix(rng, (DIM, DIM), W_LO, W_HI, mode))
        names.append(mode)
    return Qs, Ks, Vs, Ws, names


# ----------------------------------------------------------------------------
def write_hex(path, arrays, width):
    dt, fmt = (np.uint32, "%08x") if width == 32 else (np.uint64, "%016x")
    with open(path, "w") as f:
        for a in arrays:
            for w in np.ascontiguousarray(a).reshape(-1).view(dt):
                f.write((fmt % int(w)) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "DAT"))
    ap.add_argument("--no-transpose-w", action="store_true")
    ap.add_argument("--golden", choices=["online", "f64"], default="online",
                    help="what GOLD.dat (the pass/fail reference) contains: float32 online model or float64 truth")
    a = ap.parse_args()
    tw = not a.no_transpose_w

    Qs, Ks, Vs, Ws, names = gen_patterns(a.n, a.seed)
    ref = [golden_f64(Q, K, V, W, tw) for Q, K, V, W in zip(Qs, Ks, Vs, Ws)]       # float64 truth
    onl = [online_f32(Q, K, V, W, tw) for Q, K, V, W in zip(Qs, Ks, Vs, Ws)]        # float32 model
    gold = ref if a.golden == "f64" else [o.astype(np.float64) for o in onl]

    os.makedirs(a.out, exist_ok=True)
    write_hex(os.path.join(a.out, "Q.dat"), Qs, 32)
    write_hex(os.path.join(a.out, "K.dat"), Ks, 32)
    write_hex(os.path.join(a.out, "V.dat"), Vs, 32)
    write_hex(os.path.join(a.out, "W.dat"), Ws, 32)
    write_hex(os.path.join(a.out, "GOLD.dat"), [g.astype(np.float64) for g in gold], 64)  # pass/fail reference
    write_hex(os.path.join(a.out, "REF.dat"), [g.astype(np.float64) for g in ref], 64)    # float64 truth (info only)
    write_hex(os.path.join(a.out, "GOLD_F32.dat"), [g.astype(F32) for g in gold], 32)     # waveform debug

    errs = np.array([np.abs(o.astype(np.float64) - r).max() for o, r in zip(onl, ref)])
    rmax = np.concatenate([row_max_scores(Q, K) for Q, K in zip(Qs, Ks)])
    rmax_rand = np.concatenate([row_max_scores(Q, K) for Q, K, n in zip(Qs, Ks, names)
                                if n in ("uniform", "log", "extreme")] or [np.zeros(1)])
    print(f"wrote {a.n} patterns to {a.out}  (seed={a.seed}, transpose_w={tw}, GOLD.dat={a.golden})")
    print(f"max |out| of golden             : {max(np.abs(g).max() for g in ref):.4f}")
    print(f"float32 model vs float64 truth  : max abs err {errs.max():.3e}, median {np.median(errs):.3e}, "
          f"patterns > 1e-7: {(errs > 1e-7).sum()}/{a.n}")
    print(f"row max of S/sqrt2 (all)        : min {rmax.min():.1f}  max {rmax.max():.1f}  "
          f"(float32 exp overflows above 88.7)")
    print(f"row max of S/sqrt2 (random only): min {rmax_rand.min():.1f}  max {rmax_rand.max():.1f}")


if __name__ == "__main__":
    main()
