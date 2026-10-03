"""Opzioni americane (esercizio anticipato) con albero binomiale di Cox-Ross-Rubinstein.

Ad ogni nodo il valore è il massimo tra continuazione ed esercizio immediato. Per ridurre
l'oscillazione dell'albero al variare dei passi, all'ultimo passo prima della scadenza la
continuazione è il prezzo di Black-Scholes (Broadie-Detemple, "BBS"). Il dividend yield q è
continuo. Tutte le funzioni lavorano su array: strike, volatilità e tipo possono essere
vettori (un'opzione per elemento), mentre T, r, q e il numero di passi sono comuni.
"""
import numpy as np

from . import bs

N_PASSI = 300


def _est(x):
    return np.atleast_1d(np.asarray(x, dtype=float))


def _call(kind, n):
    k = np.broadcast_to(np.asarray(kind), (n,)) if np.ndim(kind) else np.full(n, kind)
    if not np.isin(k, ("call", "put")).all():
        raise ValueError("kind deve essere 'call' o 'put'")
    return k == "call"


def _intrinseco(S, K, is_call):
    return np.where(is_call, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))


def _bs(S, K, T, r, q, sigma, is_call):
    return np.where(is_call, bs.price(S, K, T, r, q, sigma, "call"),
                    bs.price(S, K, T, r, q, sigma, "put"))


def prezzo(S, K, T, r, q, sigma, kind="call", N=N_PASSI):
    """Prezzo americano. K, sigma, kind: scalari o array della stessa lunghezza B."""
    K, sigma = np.broadcast_arrays(_est(K), _est(sigma))
    B = len(K)
    is_call = _call(kind, B)[:, None]
    K, sig = K[:, None], sigma[:, None]
    dt = T / N
    mosse = np.exp(sig * np.sqrt(dt))                     # u = e^{sigma sqrt(dt)}, d = 1/u
    p = (np.exp((r - q) * dt) - 1 / mosse) / (mosse - 1 / mosse)
    if (p < 0).any() or (p > 1).any():       # sigma troppo piccolo rispetto al passo: albero non valido
        raise ValueError("sigma troppo piccolo per questo numero di passi (probabilità fuori da [0,1])")
    disc = np.exp(-r * dt)
    j = np.arange(N)
    nodi = S * mosse ** (2 * j - (N - 1))                  # prezzi al passo N-1
    V = np.maximum(_intrinseco(nodi, K, is_call), _bs(nodi, K, dt, r, q, sig, is_call))
    for _ in range(N - 1):
        V = disc * (p * V[:, 1:] + (1 - p) * V[:, :-1])
        nodi = nodi[:, 1:] / mosse           # S u^(2j-i) = S u^(2(j+1)-(i+1)) / u
        V = np.maximum(V, _intrinseco(nodi, K, is_call))
    return V[:, 0]


def iv(p, S, K, T, r, q, kind="call", N=N_PASSI, lo=0.02, hi=3.0, tol=1e-8):
    """Volatilità implicita americana per bisezione (il prezzo cresce con sigma), vettoriale.
    nan se il prezzo non supera il valore di esercizio immediato o sta fuori da [lo, hi]."""
    p, K = np.broadcast_arrays(_est(p), _est(K))
    B = len(p)
    if T <= 0:
        return np.full(B, np.nan)
    is_call = _call(kind, B)
    kind_v = np.where(is_call, "call", "put")
    a, b = np.full(B, lo), np.full(B, hi)
    ok = np.isfinite(p) & (p > _intrinseco(S, K, is_call))
    ok &= prezzo(S, K, T, r, q, a, kind_v, N) <= p
    ok &= prezzo(S, K, T, r, q, b, kind_v, N) >= p
    for _ in range(int(np.ceil(np.log2((hi - lo) / tol)))):
        m = (a + b) / 2
        sotto = prezzo(S, K, T, r, q, m, kind_v, N) < p
        a, b = np.where(sotto, m, a), np.where(sotto, b, m)
    return np.where(ok, (a + b) / 2, np.nan)
