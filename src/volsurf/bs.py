"""Black-Scholes-Merton con dividend yield continuo: prezzo, greche, volatilità implicita.

Convenzioni: S spot, K strike, T scadenza in anni, r tasso continuo, q dividend yield
continuo, sigma volatilità annua (0.20 = 20%). La vega è per variazione unitaria di
sigma (non per punto percentuale).
Richiede T > 0 e sigma > 0.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _check_kind(kind):
    if kind not in ("call", "put"):
        raise ValueError("kind deve essere 'call' o 'put'")


def _d1_d2(S, K, T, r, q, sigma):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    return d1, d1 - sigma * sqrt_T


def price(S, K, T, r, q, sigma, kind="call"):
    _check_kind(kind)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    if kind == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def delta(S, K, T, r, q, sigma, kind="call"):
    _check_kind(kind)
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    if kind == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    return -np.exp(-q * T) * norm.cdf(-d1)


def gamma(S, K, T, r, q, sigma):
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, q, sigma):
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def price_bounds(S, K, T, r, q, kind="call"):
    """Limiti di no-arbitraggio sul prezzo (inferiore, superiore) per un'opzione europea."""
    _check_kind(kind)
    fs = S * np.exp(-q * T)
    fk = K * np.exp(-r * T)
    if kind == "call":
        return max(fs - fk, 0.0), fs
    return max(fk - fs, 0.0), fk


def implied_vol(p, S, K, T, r, q, kind="call", lo=1e-6, hi=5.0):
    """Volatilità implicita da un prezzo, oppure nan se non esiste.

    Il prezzo è strettamente crescente in sigma (vega > 0), quindi se p sta nei limiti
    di no-arbitraggio la soluzione è unica e si trova con Brent su [lo, hi].
    """
    if T <= 0 or S <= 0 or K <= 0 or not np.isfinite(p):
        return np.nan
    lower, upper = price_bounds(S, K, T, r, q, kind)
    if p <= lower or p >= upper:
        return np.nan

    def f(s):
        return price(S, K, T, r, q, s, kind) - p

    if f(lo) > 0 or f(hi) < 0:
        return np.nan
    return brentq(f, lo, hi, xtol=1e-12)


def implied_vol_array(p, S, K, T, r, q, kind):
    """Versione vettoriale di implied_vol (kind può essere un array di 'call'/'put')."""
    return np.vectorize(implied_vol, otypes=[float])(p, S, K, T, r, q, kind)
