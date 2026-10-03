"""Parametrizzazione SVI della varianza totale implicita, una scadenza alla volta.

    w(k) = a + b ( rho (k - m) + sqrt((k - m)^2 + sigma^2) ),   k = ln(K / F),   w = IV^2 T

Il fit usa la riscrittura (Zeliade): con y = (k - m) / sigma,
    w = a + (p/2) (sqrt(y^2 + 1) + y) + (q/2) (sqrt(y^2 + 1) - y),
dove p = b sigma (1 + rho) e q = b sigma (1 - rho) sono le pendenze delle due ali moltiplicate
per sigma. Fissati m e sigma il problema è lineare in (a, p, q) con p, q >= 0.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, minimize

from . import bs


@dataclass(frozen=True)
class SVI:
    a: float
    b: float
    rho: float
    m: float
    sigma: float


def w(k, p):
    x = np.asarray(k, dtype=float) - p.m
    return p.a + p.b * (p.rho * x + np.sqrt(x * x + p.sigma**2))


def dw(k, p):
    x = np.asarray(k, dtype=float) - p.m
    return p.b * (p.rho + x / np.sqrt(x * x + p.sigma**2))


def d2w(k, p):
    x = np.asarray(k, dtype=float) - p.m
    return p.b * p.sigma**2 / (x * x + p.sigma**2) ** 1.5


def durrleman_wdd(k, wk, w1, w2):
    """g(k) >= 0 per ogni k  <=>  densità implicita non negativa (niente arbitraggio butterfly).

    Dipende solo da w, w' e w'' nel punto k, quindi vale per qualunque curva di varianza totale.
    """
    k = np.asarray(k, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (1 - k * w1 / (2 * wk)) ** 2 - 0.25 * w1**2 * (1 / wk + 0.25) + 0.5 * w2
    return np.where(wk > 0, g, -np.inf)


def durrleman(k, p):
    return durrleman_wdd(k, w(k, p), dw(k, p), d2w(k, p))


def _da_pq(a, pp, qq, m, sigma):
    b = (pp + qq) / (2 * sigma)
    rho = (pp - qq) / (pp + qq) if pp + qq > 0 else 0.0
    return SVI(a, b, rho, m, sigma)


def _interno(k, wm, wt, m, sigma):
    # problema lineare in (a, p, q) con 0 <= p, q <= 2 sigma (pendenze delle ali <= 2: Lee)
    y = (k - m) / sigma
    s = np.sqrt(y * y + 1)
    A = np.column_stack([np.ones_like(y), 0.5 * (s + y), 0.5 * (s - y)])
    sw = np.sqrt(wt)
    ris = lsq_linear(A * sw[:, None], wm * sw,
                     bounds=([-np.inf, 0.0, 0.0], [np.inf, 2 * sigma, 2 * sigma]))
    a, pp, qq = ris.x
    # il minimo di w vale a + sqrt(p q): deve essere positivo
    return (a, pp, qq), (ris.cost if a + np.sqrt(pp * qq) > 0 else np.inf)


def fit_slice(k, wm, weights=None):
    """Adatta SVI a una scadenza. k: log-moneyness, wm: varianza totale di mercato."""
    k, wm = np.asarray(k, dtype=float), np.asarray(wm, dtype=float)
    wt = np.ones_like(k) if weights is None else np.asarray(weights, dtype=float)
    wt = wt / wt.mean()

    def costo(z):
        return _interno(k, wm, wt, z[0], np.exp(z[1]))[1]

    # ricerca su griglia in (m, log sigma), poi affinamento dei tre punti migliori
    candidati = []
    for m in np.linspace(k.min(), k.max(), 15):
        for ls in np.linspace(np.log(0.01), np.log(1.0), 15):
            candidati.append((costo([m, ls]), m, ls))
    candidati.sort()
    migliore = None
    for _, m, ls in candidati[:3]:
        r = minimize(costo, [m, ls], method="Nelder-Mead",
                     options=dict(xatol=1e-10, fatol=1e-16, maxiter=2000))
        if migliore is None or r.fun < migliore.fun:
            migliore = r
    m, sigma = migliore.x[0], np.exp(migliore.x[1])
    (a, pp, qq), _ = _interno(k, wm, wt, m, sigma)
    return _da_pq(a, pp, qq, m, sigma)


def fit_chain(df, k_check=None):
    """Un fit SVI per ogni scadenza di una chain (colonne T, K, kind, mid, S, r, q, iv).

    Colonne del risultato: parametri SVI, rmse_iv (errore del fit in IV), g_min (Durrleman su
    k_check, di default [-1.5, 1.5]), g_dati (Durrleman sul solo range dei dati), n quote,
    k_min e k_max dei dati, al_limite (vedi sotto). Le scadenze con meno di 5 quote valide si
    saltano: si veda `fit_chain(...).index` per sapere quali restano.
    """
    k_check = np.linspace(-1.5, 1.5, 601) if k_check is None else k_check
    righe = []
    for T, g in df.groupby("T"):
        g = g.dropna(subset=["iv"])  # quote senza volatilità implicita (es. prezzo nullo)
        if len(g) < 5:
            continue
        F = g["S"].iloc[0] * np.exp((g["r"].iloc[0] - g["q"].iloc[0]) * T)
        k = np.log(g["K"].values / F)
        wm = g["iv"].values ** 2 * T
        vega = np.array([bs.vega(s, K, T, r, q, iv) for s, K, r, q, iv in
                         zip(g["S"], g["K"], g["r"], g["q"], g["iv"], strict=True)])
        p = fit_slice(k, wm, vega)
        iv_mod = np.sqrt(w(k, p) / T)
        righe.append(dict(T=T, a=p.a, b=p.b, rho=p.rho, m=p.m, sigma=p.sigma,
                          rmse_iv=np.sqrt(np.mean((iv_mod - g["iv"].values) ** 2)),
                          g_min=durrleman(k_check, p).min(),
                          g_dati=durrleman(np.linspace(k.min(), k.max(), 300), p).min(),
                          n=len(k), k_min=k.min(), k_max=k.max(),
                          # pendenza di un'ala sul limite di Lee: l'altra ala non è identificata
                          # dai dati (di solito dati su un solo lato), la curva vale solo nel range
                          al_limite=bool(max(p.b * (1 + p.rho), p.b * (1 - p.rho)) >= 2 - 1e-6)))
    return pd.DataFrame(righe).set_index("T")


def calendar_min(fits, k_grid=None):
    """Minimo di w(k, T_{j+1}) - w(k, T_j): se è negativo c'è arbitraggio calendar.
    Senza k_grid, per ogni coppia di scadenze si guarda solo l'intervallo di k coperto dai dati di entrambe."""
    P = [SVI(r.a, r.b, r.rho, r.m, r.sigma) for r in fits.itertuples()]
    out = []
    for j in range(len(P) - 1):
        if k_grid is None:
            lo = max(fits["k_min"].iloc[j], fits["k_min"].iloc[j + 1])
            hi = min(fits["k_max"].iloc[j], fits["k_max"].iloc[j + 1])
            k = np.linspace(lo, hi, 61)
        else:
            k = k_grid
        out.append((w(k, P[j + 1]) - w(k, P[j])).min())
    return min(out)
