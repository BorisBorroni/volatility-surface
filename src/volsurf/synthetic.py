"""Mercato sintetico.

Due mondi con lo stesso modello di Heston:
  P  mondo reale: governa come si muove davvero il titolo (usato dal simulatore);
  Q  mondo di prezzo: governa i prezzi delle opzioni (usato dal generatore di chain).
Il premio per il rischio di varianza `lam` li collega: sotto Q la varianza torna alla media
più lentamente e verso un livello più alto, quindi la volatilità implicita sta sopra
quella realizzata.

Il tempo è misurato in giorni di borsa / 252, sia per la simulazione sia per le scadenze.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .heston import HestonParams
from . import heston


@dataclass(frozen=True)
class Mondo:
    r: float = 0.04       # tasso privo di rischio
    q: float = 0.013      # dividend yield
    mu: float = 0.10      # rendimento atteso totale del titolo sotto P
    kappa: float = 4.0    # parametri di Heston sotto P
    theta: float = 0.0359
    xi: float = 0.48
    rho: float = -0.7
    lam: float = 2.0      # premio per il rischio di varianza (lam > 0: IV sopra RV)

    def params_P(self, v0):
        return HestonParams(v0, self.kappa, self.theta, self.xi, self.rho)

    def params_Q(self, v0):
        # dv = [kappa theta - (kappa - lam) v] dt sotto Q: kappa_Q = kappa - lam,
        # e kappa_Q theta_Q = kappa theta (xi e rho non cambiano)
        k_q = self.kappa - self.lam
        if k_q <= 0:
            raise ValueError("lam deve essere minore di kappa")
        return HestonParams(v0, k_q, self.kappa * self.theta / k_q, self.xi, self.rho)


def simulate(S0, p, mu, q, n_days, n_paths=1, n_sub=10, rng=None, v0=None):
    """Percorsi giornalieri (S, v) di Heston, schema di Eulero con troncamento della varianza.

    mu è il rendimento atteso totale: il prezzo cresce di (mu - q), il resto sono dividendi.
    Restituisce due array (n_days + 1, n_paths).
    """
    rng = np.random.default_rng() if rng is None else rng
    dt = 1.0 / 252 / n_sub
    v = np.full(n_paths, p.v0) if v0 is None else np.array(v0, dtype=float)
    logS = np.full(n_paths, np.log(S0))
    S = np.empty((n_days + 1, n_paths))
    V = np.empty((n_days + 1, n_paths))
    S[0], V[0] = S0, v
    c = np.sqrt(1 - p.rho**2)
    for t in range(1, n_days + 1):
        z = rng.standard_normal((n_sub, 2, n_paths))
        for j in range(n_sub):
            z1 = z[j, 0]
            z2 = p.rho * z1 + c * z[j, 1]
            vp = np.maximum(v, 0.0)
            logS += (mu - q - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
            v = v + p.kappa * (p.theta - vp) * dt + p.xi * np.sqrt(vp * dt) * z2
        S[t], V[t] = np.exp(logS), v
    return S, V


def simulate_P(mondo, n_days, n_paths=1, S0=100.0, n_sub=10, seed=0):
    """Percorsi nel mondo reale, con varianza iniziale dalla distribuzione stazionaria."""
    rng = np.random.default_rng(seed)
    shape = 2 * mondo.kappa * mondo.theta / mondo.xi**2
    scale = mondo.xi**2 / (2 * mondo.kappa)
    v0 = rng.gamma(shape, scale, n_paths)
    return simulate(S0, mondo.params_P(mondo.theta), mondo.mu, mondo.q,
                    n_days, n_paths, n_sub, rng, v0)


GIORNI_SCADENZA = (21, 42, 63, 126, 252)  # 1, 2, 3, 6, 12 mesi di borsa


def chain(mondo, S, v, giorni=GIORNI_SCADENZA, n_sd=2.5, n_strikes=21, sigma_ref=0.2, tick=None):
    """Listino di opzioni out-of-the-money (put sotto il forward, call sopra) in un giorno.

    I prezzi sono quelli di Heston sotto Q, con la varianza istantanea v di quel giorno.
    Gli strike coprono +/- n_sd deviazioni standard e sono arrotondati all'intero (come in
    un listino vero). Se tick è dato, i prezzi sono arrotondati al tick.
    """
    pq = mondo.params_Q(v)
    righe = []
    for g in giorni:
        T = g / 252
        F = S * np.exp((mondo.r - mondo.q) * T)
        k = np.linspace(-n_sd, n_sd, n_strikes) * sigma_ref * np.sqrt(T)
        for K in np.unique(np.round(F * np.exp(k))):
            kind = "put" if K < F else "call"
            mid = heston.price(S, K, T, mondo.r, mondo.q, pq, kind)
            if tick:
                mid = round(mid / tick) * tick
            righe.append((T, K, kind, mid))
    df = pd.DataFrame(righe, columns=["T", "K", "kind", "mid"])
    df["S"], df["r"], df["q"] = S, mondo.r, mondo.q
    return df
