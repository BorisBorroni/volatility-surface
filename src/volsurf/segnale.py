"""Segnale volatilità implicita meno realizzata sul mondo sintetico.

Calcolare la IV ATM con un prezzo di Heston per ogni giorno e percorso costerebbe ore
(5 ms a prezzo). Ma la IV ATM dipende solo dalla varianza istantanea v: la calcolo su una
griglia di v e interpolo con una spline cubica.
"""
import numpy as np
from scipy.interpolate import CubicSpline

from . import bs, heston
from . import realized


def iv_atm_esatta(mondo, v, T, S=100.0):
    """IV della call at-the-forward (K = F) con Heston sotto Q e varianza istantanea v."""
    F = S * np.exp((mondo.r - mondo.q) * T)
    pr = heston.price(S, F, T, mondo.r, mondo.q, mondo.params_Q(v), "call")
    return bs.implied_vol(pr, S, F, T, mondo.r, mondo.q, "call")


class IVAtm:
    """IV ATM in funzione di v, a scadenza T fissata (interpolata su griglia)."""

    def __init__(self, mondo, T, v_max, n=40):
        self.v = np.linspace(1e-3, v_max, n)
        iv = [iv_atm_esatta(mondo, x, T) for x in self.v]
        self._spl = CubicSpline(self.v, iv)

    def __call__(self, v):
        v = np.asarray(v, dtype=float)
        if v.min() < self.v[0] or v.max() > self.v[-1]:
            raise ValueError("v fuori dalla griglia")
        return self._spl(v)


def tabella(mondo, S, V, giorni, grid_n=40):
    """Per ogni percorso e giorno t con almeno `giorni` giorni sia prima sia dopo:
       iv      IV ATM a scadenza `giorni`,
       rv_ind  vol realizzata sugli ultimi `giorni` giorni (nota in t),
       rv_av   vol realizzata sui `giorni` giorni successivi (nota solo dopo).
    Restituisce tre array (righe valide, percorsi)."""
    T = giorni / 252
    iv_fun = IVAtm(mondo, T, V.max() * 1.01 + 1e-3, grid_n)
    rv = realized.close_to_close(S, giorni)
    t = np.arange(giorni, len(S) - giorni)
    return iv_fun(np.maximum(V[t], 1e-3)), rv[t], rv[t + giorni]
