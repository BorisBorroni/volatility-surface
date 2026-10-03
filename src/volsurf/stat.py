"""Errori standard robusti all'autocorrelazione (Newey-West).

Con finestre sovrapposte (la vol realizzata sui prossimi 21 giorni, calcolata ogni giorno) i
dati consecutivi condividono 20 giorni su 21: l'errore standard classico sbaglia di molto.
"""
import numpy as np


def _omega(u, lags):
    """Varianza di lungo periodo di una serie a media zero, con pesi di Bartlett."""
    n = len(u)
    om = u @ u / n
    for lag in range(1, lags + 1):
        om += 2 * (1 - lag / (lags + 1)) * (u[lag:] @ u[:-lag]) / n
    return om


def media_hac(x, lags):
    """Media, errore standard Newey-West e statistica t di H0: media = 0."""
    x = np.asarray(x, dtype=float)
    se = np.sqrt(_omega(x - x.mean(), lags) / len(x))
    return x.mean(), se, x.mean() / se
