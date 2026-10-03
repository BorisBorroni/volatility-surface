"""Volatilità realizzata, annualizzata con 252 giorni di borsa.

close_to_close: usa solo le chiusure (è l'unica possibile sui dati sintetici).
yang_zhang: usa apertura, massimo, minimo e chiusura (per i dati reali di SPY).
"""
import numpy as np

GIORNI_ANNO = 252


def iv_in_tempo_di_borsa(iv, giorni, giorni_cal=30):
    """Porta una IV a `giorni_cal` giorni di calendario (annualizzata con 365) alla base dei
    giorni di borsa (252), per poterla confrontare con la vol realizzata e usarla nel backtest.

    La varianza totale di mercato è iv^2 * giorni_cal / 365; quella della nostra convenzione è
    sigma^2 * giorni / 252. Per 21 giorni di borsa vs 30 di calendario il fattore è 0.9931."""
    return iv * np.sqrt((giorni_cal / 365) / (giorni / GIORNI_ANNO))


def close_to_close(S, n):
    """Vol realizzata sugli ultimi n rendimenti logaritmici: sqrt(252 * media(r^2)).

    S: prezzi di chiusura, array (giorni + 1) oppure (giorni + 1, percorsi).
    Restituisce un array con la stessa forma di S; le prime n righe sono NaN, la riga t
    usa i rendimenti dei giorni t-n+1, ..., t.
    """
    S = np.asarray(S, dtype=float)
    r2 = np.diff(np.log(S), axis=0) ** 2
    c = np.concatenate([np.zeros((1,) + r2.shape[1:]), np.cumsum(r2, axis=0)])
    out = np.full(S.shape, np.nan)
    out[n:] = np.sqrt(GIORNI_ANNO * (c[n:] - c[:-n]) / n)
    return out


def yang_zhang(ohlc, n):
    """Stimatore di Yang-Zhang su finestra mobile di n giorni.

    ohlc: DataFrame con colonne open, high, low, close. Combina tre pezzi di varianza:
    il salto notturno (chiusura -> apertura), il movimento apertura -> chiusura e quello
    di Rogers-Satchell (usa massimo e minimo), con il peso k che minimizza la varianza
    dello stimatore.
    """
    o, h, lo, c = (np.log(ohlc[x]) for x in ("open", "high", "low", "close"))
    notte = o - c.shift(1)
    giorno = c - o
    rs = (h - c) * (h - o) + (lo - c) * (lo - o)
    var_n = notte.rolling(n).var()      # ddof=1
    var_g = giorno.rolling(n).var()
    var_rs = rs.rolling(n).mean()
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return np.sqrt(GIORNI_ANNO * (var_n + k * var_g + (1 - k) * var_rs))
