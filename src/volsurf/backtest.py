"""Backtest sul mondo sintetico: ogni `giorni` giorni si vende una call at-the-forward e la si
copre in delta fino a scadenza (vedi hedge.py). Il prezzo di vendita è quello di mercato, cioè
la IV ATM del giorno (calcolata con Heston sotto Q); i percorsi sono quelli reali (sotto P)."""
import numpy as np
import pandas as pd

from . import hedge, realized, segnale


def backtest(mondo, S, V, giorni, ogni=1, costo=0.0):
    """Una riga per ogni (percorso, ingresso), con:
       iv, rv_ind (vol realizzata degli ultimi `giorni` giorni, nota all'ingresso),
       spread = iv - rv_ind, pnl (in % dello spot), teoria (somma gamma, in % dello spot)."""
    T = giorni / 252
    iv_fun = segnale.IVAtm(mondo, T, V.max() * 1.01 + 1e-3)
    rv = realized.close_to_close(S, giorni)
    blocchi = []
    for t0 in range(giorni, len(S) - giorni, giorni):
        iv = iv_fun(np.maximum(V[t0], 1e-3))
        K = S[t0] * np.exp((mondo.r - mondo.q) * T)
        pnl, teoria = hedge.call_venduta_coperta(S[t0:t0 + giorni + 1], K, mondo.r, mondo.q, iv, ogni, costo)
        blocchi.append(pd.DataFrame({
            "percorso": np.arange(S.shape[1]), "t0": t0, "iv": iv, "rv_ind": rv[t0],
            "spread": iv - rv[t0], "pnl": pnl / S[t0] * 100, "teoria": teoria / S[t0] * 100}))
    return pd.concat(blocchi, ignore_index=True)


def riassunto(df):
    """Media, deviazione standard, % di operazioni in guadagno e caso peggiore del P&L."""
    p = df["pnl"]
    return pd.Series({"n": len(p), "media": p.mean(), "dev_std": p.std(),
                      "in_guadagno": (p > 0).mean(), "peggiore": p.min()})


def media_con_ic(df):
    """P&L medio (% dello spot) con intervallo di confidenza al 95%. Gli ingressi dello stesso
    percorso sono legati tra loro (la varianza persiste), quindi si fa prima la media per
    percorso e l'errore standard si calcola tra percorsi, che sono indipendenti."""
    m = df.groupby("percorso")["pnl"].mean()
    se = m.std() / np.sqrt(len(m))
    return pd.Series({"media": m.mean(), "ic_basso": m.mean() - 1.96 * se, "ic_alto": m.mean() + 1.96 * se})


def backtest_reale(S, iv, giorni, r, q, ogni=1, costo=0.0):
    """Stessa operazione sui prezzi veri: ogni giorno si vende una call at-the-forward a
    `giorni` giorni di borsa, a prezzo BS con volatilità iv[t] (proxy della IV di mercato:
    non abbiamo le chain storiche) e la si copre. S: chiusure, iv: serie allineata (decimali).
    I giorni con iv nan non sono ingressi (ma i prezzi vanno dati tutti i giorni, per la copertura).
    Una riga per ingresso: iv, rv_ind (vol realizzata passata), pnl e teoria in % dello spot.
    Gli ingressi consecutivi si sovrappongono: per l'inferenza serve Newey-West (stat.py)."""
    S, iv = np.asarray(S, float), np.asarray(iv, float)
    rv = realized.close_to_close(S, giorni)
    T = giorni / 252
    righe = []
    for t0 in range(giorni, len(S) - giorni):
        if not np.isfinite(iv[t0]):                  # giorno senza IV di mercato: nessun ingresso
            continue
        K = S[t0] * np.exp((r - q) * T)
        pnl, teoria = hedge.call_venduta_coperta(S[t0:t0 + giorni + 1], K, r, q, iv[t0], ogni, costo)
        righe.append((t0, iv[t0], rv[t0], pnl / S[t0] * 100, teoria / S[t0] * 100))
    return pd.DataFrame(righe, columns=["t0", "iv", "rv_ind", "pnl", "teoria"])
