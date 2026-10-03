"""Dalla chain grezza scaricata (una riga per contratto) a una chain pulita nel formato del
progetto (colonne T, K, kind, mid, S, r, q, ...), pronta per chain.add_iv e svi.fit_chain.

Passi: scarto delle quote inutilizzabili, prezzo = mid, dividend yield (e quindi forward) per
ogni scadenza dal confronto tra IV americana di call e put, solo opzioni out-of-the-money.
Il tempo delle scadenze è in giorni di calendario / 365 (la vol realizzata invece usa 252).
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import americana as am


def pulisci(raw, data_rif, T_min=14 / 365, T_max=2.0, spread_max=0.5, mid_min=0.10):
    """raw: colonne scadenza, kind, strike, bid, ask. Restituisce le quote valide con T e mid."""
    d = raw.copy()
    d["T"] = (pd.to_datetime(d["scadenza"]) - pd.Timestamp(data_rif)).dt.days / 365
    d = d[(d["T"] >= T_min) & (d["T"] <= T_max)]
    d = d[(d["bid"] > 0) & (d["ask"] >= d["bid"])]          # niente quote a zero né incrociate
    d["mid"] = (d["bid"] + d["ask"]) / 2
    d["spread_rel"] = (d["ask"] - d["bid"]) / d["mid"]
    d = d[(d["spread_rel"] <= spread_max) & (d["mid"] >= mid_min)]
    d = d.drop_duplicates(["scadenza", "kind", "strike"])
    return d.rename(columns={"strike": "K"}).reset_index(drop=True)


def forward_parity(d, S, r, n_strike=5):
    """Forward per scadenza dalla put-call parity: C - P = e^{-rT} (F - K), cioè
    F = K + e^{rT} (C - P). Si usano gli n_strike strike più vicini allo spot che hanno sia
    la call sia la put, e si prende la mediana. Restituisce una Series indicizzata per T."""
    out = {}
    for T, g in d.groupby("T"):
        c = g[g["kind"] == "call"].set_index("K")["mid"]
        p = g[g["kind"] == "put"].set_index("K")["mid"]
        K = c.index.intersection(p.index)
        if len(K) == 0:
            continue
        K = K[np.argsort(np.abs(K.values - S))[:n_strike]]
        out[T] = np.median(K.values + np.exp(r * T) * (c[K].values - p[K].values))
    return pd.Series(out, name="F")


def stima_q(g, S, r, T, N=150, n_strike=5):
    """Dividend yield implicito di una scadenza con opzioni americane.

    Con esercizio anticipato la put-call parity vale solo come disuguaglianza, quindi non si
    può ricavare il forward da C - P. Si cerca invece il q per cui, sugli strike vicini allo
    spot, la IV americana della call e quella della put coincidono (se q è sbagliato le due
    curve si separano). Restituisce nan se non si trova.
    N=150 basta: su SPY il q cambia meno di 1e-5 passando da N=150 a N=300 (e costa metà).
    """
    c = g[g["kind"] == "call"].set_index("K")["mid"]
    p = g[g["kind"] == "put"].set_index("K")["mid"]
    K = c.index.intersection(p.index)
    if len(K) == 0:
        return np.nan
    K = np.sort(K.values[np.argsort(np.abs(K.values - S))[:n_strike]])
    pc, pp = c[K].values, p[K].values
    q0 = r - np.log(forward_parity(g, S, r, n_strike).iloc[0] / S) / T   # punto di partenza europeo

    def scarto(q):
        d = am.iv(pc, S, K, T, r, q, "call", N, tol=1e-6) - am.iv(pp, S, K, T, r, q, "put", N, tol=1e-6)
        return np.nanmedian(d) if np.isfinite(d).any() else np.nan

    try:
        return brentq(lambda q: scarto(q), q0 - 0.05, q0 + 0.05, xtol=1e-6)
    except ValueError:                      # nessun cambio di segno, o scarto nan
        return np.nan


def curva_tassi(T, r):
    """Tasso per scadenza: interpolazione lineare tra i punti (T in anni, r decimale), piatta
    fuori dall'intervallo. Si passa a costruisci_chain al posto di un r unico."""
    T, r = np.asarray(T, float), np.asarray(r, float)
    return lambda t: float(np.interp(t, T, r))


def costruisci_chain(raw, S, r, data_rif, N=150, **kw):
    """Chain pulita con una sola opzione per strike (put sotto il forward, call sopra) e,
    per ogni scadenza, q americano e forward F = S e^{(r-q)T}. La colonna q_eu è il q che si
    otterrebbe con la parity europea (solo per confronto). r è un numero oppure una funzione
    della scadenza (vedi curva_tassi)."""
    d = pulisci(raw, data_rif, **kw)
    blocchi, senza_q = [], []
    for T, g in d.groupby("T"):
        rT = r(T) if callable(r) else r
        q = stima_q(g, S, rT, T, N)
        if not np.isfinite(q):
            senza_q.append(T)
            continue
        F = S * np.exp((rT - q) * T)
        g = g[((g["kind"] == "put") & (g["K"] < F)) | ((g["kind"] == "call") & (g["K"] >= F))].copy()
        g["F"], g["q"], g["r"] = F, q, rT
        g["q_eu"] = rT - np.log(forward_parity(d[d["T"] == T], S, rT).iloc[0] / S) / T
        blocchi.append(g)
    if not blocchi:
        raise ValueError("nessuna scadenza con q stimabile (servono call e put sugli stessi strike)")
    d = pd.concat(blocchi)
    d["S"] = S
    cols = ["T", "K", "kind", "mid", "S", "r", "q", "F", "q_eu", "spread_rel"]
    d = d[cols].sort_values(["T", "K"]).reset_index(drop=True)
    d.attrs["scadenze_senza_q"] = senza_q            # scadenze scartate perché q non si trova
    return d


def riepilogo(passi):
    """Quote e scadenze rimaste dopo ogni passo della pulizia, per vedere cosa si perde.
    passi: lista di (nome, DataFrame con colonna T)."""
    righe, prima = [], None
    for nome, d in passi:
        T = set(np.round(d["T"], 8))
        righe.append(dict(passo=nome, quote=len(d), scadenze=len(T),
                          scadenze_perse=0 if prima is None else len(prima - T)))
        prima = T
    return pd.DataFrame(righe)


def filtra_moneyness(df, c=0.6):
    """Tiene gli strike con |ln(K/F)| <= c * sqrt(T): una fascia che si allarga con la scadenza.
    Le code lontanissime hanno prezzi quasi nulli e IV poco informative, e deformano il fit."""
    k = np.log(df["K"] / df["F"])
    return df[k.abs() <= c * np.sqrt(df["T"])].reset_index(drop=True)


def rimuovi_outlier(df, passaggi=15, n_mad=4.0, soglia_min=0.005, vicini=2):
    """Toglie le quote isolate la cui IV si scosta da quella dei vicini nella stessa scadenza.

    Gli smile sono lisci, quindi la IV di uno strike deve stare vicino alla retta che passa per
    due vicini, uno a sinistra e uno a destra (strike non equispaziati: si interpola in K). Si
    fa per ogni coppia tra i `vicini` a sinistra e a destra e si prende la mediana delle
    previsioni, così un vicino sbagliato conta poco. Per scadenza si scarta la sola quota con lo
    scarto peggiore, se supera max(n_mad * scarto robusto (MAD), soglia_min), e si ripete (un
    outlier fa sembrare sbagliati anche i vicini, per questo si toglie una quota alla volta).
    Le prime e ultime `vicini` quote di ogni scadenza non si controllano. Richiede la colonna iv.
    """
    d = df.dropna(subset=["iv"]).copy()
    for _ in range(passaggi):
        scarta = []
        for _, g in d.groupby("T"):
            g = g.sort_values("K")
            iv, K = g["iv"].values, g["K"].values
            n = len(iv)
            if n < 2 * vicini + 1:
                continue
            res = np.full(n, np.nan)
            for i in range(vicini, n - vicini):
                prev = [iv[a] + (iv[b] - iv[a]) * (K[i] - K[a]) / (K[b] - K[a])
                        for a in range(i - vicini, i) for b in range(i + 1, i + 1 + vicini)]
                res[i] = iv[i] - np.median(prev)
            ok = ~np.isnan(res)
            mad = np.median(np.abs(res[ok] - np.median(res[ok])))
            soglia = max(n_mad * 1.4826 * mad, soglia_min)
            peggiore = np.nanargmax(np.abs(res))
            if abs(res[peggiore]) > soglia:
                scarta.append(g.index[peggiore])
        if not scarta:
            break
        d = d.drop(index=scarta)
    return d.reset_index(drop=True)
