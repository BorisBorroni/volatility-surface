"""Controllo: la IV a 30 giorni del fornitore (iv_current) coincide con quella che ricaviamo noi
dalle quote? Su un campione di giorni, dalla chain di DoltHub: per le due scadenze attorno a 30
giorni calendariali si prende la IV americana delle opzioni out-of-the-money più vicine al
forward (put sotto, call sopra, media delle due), e si interpola la varianza totale a 30 giorni.
r e q sono costanti (3% e 1.3%): il controllo cattura le differenze grosse, non quelle fini.

    python scripts/controlla_iv.py [--giorni 60]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from volsurf import americana as am  # noqa: E402
from volsurf.dati import leggi  # noqa: E402

R, Q = 0.03, 0.013


def iv_atm_scadenza(g, S, T):
    """IV ATM di una scadenza: media tra put e call OTM più vicine al forward."""
    F = S * np.exp((R - Q) * T)
    ivs = []
    for kind, lato in (("put", g["strike"] <= F), ("call", g["strike"] >= F)):
        x = g[(g["call_put"].str.lower() == kind) & lato & (g["bid"] > 0) & (g["ask"] >= g["bid"])].copy()
        if x.empty:
            return np.nan
        x = x.iloc[np.argmin(np.abs(x["strike"].values - F))]
        v = am.iv(np.array([(x["bid"] + x["ask"]) / 2]), S, np.array([x["strike"]]), T, R, Q, kind, 150)[0]
        ivs.append(v)
    return np.nanmean(ivs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--giorni", type=int, default=60)
    a = ap.parse_args()
    prezzi = leggi("data/storico_SPY.csv")["Close"]
    vol = leggi("data/dolthub/vol_SPY.csv")["iv_current"].astype(float)
    c = pd.read_csv("data/dolthub/chain_SPY.csv", encoding="utf-8-sig", parse_dates=["date", "expiration"])
    date = sorted(set(c["date"]) & set(prezzi.index) & set(vol.index))
    date = date[::max(1, len(date) // a.giorni)]
    righe = []
    for d in date:
        g = c[c["date"] == d]
        giorni = (g["expiration"] - d).dt.days
        sc = sorted(set(giorni[(giorni >= 7)]))
        sotto = [x for x in sc if x <= 30][-1:]
        sopra = [x for x in sc if x > 30][:1]
        if not sotto or not sopra:
            continue
        w = []
        for x in sotto + sopra:
            iv = iv_atm_scadenza(g[giorni == x], prezzi[d], x / 365)
            w.append((x, iv * iv * x / 365))
        if any(np.isnan(v) for _, v in w):
            continue
        (t1, w1), (t2, w2) = w
        wt = w1 + (w2 - w1) * (30 - t1) / (t2 - t1)
        righe.append(dict(data=d, nostra=np.sqrt(wt / (30 / 365)), fornitore=vol[d], t1=t1, t2=t2))
    r = pd.DataFrame(righe)
    r["diff"] = (r["nostra"] - r["fornitore"]) * 100
    print(f"{len(r)} giorni confrontati ({r['data'].min().date()} -> {r['data'].max().date()})")
    print(f"differenza nostra - fornitore (punti di vol): media {r['diff'].mean():.2f}, mediana "
          f"{r['diff'].median():.2f}, dev std {r['diff'].std():.2f}, |diff|>1 punto in "
          f"{(r['diff'].abs() > 1).mean():.0%} dei giorni; correlazione {r['nostra'].corr(r['fornitore']):.3f}")
    print(r.groupby(r["data"].dt.year)["diff"].agg(["count", "mean", "std"]).round(2).to_string())
    Path("output").mkdir(exist_ok=True)
    r.to_csv("output/controllo_iv.csv", index=False)


if __name__ == "__main__":
    main()
