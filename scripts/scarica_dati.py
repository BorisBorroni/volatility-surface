"""Scarica da Yahoo Finance la option chain di un indice/ETF, lo storico OHLC e il VIX,
e dal sito della FRED la curva dei tassi. Lo storico e il VIX vanno in data/; la chain e la curva
(i dati "correnti", non versionati) in data/corrente/, sovrascritti a ogni scaricamento.

Uso (dalla cartella del progetto):
    python scripts/scarica_dati.py --solo-chain    # chain e tassi dell'ultima seduta (analisi corrente)
    python scripts/scarica_dati.py --solo-storico --anni 8   # aggiorna prezzi e VIX
    python scripts/scarica_dati.py                 # tutte e due le cose
    python scripts/scarica_dati.py --ticker QQQ --storico QQQ
"""
import argparse
import io
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

COLONNE = ["strike", "bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility",
           "lastTradeDate"]


def nome(ticker):
    return ticker.replace("^", "")


def tasso_fred(serie="DGS3MO"):
    """Ultimo valore disponibile (in %) della serie FRED, oppure None se non raggiungibile."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={serie}"
    try:
        with urllib.request.urlopen(url, timeout=30) as f:
            df = pd.read_csv(io.StringIO(f.read().decode()))
    except Exception as e:
        print("FRED non raggiungibile:", e)
        return None
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna()
    return float(s.iloc[-1])


CURVA = {"DGS1MO": 1 / 12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1.0, "DGS2": 2.0}


def curva_fred():
    """Ultimo valore disponibile di ogni punto della curva dei Treasury (T in anni, r in %)."""
    righe = []
    for serie, T in CURVA.items():
        r = tasso_fred(serie)
        if r is not None:
            righe.append((T, r, serie))
    return pd.DataFrame(righe, columns=["T", "r_percento", "serie"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY", help="sottostante delle opzioni (americane)")
    ap.add_argument("--storico", default="SPY", help="sottostante per lo storico OHLC")
    ap.add_argument("--anni", type=int, default=8)
    ap.add_argument("--r", type=float, default=None, help="tasso in %%, se FRED non funziona")
    ap.add_argument("--solo-storico", action="store_true", help="scarica solo storico e VIX, non la chain")
    ap.add_argument("--solo-chain", action="store_true", help="scarica solo chain e tassi, non lo storico")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    # storico OHLC (serve per la vol realizzata) e VIX (serve per un controllo)
    periodo = "5d" if a.solo_chain else f"{a.anni}y"      # con --solo-chain serve solo l'ultimo prezzo
    st = yf.Ticker(a.storico).history(period=periodo, auto_adjust=False)
    st.index = st.index.tz_localize(None).normalize()
    if not a.solo_chain:
        st[["Open", "High", "Low", "Close", "Volume"]].to_csv(out / f"storico_{nome(a.storico)}.csv")
        vix = yf.Ticker("^VIX").history(period=periodo)
        vix.index = vix.index.tz_localize(None).normalize()
        vix[["Close"]].to_csv(out / "vix.csv")
    if a.solo_storico:
        print(f"storico {a.storico}: {st.index[0].date()} -> {st.index[-1].date()}, {len(st)} giorni")
        return
    corr = out / "corrente"
    corr.mkdir(exist_ok=True)
    data_rif = st.index[-1].date()       # giorno dell'ultima chiusura: le quotazioni sono di quel giorno
    spot = float(st["Close"].iloc[-1])

    t = yf.Ticker(a.ticker)
    righe = []
    for exp in t.options:
        try:
            oc = t.option_chain(exp)
        except Exception as e:
            print("scadenza saltata", exp, e)
            continue
        for kind, d in (("call", oc.calls), ("put", oc.puts)):
            d = d[COLONNE].copy()
            d.insert(0, "kind", kind)
            d.insert(0, "scadenza", exp)
            righe.append(d)
        time.sleep(0.3)
    chain = pd.concat(righe, ignore_index=True)
    chain.to_csv(corr / f"chain_{nome(a.ticker)}.csv", index=False)

    r = a.r if a.r is not None else tasso_fred()
    curva = curva_fred()
    if len(curva):
        curva.to_csv(corr / "curva_tassi.csv", index=False)
    meta = dict(ticker=a.ticker, storico=a.storico, data_rif=str(data_rif), spot_storico=spot,
                r_percento=r, scaricato_il=datetime.now().isoformat(timespec="seconds"),
                n_scadenze=len(t.options), n_quote=len(chain))
    (corr / f"meta_{nome(a.ticker)}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
