"""Backtest su SPY con la IV di mercato storica, e premio IV - vol realizzata.

Backtest: in ogni giorno con una IV disponibile si vende una call at-the-forward a 21 giorni di
borsa, a prezzo Black-Scholes con la IV a 30 giorni di quel giorno (iv_current di DoltHub), e la
si copre in delta fino a scadenza. r e q sono costanti. Gli ingressi consecutivi si sovrappongono
per ~21 giorni, quindi gli errori standard sono Newey-West (stat.py).

Servono data/dolthub/vol_SPY.csv, data/storico_SPY.csv e data/vix.csv
(scarica_dati.py --solo-storico --anni 8; il file di Dolt è descritto nel README).

    python scripts/backtest_storico.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # le cartelle data/ e output/ sono quelle del progetto
sys.path.insert(0, str(ROOT / "src"))
from volsurf import backtest as bt  # noqa: E402
from volsurf import realized, stat  # noqa: E402
from volsurf.dati import leggi  # noqa: E402

GIORNI, R, Q = 21, 0.03, 0.013
LAG = 21


def riga(df):
    m, se, t = stat.media_hac(df["pnl"], LAG)
    return dict(n=len(df), media=m, se_NW=se, t=t, dev_std=df["pnl"].std(),
                in_guadagno=(df["pnl"] > 0).mean(), iv=df["iv"].mean() * 100, rv=df["rv_ind"].mean() * 100)


def premio(nome, iv, rv_futura):
    """IV - vol realizzata nei 21 giorni successivi, in punti di vol, con errore Newey-West."""
    x = (iv - rv_futura).dropna() * 100
    m, se, t = stat.media_hac(x, LAG)
    print(f"  {nome:<26} media {m:+.2f}  (se {se:.2f}, t {t:+.1f}, positivo {(x > 0).mean():.0%} dei giorni)")


def main():
    ohlc = leggi(ROOT / "data/storico_SPY.csv")
    prezzi = ohlc["Close"]
    iv = leggi(ROOT / "data/dolthub/vol_SPY.csv")["iv_current"].astype(float)
    iv = iv[(iv > 0.03) & (iv < 3)].reindex(prezzi.index)            # solo giorni di borsa con IV valida
    vix = leggi(ROOT / "data/vix.csv")["Close"].reindex(prezzi.index) / 100
    print(f"SPY: {prezzi.index[0].date()} -> {prezzi.index[-1].date()}, {len(prezzi)} giorni, "
          f"{iv.notna().sum()} con IV\n")

    rv = pd.Series(realized.close_to_close(prezzi.values, GIORNI), index=prezzi.index)
    yz = realized.yang_zhang(ohlc.rename(columns=str.lower), GIORNI)
    print("Premio sulla vol realizzata dei 21 giorni successivi (punti di vol):")
    # IV e VIX sono a 30 giorni di calendario (base 365), la RV è in giorni di borsa (base 252):
    # prima di confrontarle porto la IV alla base 252 (fattore 0.9931)
    c = realized.iv_in_tempo_di_borsa
    premio("VIX - RV (close-to-close)", c(vix, GIORNI), rv.shift(-GIORNI))
    premio("VIX - RV (Yang-Zhang)", c(vix, GIORNI), yz.shift(-GIORNI))
    premio("IV 30gg - RV (close-to-close)", c(iv, GIORNI), rv.shift(-GIORNI))
    premio("VIX - RV, solo giorni con IV", c(vix, GIORNI).where(iv.notna()), rv.shift(-GIORNI))
    r = (iv / vix).dropna()
    print(f"  rapporto IV / VIX: mediana {r.median():.2f}, percentili 10-90: "
          f"{r.quantile(.1):.2f}-{r.quantile(.9):.2f}\n")

    iv252 = c(iv, GIORNI)
    df = bt.backtest_reale(prezzi.values, iv252.values, GIORNI, R, Q)
    df["anno"] = prezzi.index[df["t0"].values].year
    tab = pd.DataFrame({"tutto": riga(df)} | {a: riga(g) for a, g in df.groupby("anno") if len(g) > 30}).T
    pd.options.display.float_format = "{:.2f}".format
    print("Backtest, P&L per operazione (% dello spot), senza costi:\n", tab.to_string())
    con_costo = bt.backtest_reale(prezzi.values, iv252.values, GIORNI, R, Q, costo=1e-4)
    print("\ncon 1 punto base di costo:\n", pd.DataFrame({"tutto": riga(con_costo)}).T.to_string())

    # la nostra IV (controlla_iv.py) sta circa 0.7 punti sotto quella del fornitore: quanto pesa?
    print("\nSensibilità a r e q costanti:")
    for r, q in ((0.0, Q), (0.05, Q), (R, 0.0), (R, 0.02)):
        d = bt.backtest_reale(prezzi.values, iv252.values, GIORNI, r, q)
        m, se, t = stat.media_hac(d["pnl"], LAG)
        print(f"  r {r:.0%}, q {q:.1%}: P&L medio {m:+.3f}%  (se {se:.3f}, t {t:+.2f})")
    print("\nSensibilità: IV del fornitore spostata di x punti di vol (prima della conversione)")
    for sh in (0, -0.5, -0.73, -1.0):
        d = bt.backtest_reale(prezzi.values, c(iv + sh / 100, GIORNI).values, GIORNI, R, Q)
        m, se, t = stat.media_hac(d["pnl"], LAG)
        print(f"  {sh:+.2f} punti: P&L medio {m:+.3f}%  (se {se:.3f}, t {t:+.2f})")
    (ROOT / "output").mkdir(exist_ok=True)
    df.to_csv(ROOT / "output/backtest_storico.csv", index=False)


if __name__ == "__main__":
    main()
