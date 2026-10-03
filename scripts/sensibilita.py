"""Sensibilità del backtest sintetico (call venduta e coperta in delta) a tre scelte:
il premio per il rischio di varianza lam, la frequenza di copertura e i costi di transazione.
I percorsi sono gli stessi in ogni caso (cambiano solo prezzi e copertura), così le differenze
non dipendono dal caso. P&L in % dello spot, media con intervallo al 95% tra percorsi.

    python scripts/sensibilita.py [--percorsi 300]
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from volsurf import backtest as bt  # noqa: E402
from volsurf import synthetic as sy  # noqa: E402


def riga(df):
    ic = bt.media_con_ic(df)
    return dict(media=ic["media"], ic95=f"[{ic['ic_basso']:.2f}, {ic['ic_alto']:.2f}]",
                dev_std=df["pnl"].std(), in_guadagno=(df["pnl"] > 0).mean(), peggiore=df["pnl"].min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--percorsi", type=int, default=300)
    a = ap.parse_args()
    M = sy.Mondo()
    S, V = sy.simulate_P(M, 1260, n_paths=a.percorsi, seed=7)
    pd.options.display.float_format = "{:.2f}".format
    out = Path("output")
    out.mkdir(exist_ok=True)

    A = pd.DataFrame({(g, lam): riga(bt.backtest(replace(M, lam=lam), S, V, g))
                      for g in (21, 63) for lam in (0, 1, 2, 3)}).T
    A.index.names = ["giorni", "lam"]
    print("A. premio per il rischio di varianza (copertura giornaliera, senza costi)\n", A, "\n")

    B = pd.DataFrame({(g, o): riga(bt.backtest(M, S, V, g, ogni=o))
                      for g in (21, 63) for o in (1, 2, 5, 10)}).T
    B.index.names = ["giorni", "ogni"]
    print("B. frequenza di copertura (lam=2, ribilancio ogni `ogni` giorni)\n", B, "\n")

    C = pd.DataFrame({(g, bp): riga(bt.backtest(M, S, V, g, costo=bp / 1e4))
                      for g in (21, 63) for bp in (0, 1, 2, 5)}).T
    C.index.names = ["giorni", "costo_bp"]
    print("C. costi di transazione (lam=2, copertura giornaliera, bp del controvalore)\n", C)
    for nome, t in (("lam", A), ("frequenza", B), ("costi", C)):
        t.to_csv(out / f"sensibilita_{nome}.csv")


if __name__ == "__main__":
    main()
