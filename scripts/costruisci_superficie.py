"""Dalla chain scaricata (data/corrente/, vedi scarica_dati.py) alla superficie di volatilità
implicita, in un solo passo. Se la chain non c'è ancora la scarica da sola (serve la rete).

    python scripts/costruisci_superficie.py                  # SPY

Passi: pulizia delle quote, q implicito e forward per scadenza (opzioni americane, curva dei
tassi), filtro di moneyness, IV americana, rimozione degli outlier, fit SVI per scadenza.
Scrive in output/ la chain finale, i parametri SVI e il riepilogo di cosa si perde a ogni passo.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # le cartelle data/ e output/ sono quelle del progetto
sys.path.insert(0, str(ROOT / "src"))
from volsurf import chain, real, svi  # noqa: E402

DATA, OUT = ROOT / "data/corrente", ROOT / "output"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    a = ap.parse_args()

    if not (DATA / f"chain_{a.ticker}.csv").exists():             # prima esecuzione: scarico la chain
        print("Chain non trovata: la scarico (serve la rete)")
        scarica = subprocess.run([sys.executable, str(ROOT / "scripts/scarica_dati.py"), "--solo-chain",
                                  "--ticker", a.ticker], check=False)
        if scarica.returncode != 0 or not (DATA / f"chain_{a.ticker}.csv").exists():
            sys.exit("Download non riuscito: controlla la connessione e riprova")
    raw = pd.read_csv(DATA / f"chain_{a.ticker}.csv")
    meta = pd.read_json(DATA / f"meta_{a.ticker}.json", typ="series")
    cv = pd.read_csv(DATA / "curva_tassi.csv")
    a.data_rif = str(meta["data_rif"])[:10]               # giorno delle quotazioni
    S = float(meta["spot_storico"])
    r = real.curva_tassi(cv["T"], cv["r_percento"] / 100)
    OUT.mkdir(exist_ok=True)

    passi = [("grezza", raw.assign(T=(pd.to_datetime(raw["scadenza"]) - pd.Timestamp(a.data_rif)).dt.days / 365))]
    passi.append(("quote valide", real.pulisci(raw, a.data_rif)))
    try:
        d = real.costruisci_chain(raw, S, r, a.data_rif)             # q americano, un lato per strike
    except ValueError as e:
        sys.exit(f"{e}. Yahoo non garantisce le quote fuori orario: riprova con il mercato aperto.")
    passi.append(("q stimabile, OTM", d))
    d = real.filtra_moneyness(d)
    passi.append(("moneyness", d))
    d = chain.add_iv_americana(d)
    d = real.rimuovi_outlier(d)
    passi.append(("outlier", d))
    fits = svi.fit_chain(d)
    passi.append(("fit SVI (>=5 quote)", d[d["T"].isin(fits.index)]))

    rip = real.riepilogo(passi)
    print(f"{a.ticker} {a.data_rif}, spot {S:.2f}\n")
    print(rip.to_string(index=False))
    print("\nscadenze senza q:", [round(t * 365) for t in d.attrs.get("scadenze_senza_q", [])], "giorni")
    ok = fits.assign(giorni=(fits.index * 365).round())
    print("\n", ok[["giorni", "n", "rmse_iv", "g_dati", "al_limite"]].round(4).to_string(index=False))
    if len(fits) > 1:
        print(f"\ncontrollo calendar: min w(T_j+1) - w(T_j) = {svi.calendar_min(fits):.5f} (>= 0: nessun arbitraggio)")
    d.to_csv(OUT / f"chain_finale_{a.ticker}.csv", index=False)
    fits.to_csv(OUT / f"fit_svi_{a.ticker}.csv")
    rip.to_csv(OUT / f"riepilogo_pulizia_{a.ticker}.csv", index=False)


if __name__ == "__main__":
    main()
