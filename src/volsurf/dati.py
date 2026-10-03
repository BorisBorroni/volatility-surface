"""Lettura delle serie giornaliere salvate in data/ (indice = data, senza fuso orario)."""
import pandas as pd


def leggi(percorso):
    d = pd.read_csv(percorso, index_col=0, encoding="utf-8-sig")   # i CSV di Dolt hanno il BOM
    d.index = pd.to_datetime(d.index, utc=True).tz_localize(None).normalize()
    return d
