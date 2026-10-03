"""Operazioni su una chain di opzioni, con colonne T, K, kind, mid, S, r, q.

Lo stesso formato vale per la chain sintetica e per quella reale.
"""
import numpy as np

from . import americana, bs


def add_iv(df):
    out = df.copy()
    out["iv"] = bs.implied_vol_array(out["mid"].values, out["S"].values, out["K"].values,
                                     out["T"].values, out["r"].values, out["q"].values,
                                     out["kind"].values)
    return out


def add_iv_americana(df, N=americana.N_PASSI):
    """Come add_iv, ma con prezzo americano (albero binomiale). r e q sono costanti per scadenza."""
    out = df.copy()
    out["iv"] = np.nan
    for T, g in out.groupby("T"):
        out.loc[g.index, "iv"] = americana.iv(g["mid"].values, g["S"].iloc[0], g["K"].values, T,
                                              g["r"].iloc[0], g["q"].iloc[0], g["kind"].values, N)
    return out
