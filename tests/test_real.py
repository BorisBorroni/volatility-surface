import functools

import numpy as np
import pandas as pd
import pytest

from volsurf import americana as am
from volsurf import heston, real
from volsurf import chain as ch
from volsurf import synthetic as sy

pytestmark = pytest.mark.slow

M = sy.Mondo()
S0, V0 = 5000.0, 0.04
DATA = "2025-01-02"
GIORNI = (10, 30, 60, 120, 900)       # 10 giorni: troppo corta; 900 giorni: oltre i 2 anni
Q_VERO = 0.013


@functools.lru_cache(maxsize=4)
def _raw_finta(semi_spread, junk):
    """Chain grezza con call e put a ogni strike, prezzi Heston (Q), spread simmetrico."""
    pq = M.params_Q(V0)
    righe = []
    for g in GIORNI:
        T = g / 365
        F = S0 * np.exp((M.r - Q_VERO) * T)
        scad = (pd.Timestamp(DATA) + pd.Timedelta(days=g)).strftime("%Y-%m-%d")
        for K in np.arange(round(F * 0.9, -1), F * 1.1, 25.0):
            for kind in ("call", "put"):
                mid = heston.price(S0, K, T, M.r, Q_VERO, pq, kind)
                righe.append((scad, kind, K, mid - semi_spread, mid + semi_spread))
    raw = pd.DataFrame(righe, columns=["scadenza", "kind", "strike", "bid", "ask"])
    if junk:
        scad60 = (pd.Timestamp(DATA) + pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        junk_rows = raw[raw["scadenza"] == scad60].iloc[[1, 4, 7, 10]].copy()   # scadenza valida
        junk_rows.loc[junk_rows.index[0], ["bid", "ask"]] = [0.0, 0.5]       # bid nullo
        junk_rows.loc[junk_rows.index[1], ["bid", "ask"]] = [2.0, 1.0]       # incrociata
        junk_rows.loc[junk_rows.index[2], ["bid", "ask"]] = [0.2, 1.0]       # spread troppo largo
        junk_rows.loc[junk_rows.index[3], ["bid", "ask"]] = [0.01, 0.02]     # prezzo minuscolo
        junk_rows["strike"] += 1.0                                           # strike nuovi
        raw = pd.concat([raw, junk_rows], ignore_index=True)
    return raw


def raw_finta(semi_spread=0.05, junk=True):
    return _raw_finta(semi_spread, junk).copy()      # copia: i test modificano i dati


def test_pulisci_scarta_il_cattivo_e_calcola_T():
    d = real.pulisci(raw_finta(), DATA)
    assert set(np.round(d["T"] * 365)) == {30, 60, 120}      # 10 e 900 giorni fuori dai limiti
    assert (d["bid"] > 0).all() and (d["ask"] >= d["bid"]).all()
    assert (d["spread_rel"] <= 0.5).all() and (d["mid"] >= 0.10).all()
    assert len(d) == len(real.pulisci(raw_finta(junk=False), DATA))   # le 4 righe false sono sparite
    assert d["T"].iloc[0] == pytest.approx(round(d["T"].iloc[0] * 365) / 365)     # giorni / 365


def test_pulisci_elimina_i_doppioni():
    raw = raw_finta(junk=False)
    sessanta = raw[raw["scadenza"] == (pd.Timestamp(DATA) + pd.Timedelta(days=60)).strftime("%Y-%m-%d")]
    d = real.pulisci(pd.concat([raw, sessanta.iloc[:8]]), DATA)
    assert not d.duplicated(["scadenza", "kind", "K"]).any()


def test_forward_da_parity():
    d = real.pulisci(raw_finta(), DATA)
    F = real.forward_parity(d, S0, M.r)
    for T, f in F.items():
        assert f == pytest.approx(S0 * np.exp((M.r - Q_VERO) * T), abs=0.05)


def test_forward_robusto_a_una_quota_sbagliata():
    d = real.pulisci(raw_finta(junk=False), DATA)
    giusto = real.forward_parity(d, S0, M.r)
    T = giusto.index[0]
    calls = d[(d["T"] == T) & (d["kind"] == "call")]
    atm = (calls["K"] - S0).abs().idxmin()
    d.loc[atm, "mid"] += 15.0                                # una call ATM con prezzo errato
    sbagliato = real.forward_parity(d, S0, M.r)
    assert sbagliato[T] == pytest.approx(giusto[T], abs=0.6)   # la mediana non si sposta
    assert (sbagliato.drop(T) == giusto.drop(T)).all()


SMILE = dict(a=0.20, b=-0.20, c=0.10)       # sigma(k) = a + b k + c k^2, k = ln(K/F)


def sigma_vera(k):
    return SMILE["a"] + SMILE["b"] * k + SMILE["c"] * k**2


def raw_americana(semi_spread=0.01):
    """Chain grezza americana: call e put prezzate con l'albero (q vero, smile noto)."""
    righe = []
    for g in (30, 60, 120, 250):
        T = g / 365
        F = S0A * np.exp((R_A - Q_VERO) * T)
        scad = (pd.Timestamp(DATA) + pd.Timedelta(days=g)).strftime("%Y-%m-%d")
        K = np.arange(round(F * 0.85), F * 1.15, 2.5)
        sig = sigma_vera(np.log(K / F))
        for kind in ("call", "put"):
            for k_, m in zip(K, am.prezzo(S0A, K, T, R_A, Q_VERO, sig, kind, 300)):
                righe.append((scad, kind, k_, m - semi_spread, m + semi_spread))
    return pd.DataFrame(righe, columns=["scadenza", "kind", "strike", "bid", "ask"])


S0A, R_A = 100.0, 0.04


@pytest.fixture(scope="module")
def raw_am():
    return raw_americana()


@pytest.fixture(scope="module")
def chain_am(raw_am):
    return real.costruisci_chain(raw_am, S0A, R_A, DATA)


def test_q_americano_si_ritrova_e_la_parity_europea_no(chain_am):
    c = chain_am
    per_T = c.groupby("T")[["q", "q_eu"]].first()
    assert len(per_T) == 4
    assert np.allclose(per_T["q"], Q_VERO, atol=2e-4)
    # con esercizio anticipato la parity europea sbaglia il dividend yield di almeno 10 punti base
    assert (per_T["q_eu"] - Q_VERO > 1e-3).all()


def test_chain_finale(chain_am):
    c = chain_am
    assert list(c.columns) == ["T", "K", "kind", "mid", "S", "r", "q", "F", "q_eu", "spread_rel"]
    assert ((c["kind"] == "put") == (c["K"] < c["F"])).all()          # solo OTM
    assert not c.duplicated(["T", "K"]).any()
    assert np.allclose(c["S"] * np.exp((c["r"] - c["q"]) * c["T"]), c["F"], rtol=1e-12)


def test_iv_americana_della_chain_ritrova_lo_smile(chain_am):
    c = ch.add_iv_americana(chain_am)
    k = np.log(c["K"] / c["F"])
    assert c["iv"].notna().all()
    assert np.abs(c["iv"] - sigma_vera(k)).max() < 1.5e-3           # errore di q e dello spread
    # ignorare l'esercizio anticipato gonfia la IV delle put (e non tocca quella delle call)
    eu = ch.add_iv(c.drop(columns="iv"))
    diff = (eu["iv"] - c["iv"]).groupby(c["kind"]).mean()
    assert diff["put"] > 5e-4 and abs(diff["call"]) < 1e-4


def test_stima_q_nan_se_manca_una_gamba(raw_am):
    c = real.pulisci(raw_am, DATA)
    solo_call = c[c["kind"] == "call"]
    assert np.isnan(real.stima_q(solo_call[solo_call["T"] == solo_call["T"].iloc[0]], S0A, R_A, 30 / 365))


def quota(bid, ask, strike=5000.0):
    return dict(scadenza="2025-02-01", kind="call", strike=strike, bid=bid, ask=ask)


def test_ogni_filtro_da_solo():
    raw = pd.DataFrame([quota(10.0, 10.4, 5000.0), quota(0.0, 0.5, 5010.0),
                        quota(8.0, 7.0, 5020.0), quota(4.0, 8.0, 5030.0),
                        quota(0.02, 0.06, 5040.0)])
    tutti_aperti = dict(spread_max=99, mid_min=0)
    # bid nullo e quota incrociata cadono anche con gli altri filtri spenti
    assert list(real.pulisci(raw, DATA, **tutti_aperti)["K"]) == [5000.0, 5030.0, 5040.0]
    # spread troppo largo: 4/8 ha spread relativo 0,67
    assert list(real.pulisci(raw, DATA, spread_max=0.5, mid_min=0)["K"]) == [5000.0]
    # prezzo minuscolo (mid 0,04)
    assert list(real.pulisci(raw, DATA, spread_max=99, mid_min=0.10)["K"]) == [5000.0, 5030.0]


def test_mid_e_la_media_di_bid_e_ask():
    d = real.pulisci(pd.DataFrame([quota(10.0, 10.6, 5000.0), quota(20.0, 21.0, 5010.0)]), DATA)
    assert list(d["mid"]) == [10.3, 20.5]
    assert list(d["spread_rel"]) == pytest.approx([0.6 / 10.3, 1.0 / 20.5])


def test_forward_usa_gli_strike_vicini_allo_spot():
    d = real.pulisci(raw_finta(junk=False), DATA)
    giusto = real.forward_parity(d, S0, M.r)
    T = giusto.index[1]
    K = np.sort(d[(d["T"] == T)]["K"].unique())
    lontani = d["T"].eq(T) & d["K"].isin(np.concatenate([K[:8], K[-8:]])) & (d["kind"] == "call")
    d.loc[lontani, "mid"] += 30.0
    assert real.forward_parity(d, S0, M.r)[T] == pytest.approx(giusto[T], abs=1e-9)


def test_rimuovi_outlier_toglie_le_quote_sbagliate_e_tiene_le_altre():
    d = ch.add_iv(sy.chain(M, 100.0, 0.04, giorni=(63, 126)))
    d["F"] = d["S"] * np.exp((d["r"] - d["q"]) * d["T"])
    # due quote per scadenza con prezzo gonfiato del 35%: incoerenti con i vicini
    sbagliate = [i for _, g in d.groupby("T") for i in g.index[[3, len(g) // 2]]]
    sporco = d.copy()
    sporco.loc[sbagliate, "mid"] *= 1.35
    pulito = real.rimuovi_outlier(ch.add_iv(sporco.drop(columns="iv")))
    chiavi = lambda x: set(zip(x["T"].round(9), x["K"]))
    assert not (chiavi(d.loc[sbagliate]) & chiavi(pulito))          # le sbagliate sono sparite
    buone = d.drop(index=sbagliate).dropna(subset=["iv"])
    assert len(chiavi(buone) & chiavi(pulito)) >= 0.9 * len(chiavi(buone))   # le buone restano


def test_rimuovi_outlier_non_tocca_dati_puliti():
    d = ch.add_iv(sy.chain(M, 100.0, 0.04, giorni=(63, 126)))
    d["F"] = d["S"] * np.exp((d["r"] - d["q"]) * d["T"])
    assert len(real.rimuovi_outlier(d)) >= 0.97 * len(d.dropna(subset=["iv"]))


def test_curva_tassi_si_propaga_alla_chain(raw_am):
    curva = real.curva_tassi([0.1, 1.0], [0.03, 0.05])
    assert curva(0.0) == 0.03 and curva(5.0) == 0.05 and np.isclose(curva(0.55), 0.04)
    c = real.costruisci_chain(raw_am, S0A, curva, DATA)
    per_T = c.groupby("T")["r"].first()
    assert np.allclose(per_T.values, [curva(T) for T in per_T.index])
    assert per_T.nunique() > 1
    assert np.allclose(c["F"], c["S"] * np.exp((c["r"] - c["q"]) * c["T"]))


def test_riepilogo_conta_quote_e_scadenze_perse():
    a = pd.DataFrame({"T": [0.1, 0.1, 0.2, 0.3]})
    b = a[a["T"] != 0.2]
    rip = real.riepilogo([("grezza", a), ("filtrata", b)])
    assert rip["quote"].tolist() == [4, 3]
    assert rip["scadenze"].tolist() == [3, 2]
    assert rip["scadenze_perse"].tolist() == [0, 1]


def test_senza_put_nessun_q_stimabile(raw_am):
    with pytest.raises(ValueError, match="q stimabile"):
        real.costruisci_chain(raw_am[raw_am["kind"] == "call"], S0A, R_A, DATA)


def test_scadenza_senza_put_scartata_e_segnalata(raw_am):
    scad = sorted(raw_am["scadenza"].unique())[0]
    raw = raw_am[~((raw_am["scadenza"] == scad) & (raw_am["kind"] == "put"))]
    c = real.costruisci_chain(raw, S0A, R_A, DATA)
    assert len(c.attrs["scadenze_senza_q"]) == 1 and c["T"].nunique() == 3


def test_q_con_150_passi_come_con_300(raw_am):
    d = real.pulisci(raw_am, DATA)
    T = sorted(d["T"].unique())[1]
    g = d[d["T"] == T]
    assert abs(real.stima_q(g, S0A, R_A, T, 150) - real.stima_q(g, S0A, R_A, T, 300)) < 1e-4
