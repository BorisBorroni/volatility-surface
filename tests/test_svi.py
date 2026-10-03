import numpy as np
import pandas as pd
import pytest

from volsurf import bs, heston, svi
from volsurf import chain as ch
from volsurf import synthetic as sy

pytestmark = pytest.mark.slow

M = sy.Mondo()
P0 = svi.SVI(a=0.02, b=0.10, rho=-0.4, m=0.02, sigma=0.15)


def iv_vera(v0, T, k):
    """Volatilità implicita esatta del mondo sintetico a log-moneyness k."""
    F = 100.0 * np.exp((M.r - M.q) * T)
    K = F * np.exp(k)
    pq = M.params_Q(v0)
    out = []
    for K_ in K:
        kind = "put" if K_ < F else "call"
        pr = heston.price(100.0, K_, T, M.r, M.q, pq, kind)
        out.append(bs.implied_vol(pr, 100.0, K_, T, M.r, M.q, kind))
    return np.array(out)


def scarto_centro(df, fits, v0):
    """Massimo scarto tra superficie fittata e vera nella zona centrale (+/- 1 dev. standard)."""
    peggiore = 0.0
    for T, row in fits.iterrows():
        p = svi.SVI(row.a, row.b, row.rho, row.m, row.sigma)
        k = np.linspace(-0.2 * np.sqrt(T), 0.2 * np.sqrt(T), 25)
        peggiore = max(peggiore, np.abs(np.sqrt(svi.w(k, p) / T) - iv_vera(v0, T, k)).max())
    return peggiore


def test_derivate_vs_differenze_finite():
    k = np.linspace(-0.4, 0.5, 7)
    h = 1e-5
    assert np.allclose(svi.dw(k, P0), (svi.w(k + h, P0) - svi.w(k - h, P0)) / (2 * h), atol=1e-8)
    assert np.allclose(svi.d2w(k, P0), (svi.dw(k + h, P0) - svi.dw(k - h, P0)) / (2 * h), atol=1e-6)


def test_durrleman_volatilita_piatta():
    # w costante (b = 0): densità lognormale, g = 1 ovunque
    flat = svi.SVI(a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.1)
    assert np.allclose(svi.durrleman(np.linspace(-1, 1, 21), flat), 1.0)


def test_durrleman_riconosce_arbitraggio():
    # esempio di Gatheral con densità negativa
    p = svi.SVI(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)
    assert svi.durrleman(np.linspace(-1.5, 1.5, 3001), p).min() < -0.01
    # una curva ragionevole non ne ha
    assert svi.durrleman(np.linspace(-1.5, 1.5, 3001), P0).min() > 0


def test_fit_recupera_parametri_noti():
    k = np.linspace(-0.5, 0.5, 25)
    p = svi.fit_slice(k, svi.w(k, P0))
    assert np.allclose(svi.w(k, p), svi.w(k, P0), atol=1e-8)
    for nome in ("a", "b", "rho", "m", "sigma"):
        assert abs(getattr(p, nome) - getattr(P0, nome)) < 1e-3


@pytest.fixture(scope="module")
def chain_base():
    df = ch.add_iv(sy.chain(M, 100.0, 0.0359))
    return df, svi.fit_chain(df)


def test_fit_su_chain_heston(chain_base):
    df, fits = chain_base
    assert len(fits) == 5
    assert (fits["rmse_iv"] < 0.002).all()       # meno di 20 bp sui punti
    assert scarto_centro(df, fits, 0.0359) < 0.0010   # meno di 10 bp dalla superficie vera
    assert (fits["g_min"] > 0).all()             # niente arbitraggio butterfly


def test_niente_arbitraggio_calendar(chain_base):
    _, fits = chain_base
    assert svi.calendar_min(fits, np.linspace(-0.3, 0.3, 121)) > 0


@pytest.mark.parametrize("v0", [0.02, 0.07])
def test_fit_con_altri_livelli_di_varianza(v0):
    df = ch.add_iv(sy.chain(M, 100.0, v0))
    fits = svi.fit_chain(df)
    assert scarto_centro(df, fits, v0) < 0.0010
    assert (fits["g_min"] > 0).all()


def test_robusto_a_prezzi_arrotondati():
    # con tick 0.01 alcuni prezzi lontani valgono 0: la IV non esiste e la riga va scartata
    df = ch.add_iv(sy.chain(M, 100.0, 0.0359, tick=0.01))
    assert df["iv"].isna().any()
    fits = svi.fit_chain(df)
    assert len(fits) == 5
    assert scarto_centro(df, fits, 0.0359) < 0.0025


def test_robusto_al_rumore():
    df = ch.add_iv(sy.chain(M, 100.0, 0.0359))
    df["iv"] = df["iv"] + np.random.default_rng(0).normal(0, 0.0010, len(df))   # 10 bp
    fits = svi.fit_chain(df)
    assert scarto_centro(df, fits, 0.0359) < 0.0025
    assert (fits["g_min"] > 0).all()


def test_scadenza_con_pochi_punti_saltata():
    df = ch.add_iv(sy.chain(M, 100.0, 0.0359))
    T0 = df["T"].min()
    ridotto = df.drop(df[df["T"] == T0].index[4:])   # una scadenza con soli 4 punti
    assert len(svi.fit_chain(ridotto)) == 4


def test_durrleman_vs_densita_numerica():
    # verifica indipendente: la densità di S_T ottenuta derivando due volte i prezzi delle call
    # (Breeden-Litzenberger) coincide con g(k) / (K sqrt(2 pi w)) exp(-d2^2 / 2).
    T, h = 1.0, 1e-3

    def call(K):
        k = np.log(K)
        sig = np.sqrt(svi.w(k, P0) / T)
        return bs.price(1.0, K, T, 0.0, 0.0, sig, "call")

    for k in (-0.3, 0.0, 0.2):
        K = np.exp(k)
        dens_num = (call(K + h) - 2 * call(K) + call(K - h)) / h**2
        wk = svi.w(k, P0)
        d2 = -k / np.sqrt(wk) - np.sqrt(wk) / 2
        dens_g = svi.durrleman(k, P0) / (K * np.sqrt(2 * np.pi * wk)) * np.exp(-d2**2 / 2)
        assert abs(dens_num / dens_g - 1) < 1e-4


def test_fit_chain_riporta_g_sul_solo_intervallo_dei_dati_e_il_numero_di_quote():
    df = ch.add_iv(sy.chain(M, 100.0, 0.04, giorni=(63, 126)))
    fits = svi.fit_chain(df)
    n = df.dropna(subset=["iv"]).groupby("T").size()
    assert (fits["n"].values == n.values).all()
    assert (fits["g_dati"] >= fits["g_min"] - 1e-12).all()      # intervallo dei dati dentro la griglia di controllo
    assert (fits["g_dati"] > fits["g_min"]).any()           # l'intervallo dei dati esclude le code
    assert (fits["g_dati"] > 0).all()


def test_ala_al_limite_segnalata_quando_i_dati_stanno_su_un_lato():
    # dati su un solo lato dello smile (come nelle scadenze corte reali): un'ala non è
    # identificata e la sua pendenza va sul limite di Lee; con dati su entrambi i lati no
    T, S, r, q = 14 / 365, 100.0, 0.04, 0.013
    F = S * np.exp((r - q) * T)
    vero = svi.SVI(a=-0.2305, b=1.0554, rho=-0.8951, m=-0.9518, sigma=0.4906)
    k = np.linspace(-0.17, 0.04, 40)
    df = pd.DataFrame({"T": T, "K": F * np.exp(k), "kind": "put", "mid": 1.0, "S": S, "r": r,
                       "q": q, "iv": np.sqrt(svi.w(k, vero) / T)})
    una_sola = svi.fit_chain(df)
    assert una_sola["al_limite"].iloc[0]
    assert una_sola["k_min"].iloc[0] == pytest.approx(k.min())
    assert una_sola["k_max"].iloc[0] == pytest.approx(k.max())
    due_lati = svi.fit_chain(ch.add_iv(sy.chain(M, 100.0, 0.04, giorni=(63, 126))))
    assert not due_lati["al_limite"].any()
