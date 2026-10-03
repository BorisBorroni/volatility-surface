import numpy as np
import pandas as pd
import pytest

from volsurf import bs, heston, svi
from volsurf import chain as ch
from volsurf import synthetic as sy
from volsurf.surface import Superficie

pytestmark = pytest.mark.slow

TS = [0.1, 0.25, 0.5, 1.0]
S0, R, Q = 100.0, 0.04, 0.013


def fits_finti():
    righe = [dict(a=0.04 * T, b=0.10, rho=-0.4, m=0.0, sigma=0.15) for T in TS]
    return pd.DataFrame(righe, index=pd.Index(TS, name="T"))


@pytest.fixture
def surf():
    return Superficie(fits_finti(), S0, R, Q)


def w_slice(surf, i, k):
    return svi.w(k, surf.slices[i])


K_GRID = np.linspace(-0.5, 0.5, 41)


def test_riproduce_scadenze_quotate(surf):
    for i, T in enumerate(TS):
        assert np.allclose(surf.w(K_GRID, T), w_slice(surf, i, K_GRID), atol=1e-14)


def test_interpolazione_pesata(surf):
    # a 1/4 del tratto tra T=0.25 e T=0.5 pesa 0.75 sulla prima e 0.25 sulla seconda
    T = 0.25 + 0.25 * 0.25
    atteso = 0.75 * w_slice(surf, 1, K_GRID) + 0.25 * w_slice(surf, 2, K_GRID)
    assert np.allclose(surf.w(K_GRID, T), atteso, atol=1e-14)


def test_calendar_su_griglia_fine(surf):
    Tf = np.linspace(0.02, 2.0, 400)
    w = np.array([surf.w(K_GRID, T) for T in Tf])
    assert np.all(np.diff(w, axis=0) > 0)


def test_estrapolazione_iv_costante(surf):
    # fuori dalle scadenze la IV (a pari k) resta quella del bordo
    for T, i in [(0.03, 0), (3.0, 3)]:
        iv_bordo = np.sqrt(w_slice(surf, i, K_GRID) / TS[i])
        assert np.allclose(np.sqrt(surf.w(K_GRID, T) / T), iv_bordo, atol=1e-12)


def test_butterfly_sulle_curve_interpolate(surf):
    for T in [0.05, 0.17, 0.4, 0.75, 1.5]:
        assert surf.g(np.linspace(-1.5, 1.5, 301), T).min() >= 0


def test_g_coincide_con_svi_alle_scadenze(surf):
    k = np.linspace(-0.8, 0.8, 33)
    assert np.allclose(surf.g(k, 0.5), svi.durrleman(k, surf.slices[2]), atol=1e-12)


def test_iv_usa_il_forward(surf):
    T = 0.5
    F = S0 * np.exp((R - Q) * T)
    assert surf.forward(T) == pytest.approx(F)
    k = np.array([-0.2, 0.0, 0.3])
    iv_k = np.sqrt(surf.w(k, T) / T)
    assert np.allclose(surf.iv(F * np.exp(k), T), iv_k, atol=1e-14)


def test_T_non_positivo(surf):
    with pytest.raises(ValueError):
        surf.w(0.0, 0.0)
    with pytest.raises(ValueError):
        surf.iv(100.0, -1.0)


def test_scalare_e_array(surf):
    assert np.ndim(surf.iv(100.0, 0.4)) == 0
    assert surf.iv(np.array([90.0, 100.0, 110.0]), 0.4).shape == (3,)


def test_accuratezza_su_chain_sintetica():
    # la superficie fittata, in una scadenza non quotata, resta vicina alla IV vera di Heston
    M = sy.Mondo()
    v0 = 0.04
    df = ch.add_iv(sy.chain(M, 100.0, v0))
    surf = Superficie.da_chain(df)
    pq = M.params_Q(v0)
    for giorni in (30, 90):
        T = giorni / 252
        F = 100.0 * np.exp((M.r - M.q) * T)
        K = F * np.exp(np.linspace(-1.5, 1.5, 7) * 0.2 * np.sqrt(T))
        vere = []
        for K_ in K:
            kind = "put" if K_ < F else "call"
            p = heston.price(100.0, K_, T, M.r, M.q, pq, kind)
            vere.append(bs.implied_vol(p, 100.0, K_, T, M.r, M.q, kind))
        assert np.abs(surf.iv(K, T) - np.array(vere)).max() < 0.0060


def test_da_chain_legge_parametri_di_mercato():
    M = sy.Mondo()
    df = ch.add_iv(sy.chain(M, 100.0, 0.04, giorni=(63, 126)))
    surf = Superficie.da_chain(df)
    assert (surf.S, surf.r, surf.q) == (100.0, M.r, M.q)
    assert np.allclose(surf.T, [63 / 252, 126 / 252])


def test_forward_con_carry_per_scadenza():
    carry = [0.010, 0.020, 0.030, 0.040]
    s = Superficie(fits_finti(), S0, R, Q, carry=carry)
    for T, c in zip(TS, carry, strict=True):
        assert s.forward(T) == pytest.approx(S0 * np.exp(c * T))
    T = 0.375                                     # a metà tra 0.25 e 0.5: carry 0.025
    assert s.forward(T) == pytest.approx(S0 * np.exp(0.025 * T))
    assert s.forward(0.01) == pytest.approx(S0 * np.exp(0.010 * 0.01))   # fuori: carry del bordo
    assert s.forward(3.0) == pytest.approx(S0 * np.exp(0.040 * 3.0))


def test_iv_con_carry_usa_il_forward_giusto():
    s = Superficie(fits_finti(), S0, R, Q, carry=[0.01, 0.02, 0.03, 0.04])
    T, K = 0.5, 105.0
    k = np.log(K / (S0 * np.exp(0.03 * T)))
    assert s.iv(K, T) == pytest.approx(np.sqrt(s.w(k, T) / T), abs=1e-14)


def test_filtra_moneyness_e_g_dati():
    from volsurf import real
    df = pd.DataFrame({"T": [0.25] * 4, "K": [60.0, 95.0, 100.0, 140.0], "F": [100.0] * 4})
    tenuti = real.filtra_moneyness(df, c=0.6)         # soglia 0.6 * sqrt(0.25) = 0.3
    assert list(tenuti["K"]) == [95.0, 100.0]
    assert real.filtra_moneyness(df, c=2.0).shape[0] == 4


def test_da_chain_con_q_diverso_per_scadenza():
    M = sy.Mondo()
    df = ch.add_iv(sy.chain(M, 100.0, 0.04, giorni=(63, 126)))
    df.loc[df["T"] == 126 / 252, "q"] += 0.01          # q implicito diverso nella seconda scadenza
    surf = Superficie.da_chain(df)
    assert np.allclose(surf.carry, [M.r - M.q, M.r - M.q - 0.01])
    assert surf.forward(126 / 252) == pytest.approx(100.0 * np.exp((M.r - M.q - 0.01) * 126 / 252))


def fits_con_dominio():
    f = fits_finti()
    f["k_min"] = [-0.30, -0.40, -0.50, -0.60]
    f["k_max"] = [0.10, 0.20, 0.30, 0.40]
    return f


def test_dominio_e_iv_fuori_dai_dati():
    s = Superficie(fits_con_dominio(), S0, R, Q)
    assert s.dominio(0.25) == (-0.40, 0.20)                 # scadenza quotata
    assert s.dominio(0.375) == (-0.40, 0.20)                # tra 0.25 e 0.5: intersezione
    assert s.dominio(0.05) == (-0.30, 0.10)                 # prima della prima: bordo
    assert s.dominio(3.0) == (-0.60, 0.40)                  # dopo l'ultima: bordo
    T = 0.375
    F = s.forward(T)
    dentro, fuori = F * np.exp(0.0), F * np.exp(0.25)       # k = 0 e k = 0.25 (oltre 0.20)
    assert np.isfinite(s.iv(dentro, T))
    assert np.isnan(s.iv(fuori, T))
    assert np.isfinite(s.iv(fuori, T, fuori="estrapola"))
    assert np.isnan(s.iv(F * np.exp(-0.45), T))             # sotto il minimo dell'intersezione
    with pytest.raises(ValueError):
        s.iv(dentro, T, fuori="boh")


def test_senza_dominio_noto_non_si_taglia_niente(surf):
    assert surf.dominio(0.3) == (-np.inf, np.inf)
    assert np.isfinite(surf.iv(100.0 * np.exp(5.0), 0.3))


def test_dominio_da_chain_coincide_con_i_dati():
    M = sy.Mondo()
    df = ch.add_iv(sy.chain(M, 100.0, 0.04, giorni=(63, 126)))
    surf = Superficie.da_chain(df)
    for T, g in df.dropna(subset=["iv"]).groupby("T"):
        F = 100.0 * np.exp((M.r - M.q) * T)
        k = np.log(g["K"] / F)
        i = int(np.argmin(np.abs(surf.T - T)))
        assert (surf.k_min[i], surf.k_max[i]) == pytest.approx((k.min(), k.max()))
