import numpy as np
import pandas as pd
import pytest

from volsurf import realized as rz


def gbm_ohlc(sig_notte, sig_giorno, n_days, m=60, seed=0):
    """OHLC giornalieri da un moto browniano geometrico con salto notturno e m passi intraday."""
    rng = np.random.default_rng(seed)
    c_prev, righe = 100.0, []
    for _ in range(n_days):
        o = c_prev * np.exp(sig_notte * rng.standard_normal())
        passi = np.concatenate([[0.0], np.cumsum(sig_giorno / np.sqrt(m) * rng.standard_normal(m))])
        p = o * np.exp(passi)
        righe.append((o, p.max(), p.min(), p[-1]))
        c_prev = p[-1]
    return pd.DataFrame(righe, columns=["open", "high", "low", "close"])


def test_close_to_close_a_mano():
    S = np.array([100.0, 101.0, 99.0, 102.0, 103.0])
    r = np.diff(np.log(S))
    v = rz.close_to_close(S, 3)
    assert np.isnan(v[:3]).all()
    assert v[3] == pytest.approx(np.sqrt(252 * np.mean(r[0:3] ** 2)))
    assert v[4] == pytest.approx(np.sqrt(252 * np.mean(r[1:4] ** 2)))


def test_close_to_close_prezzi_costanti_e_vol_nota():
    assert rz.close_to_close(np.full(30, 100.0), 10)[-1] == 0.0
    # rendimenti alternati +/- s: la vol realizzata è esattamente s*sqrt(252)
    s = 0.01
    S = 100 * np.exp(np.cumsum(np.tile([s, -s], 20)))
    assert rz.close_to_close(S, 20)[-1] == pytest.approx(s * np.sqrt(252))


def test_close_to_close_percorsi_indipendenti():
    rng = np.random.default_rng(1)
    S = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal((60, 3)), axis=0))
    v = rz.close_to_close(S, 20)
    assert v.shape == S.shape
    for j in range(3):
        assert np.allclose(v[20:, j], rz.close_to_close(S[:, j], 20)[20:])


def test_close_to_close_stima_la_vera_vol():
    rng = np.random.default_rng(2)
    sig = 0.20
    S = 100 * np.exp(np.cumsum(sig / np.sqrt(252) * rng.standard_normal(60_000)))
    assert rz.close_to_close(S, 59_000)[-1] == pytest.approx(sig, rel=0.01)


def yz_a_mano(df, n):
    """Yang-Zhang scritto con cicli espliciti sull'ultima finestra."""
    o, h, lo, c = (np.log(df[x].values) for x in ("open", "high", "low", "close"))
    t = range(len(df) - n, len(df))
    notte = np.array([o[i] - c[i - 1] for i in t])
    giorno = np.array([c[i] - o[i] for i in t])
    rs = np.array([(h[i] - c[i]) * (h[i] - o[i]) + (lo[i] - c[i]) * (lo[i] - o[i]) for i in t])
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = notte.var(ddof=1) + k * giorno.var(ddof=1) + (1 - k) * rs.mean()
    return np.sqrt(252 * var)


def test_yang_zhang_a_mano():
    df = gbm_ohlc(0.004, 0.009, 80, seed=3)
    assert rz.yang_zhang(df, 21).iloc[-1] == pytest.approx(yz_a_mano(df, 21), rel=1e-12)


def test_yang_zhang_stima_la_vera_vol_e_batte_close_to_close():
    sn, sg = 0.004, 0.009
    vera = np.sqrt(252 * (sn**2 + sg**2))
    n = 21
    yz, cc = [], []
    for seed in range(150):
        df = gbm_ohlc(sn, sg, n + 1, m=1000, seed=seed)
        yz.append(rz.yang_zhang(df, n).iloc[-1])
        cc.append(rz.close_to_close(df["close"].values, n)[-1])
    # massimo e minimo campionati su m passi sottostimano gli estremi veri: il bias di YZ è
    # circa -9% con m=60, -5% con 240, -2% con 1000 (scala come 1/sqrt(m)), quindi qui 4%
    assert np.mean(yz) == pytest.approx(vera, rel=0.04)
    assert np.mean(cc) == pytest.approx(vera, rel=0.04)
    assert np.std(yz) < np.std(cc)


def test_iv_in_tempo_di_borsa_conserva_la_varianza_totale():
    iv = np.array([0.10, 0.25])
    s = rz.iv_in_tempo_di_borsa(iv, 21)
    assert s**2 * 21 / 252 == pytest.approx(iv**2 * 30 / 365)
    assert rz.iv_in_tempo_di_borsa(0.2, 21) == pytest.approx(0.2 * 0.99313, rel=1e-4)
