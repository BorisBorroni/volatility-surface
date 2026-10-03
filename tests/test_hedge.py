from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from volsurf import backtest as bt
from volsurf import bs, hedge
from volsurf import synthetic as sy

R, Q, MU = 0.04, 0.013, 0.10
S0, N = 100.0, 63
T = N / 252
K = S0 * np.exp((R - Q) * T)


def gbm(sig, n_paths, n=N, seed=0):
    rng = np.random.default_rng(seed)
    inc = (MU - Q - 0.5 * sig**2) / 252 + sig / np.sqrt(252) * rng.standard_normal((n, n_paths))
    return S0 * np.exp(np.cumsum(np.vstack([np.zeros((1, n_paths)), inc]), axis=0))


def test_un_giorno_a_mano():
    # n = 1: nessun ribilanciamento, il conto si fa con le mani
    S = np.array([100.0, 101.5])
    K1, sig = 100.0, 0.2
    T1 = 1 / 252
    d = bs.delta(100.0, K1, T1, R, Q, sig)
    cassa = (bs.price(100.0, K1, T1, R, Q, sig) - d * 100.0) * np.exp(R * T1) \
        + d * 100.0 * (np.exp(Q * T1) - 1)
    atteso = cassa + d * 101.5 - 1.5
    pnl, _ = hedge.call_venduta_coperta(S, K1, R, Q, sig)
    assert pnl == pytest.approx(atteso, rel=1e-12)


def test_guadagno_atteso_uguale_a_differenza_di_prezzi():
    # volatilità reale 15% contro implicita 20%: il P&L medio è e^{rT} (C(20%) - C(15%)),
    # qualunque sia il rendimento atteso mu
    S = gbm(0.15, 4000)
    pnl, _ = hedge.call_venduta_coperta(S, K, R, Q, 0.20)
    atteso = np.exp(R * T) * (bs.price(S0, K, T, R, Q, 0.20) - bs.price(S0, K, T, R, Q, 0.15))
    assert pnl.mean() == pytest.approx(atteso, abs=0.04)


def test_vol_reale_uguale_a_implicita_guadagno_nullo():
    pnl, _ = hedge.call_venduta_coperta(gbm(0.20, 4000, seed=1), K, R, Q, 0.20)
    assert abs(pnl.mean()) < 0.04


def test_vol_reale_sopra_implicita_perdita():
    pnl, _ = hedge.call_venduta_coperta(gbm(0.25, 4000, seed=2), K, R, Q, 0.20)
    assert pnl.mean() < -0.8


def test_formula_gamma_segue_il_pnl_percorso_per_percorso():
    pnl, teoria = hedge.call_venduta_coperta(gbm(0.18, 3000, seed=3), K, R, Q, 0.20)
    assert np.corrcoef(pnl, teoria)[0, 1] > 0.97
    assert (pnl - teoria).std() < 0.2 * pnl.std()
    assert pnl.mean() == pytest.approx(teoria.mean(), abs=0.03)


def test_copertura_meno_frequente_aumenta_la_dispersione():
    S = gbm(0.20, 3000, seed=4)
    giornaliera, _ = hedge.call_venduta_coperta(S, K, R, Q, 0.20, ogni=1)
    settimanale, _ = hedge.call_venduta_coperta(S, K, R, Q, 0.20, ogni=5)
    assert settimanale.std() > 1.4 * giornaliera.std()


def test_percorso_singolo_uguale_alla_colonna():
    S = gbm(0.2, 5, seed=5)
    tutti, th = hedge.call_venduta_coperta(S, K, R, Q, 0.2)
    uno, th1 = hedge.call_venduta_coperta(S[:, 2], K, R, Q, 0.2)
    assert uno == pytest.approx(tutti[2]) and th1 == pytest.approx(th[2])


def test_backtest_struttura_e_coerenza_con_la_teoria():
    M = sy.Mondo()
    S, V = sy.simulate_P(M, 504, n_paths=60, seed=11)
    df = bt.backtest(M, S, V, 63)
    # S ha 505 righe (t = 0..504): ingressi a t = 63, ..., 441; l'ultimo scade esattamente a 504
    assert sorted(df["t0"].unique()) == [63, 126, 189, 252, 315, 378, 441]
    assert len(df) == 60 * 7
    assert np.allclose(df["spread"], df["iv"] - df["rv_ind"])
    assert np.corrcoef(df["pnl"], df["teoria"])[0, 1] > 0.97


def test_backtest_ritrova_il_premio():
    M = sy.Mondo()
    S, V = sy.simulate_P(M, 1260, n_paths=300, seed=7)
    con = bt.backtest(M, S, V, 63)
    senza = bt.backtest(replace(M, lam=0.0), S, V, 63)
    assert 0.15 < con["pnl"].mean() < 0.40       # vendere vol con premio guadagna (% dello spot)
    assert abs(senza["pnl"].mean()) < 0.15       # senza premio resta circa zero
    assert con["pnl"].mean() - senza["pnl"].mean() > 0.2


def test_riassunto():
    import pandas as pd
    r = bt.riassunto(pd.DataFrame({"pnl": [1.0, -2.0, 3.0, 2.0]}))
    assert r["n"] == 4 and r["media"] == 1.0 and r["in_guadagno"] == 0.75 and r["peggiore"] == -2.0
    assert r["dev_std"] == pytest.approx(np.std([1, -2, 3, 2], ddof=1))


def test_costi_di_transazione_abbassano_il_pnl_come_previsto():
    M = sy.Mondo()
    S, _ = sy.simulate_P(M, 63, n_paths=50, seed=5)
    K = S[0] * np.exp((M.r - M.q) * 63 / 252)
    senza, _ = hedge.call_venduta_coperta(S, K, M.r, M.q, 0.2)
    con, _ = hedge.call_venduta_coperta(S, K, M.r, M.q, 0.2, costo=0.001)
    assert (con < senza).all()
    # costo minimo: l'acquisto iniziale e la chiusura finale della copertura
    assert (senza - con > 0.001 * bs.delta(S[0], K, 63 / 252, M.r, M.q, 0.2) * S[0]).all()
    nullo, _ = hedge.call_venduta_coperta(S, K, M.r, M.q, 0.2, costo=0.0)
    assert np.array_equal(nullo, senza)


def test_media_con_ic_contiene_la_media_e_usa_i_percorsi():
    df = pd.DataFrame({"percorso": np.repeat(np.arange(50), 4),
                       "pnl": np.random.default_rng(0).standard_normal(200)})
    r = bt.media_con_ic(df)
    assert r["ic_basso"] < r["media"] < r["ic_alto"]
    assert np.isclose(r["media"], df["pnl"].mean())


def test_backtest_reale_su_prezzi_gbm_ritrova_il_premio_della_vol_usata():
    # prezzi con vol 15% e opzione venduta a 20%: P&L positivo; a 15% circa nullo
    rng = np.random.default_rng(3)
    r_ = rng.standard_normal(1500) * 0.15 / np.sqrt(252)
    S = 100 * np.exp(np.cumsum(np.r_[0, r_]))
    alta = bt.backtest_reale(S, np.full(len(S), 0.20), 21, 0.0, 0.0)
    pari = bt.backtest_reale(S, np.full(len(S), 0.15), 21, 0.0, 0.0)
    assert len(alta) == len(S) - 42
    assert alta["pnl"].mean() > 0.5 and abs(pari["pnl"].mean()) < 0.2
    assert np.corrcoef(alta["pnl"], alta["teoria"])[0, 1] > 0.95


def test_backtest_reale_salta_i_giorni_senza_iv():
    rng = np.random.default_rng(4)
    S = 100 * np.exp(np.cumsum(np.r_[0, rng.standard_normal(200) * 0.01]))
    iv = np.full(len(S), 0.2)
    iv[::2] = np.nan
    df = bt.backtest_reale(S, iv, 21, 0.0, 0.0)
    assert len(df) == len([t for t in range(21, len(S) - 21) if t % 2 == 1])
    assert np.isfinite(df["pnl"]).all() and (df["t0"] % 2 == 1).all()
