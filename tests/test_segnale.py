from dataclasses import replace

import numpy as np
import pytest

from volsurf import bs, heston
from volsurf import segnale as sg
from volsurf import synthetic as sy

M = sy.Mondo()


def test_iv_atm_senza_vol_della_vol():
    # con xi ~ 0 la varianza è deterministica: IV^2 = media di v(t) sotto Q su [0, T]
    m = replace(M, xi=0.01, rho=0.0, lam=1.0)
    kq, thq = m.kappa - m.lam, m.kappa * m.theta / (m.kappa - m.lam)
    for v, T in [(0.02, 0.25), (0.06, 0.5), (0.04, 1.0)]:
        media = thq + (v - thq) * (1 - np.exp(-kq * T)) / (kq * T)
        assert sg.iv_atm_esatta(m, v, T) == pytest.approx(np.sqrt(media), abs=2e-4)


def test_iv_atm_e_a_strike_forward():
    # con skew negativo la IV a K=F è diversa da quella a K=S: verifico lo strike giusto
    v, T = 0.04, 0.5
    F = 100.0 * np.exp((M.r - M.q) * T)
    pq = M.params_Q(v)

    def iv_a(K):
        return bs.implied_vol(heston.price(100.0, K, T, M.r, M.q, pq, "call"),
                              100.0, K, T, M.r, M.q, "call")
    assert sg.iv_atm_esatta(M, v, T) == pytest.approx(iv_a(F), abs=1e-9)
    assert abs(iv_a(F) - iv_a(100.0)) > 5e-4


def test_iv_atm_cresce_con_v():
    iv = [sg.iv_atm_esatta(M, v, 0.5) for v in (0.01, 0.03, 0.06, 0.12)]
    assert np.all(np.diff(iv) > 0)


def test_griglia_vs_esatta():
    f = sg.IVAtm(M, 0.5, 0.2)
    for v in (0.0123, 0.0359, 0.0777, 0.1851):
        assert abs(f(v) - sg.iv_atm_esatta(M, v, 0.5)) < 1e-5


def test_griglia_fuori_intervallo():
    f = sg.IVAtm(M, 0.5, 0.2)
    with pytest.raises(ValueError):
        f(0.25)
    with pytest.raises(ValueError):
        f(np.array([0.05, 1e-5]))


def test_allineamento_passato_futuro():
    # prima metà ferma, seconda metà con rendimenti +/- s: la vol "passata" e quella "futura"
    s = 0.01
    ret = np.concatenate([np.zeros(40), np.tile([s, -s], 20)])
    S = (100 * np.exp(np.concatenate([[0], np.cumsum(ret)])))[:, None]
    V = np.full(S.shape, 0.04)
    iv, rb, rf = sg.tabella(M, S, V, 20)
    assert iv.shape == rb.shape == rf.shape == (41, 1)       # t = 20, ..., 60
    vol = s * np.sqrt(252)
    assert rb[0, 0] == 0.0 and rf[0, 0] == 0.0               # t = 20
    assert rb[20, 0] == pytest.approx(0.0) and rf[20, 0] == pytest.approx(vol)   # t = 40
    assert rb[40, 0] == pytest.approx(vol) and rf[40, 0] == pytest.approx(vol)   # t = 60
    assert np.allclose(iv, sg.iv_atm_esatta(M, 0.04, 20 / 252), atol=1e-4)


def premio(mondo, S, V, giorni):
    iv, _, rf = sg.tabella(mondo, S, V, giorni)
    return (iv - rf).mean() * 100          # punti di volatilità


def test_il_premio_inserito_si_ritrova():
    # stessi percorsi, con e senza premio: la differenza è il premio che abbiamo messo noi
    S, V = sy.simulate_P(M, 756, n_paths=400, seed=1)
    senza, con = replace(M, lam=0.0), M
    diff = {}
    for g in (21, 63, 126):
        p0, p2 = premio(senza, S, V, g), premio(con, S, V, g)
        assert abs(p0) < 0.5                  # senza premio resta solo convessità/skew (< 0,5 pt)
        diff[g] = p2 - p0
    assert 0.4 < diff[21] < 1.0
    assert 1.3 < diff[63] < 2.2
    assert 2.2 < diff[126] < 3.4
    assert diff[21] < diff[63] < diff[126]    # il premio cresce con la scadenza
