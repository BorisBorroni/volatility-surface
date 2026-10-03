import numpy as np
import pytest

from volsurf import bs, heston
from volsurf import chain as ch
from volsurf import synthetic as sy

M = sy.Mondo()


def test_legame_P_Q():
    p, q = M.params_P(0.04), M.params_Q(0.04)
    assert q.xi == p.xi and q.rho == p.rho
    assert abs(q.kappa * q.theta - p.kappa * p.theta) < 1e-12  # stesso drift a v = 0
    assert q.kappa < p.kappa and q.theta > p.theta
    assert p.feller_ok() and q.feller_ok()
    # senza premio i due mondi coincidono
    m0 = sy.Mondo(lam=0.0)
    a, b = m0.params_P(0.04), m0.params_Q(0.04)
    assert np.allclose([a.kappa, a.theta, a.xi, a.rho], [b.kappa, b.theta, b.xi, b.rho])
    with pytest.raises(ValueError):
        sy.Mondo(lam=4.0).params_Q(0.04)


def test_statistiche_P():
    S, V = sy.simulate_P(M, 1260, n_paths=300, n_sub=5, seed=1)
    r = np.diff(np.log(S), axis=0)
    assert abs(r.std() * np.sqrt(252) - np.sqrt(M.theta)) < 0.005   # volatilità annua
    assert abs(V.mean() / M.theta - 1) < 0.03                       # varianza media
    dv = np.diff(V, axis=0)
    assert np.corrcoef(r.ravel(), dv.ravel())[0, 1] < -0.6          # effetto leva
    r2 = r**2
    assert np.corrcoef(r2[:-1].ravel(), r2[1:].ravel())[0, 1] > 0.1  # volatility clustering


def test_martingala_P():
    # E[S_T] = S0 exp((mu - q) T): controlla il drift del simulatore
    S, _ = sy.simulate_P(M, 252, n_paths=20000, n_sub=4, seed=2)
    x = S[-1] * np.exp(-(M.mu - M.q) * 1.0)
    assert abs(x.mean() - 100.0) < 4 * x.std() / np.sqrt(len(x))


def test_simulatore_coerente_con_prezzatore():
    # simulando sotto Q con drift r, il prezzo Monte Carlo deve coincidere con quello di Heston
    pq = M.params_Q(0.04)
    S, _ = sy.simulate(100.0, pq, M.r, M.q, 126, n_paths=100_000, n_sub=4,
                       rng=np.random.default_rng(3))
    T = 0.5
    for K in (90.0, 100.0, 110.0):
        pay = np.exp(-M.r * T) * np.maximum(S[-1] - K, 0.0)
        exact = heston.call_price(100.0, K, T, M.r, M.q, pq)
        assert abs(pay.mean() - exact) < 4 * pay.std() / np.sqrt(len(pay)) + 0.03


def test_struttura_chain():
    S = 100.0
    df = sy.chain(M, S, 0.04)
    assert list(df.columns) == ["T", "K", "kind", "mid", "S", "r", "q"]
    assert not df.duplicated(["T", "K"]).any()
    assert (df["K"] == df["K"].round()).all()
    F = S * np.exp((M.r - M.q) * df["T"])
    assert ((df["kind"] == "put") == (df["K"] < F)).all()   # solo opzioni out-of-the-money
    for row in df.itertuples():
        lo, up = bs.price_bounds(S, row.K, row.T, M.r, M.q, row.kind)
        assert lo < row.mid < up


def test_chain_skew_negativo():
    d = ch.add_iv(sy.chain(M, 100.0, 0.04))
    assert d["iv"].between(0.05, 1.0).all()
    for T, g in d.groupby("T"):
        F = 100.0 * np.exp((M.r - M.q) * T)
        iv_at = lambda K0: g.iloc[(g["K"] - K0).abs().argmin()]["iv"]
        assert g.iloc[g["K"].argmin()]["iv"] > iv_at(F) > iv_at(F * np.exp(0.2 * np.sqrt(T)))


def test_tick():
    df = sy.chain(M, 100.0, 0.04, tick=0.01)
    assert np.allclose(df["mid"] * 100, np.round(df["mid"] * 100), atol=1e-9)


def _var_media(kappa, theta, v0, T):
    # E[(1/T) integrale di v] per un processo CIR che parte da v0
    return theta + (v0 - theta) * (1 - np.exp(-kappa * T)) / (kappa * T)


@pytest.mark.parametrize("v0", [0.02, 0.0359, 0.07])
def test_premio_pianificato(v0):
    # la IV ATM deve stare sopra la varianza media attesa sotto P (premio voluto)
    # e un po' sotto quella attesa sotto Q (effetto dello skew)
    g = 126
    T = g / 252
    F = 100.0 * np.exp((M.r - M.q) * T)
    pq = M.params_Q(v0)
    iv = bs.implied_vol(heston.call_price(100.0, F, T, M.r, M.q, pq), 100.0, F, T, M.r, M.q, "call")
    var_P = _var_media(M.kappa, M.theta, v0, T)
    var_Q = _var_media(pq.kappa, pq.theta, v0, T)
    assert 0.8 < iv**2 / var_Q < 1.0
    assert iv**2 / var_P > 1.05


def test_varianza_media_P_simulata():
    # il simulatore deve riprodurre la varianza media attesa sotto P
    v0, g = 0.07, 126
    _, V = sy.simulate(100.0, M.params_P(v0), M.mu, M.q, g, n_paths=20000, n_sub=5,
                       rng=np.random.default_rng(4))
    avg = 0.5 * (V[:-1] + V[1:]).mean(axis=0)
    expected = _var_media(M.kappa, M.theta, v0, g / 252)
    assert abs(avg.mean() - expected) < 4 * avg.std() / np.sqrt(len(avg)) + 0.0005


def test_chain_usa_parametri_Q():
    # i prezzi della chain sono quelli di Heston sotto Q, e non quelli sotto P
    S, v = 100.0, 0.05
    df = sy.chain(M, S, v)
    riga = df.iloc[len(df) // 2]
    exact_Q = heston.price(S, riga["K"], riga["T"], M.r, M.q, M.params_Q(v), riga["kind"])
    exact_P = heston.price(S, riga["K"], riga["T"], M.r, M.q, M.params_P(v), riga["kind"])
    assert abs(riga["mid"] - exact_Q) < 1e-12
    assert abs(riga["mid"] - exact_P) > 1e-4


def test_varianza_iniziale_stazionaria():
    # la varianza di partenza di simulate_P segue la distribuzione stazionaria del CIR:
    # media theta, varianza xi^2 theta / (2 kappa)
    _, V = sy.simulate_P(M, 1, n_paths=50_000, seed=5)
    v0 = V[0]
    assert abs(v0.mean() - M.theta) < 4 * v0.std() / np.sqrt(len(v0))
    assert abs(v0.var() / (M.xi**2 * M.theta / (2 * M.kappa)) - 1) < 0.05
