import itertools

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from volsurf import bs

# parametri di un caso generico, usati in più test
P = dict(S=100.0, K=105.0, T=0.5, r=0.03, q=0.01, sigma=0.25)


def test_valori_hull():
    # S=42, K=40, r=10%, sigma=20%, T=6 mesi: call 4.76, put 0.81 (Hull)
    c = bs.price(42, 40, 0.5, 0.10, 0.0, 0.20, "call")
    p = bs.price(42, 40, 0.5, 0.10, 0.0, 0.20, "put")
    assert abs(c - 4.76) < 0.005
    assert abs(p - 0.81) < 0.005


def test_put_call_parity():
    rng = np.random.default_rng(0)
    for _ in range(200):
        S = rng.uniform(50, 150)
        K = rng.uniform(50, 150)
        T = rng.uniform(0.02, 3)
        r = rng.uniform(-0.01, 0.06)
        q = rng.uniform(0, 0.04)
        sigma = rng.uniform(0.05, 1.0)
        c = bs.price(S, K, T, r, q, sigma, "call")
        p = bs.price(S, K, T, r, q, sigma, "put")
        assert abs((c - p) - (S * np.exp(-q * T) - K * np.exp(-r * T))) < 1e-10


@pytest.mark.parametrize("kind", ["call", "put"])
def test_prezzo_vs_integrazione_numerica(kind):
    # verifica indipendente dalla formula chiusa: e^{-rT} E[payoff] con S_T lognormale
    S, K, T, r, q, sigma = (P[k] for k in ("S", "K", "T", "r", "q", "sigma"))
    m = (r - q - 0.5 * sigma**2) * T
    s = sigma * np.sqrt(T)
    z_star = (np.log(K / S) - m) / s  # punto in cui il payoff si annulla
    if kind == "call":
        val, _ = quad(lambda z: (S * np.exp(m + s * z) - K) * norm.pdf(z), z_star, 12)
    else:
        val, _ = quad(lambda z: (K - S * np.exp(m + s * z)) * norm.pdf(z), -12, z_star)
    assert abs(np.exp(-r * T) * val - bs.price(**P, kind=kind)) < 1e-8


def test_greche_vs_differenze_finite():
    S, K, T, r, q, sigma = (P[k] for k in ("S", "K", "T", "r", "q", "sigma"))

    def pr(S_=S, sig=sigma, kind="call"):
        return bs.price(S_, K, T, r, q, sig, kind)

    h = 0.01
    for kind in ("call", "put"):
        d_fd = (pr(S + h, kind=kind) - pr(S - h, kind=kind)) / (2 * h)
        assert abs(d_fd - bs.delta(S, K, T, r, q, sigma, kind)) < 1e-6

    h = 0.1
    g_fd = (pr(S + h) - 2 * pr() + pr(S - h)) / h**2
    assert abs(g_fd - bs.gamma(S, K, T, r, q, sigma)) < 1e-5

    h = 1e-5
    v_fd = (pr(sig=sigma + h) - pr(sig=sigma - h)) / (2 * h)
    assert abs(v_fd - bs.vega(S, K, T, r, q, sigma)) < 1e-5


def test_delta_call_meno_put():
    # delta_call - delta_put = e^{-qT}
    dc = bs.delta(**P, kind="call")
    dp = bs.delta(**P, kind="put")
    assert abs((dc - dp) - np.exp(-P["q"] * P["T"])) < 1e-12


@pytest.mark.parametrize("kind", ["call", "put"])
def test_round_trip_prezzo_iv(kind):
    S, r, q = 100.0, 0.03, 0.01
    n = 0
    for K, T, sigma in itertools.product(
        [60, 80, 95, 100, 105, 120, 150], [0.05, 0.25, 1.0, 2.0], [0.05, 0.2, 0.6, 1.5]
    ):
        p = bs.price(S, K, T, r, q, sigma, kind)
        iv = bs.implied_vol(p, S, K, T, r, q, kind)
        if np.isnan(iv):
            continue  # prezzo praticamente uguale al valore intrinseco: IV non identificabile
        # il prezzo ricostruito deve coincidere sempre
        assert abs(bs.price(S, K, T, r, q, iv, kind) - p) < 1e-9
        # la vol si recupera dove il prezzo è sensibile a sigma
        if bs.vega(S, K, T, r, q, sigma) > 1e-2:
            assert abs(iv - sigma) < 1e-7
            n += 1
    assert n > 40  # il test non deve svuotarsi


def test_iv_fuori_dai_limiti():
    S, K, T, r, q = 100, 100, 0.5, 0.03, 0.01
    lower, upper = bs.price_bounds(S, K, T, r, q, "call")
    assert np.isnan(bs.implied_vol(lower - 0.01, S, K, T, r, q, "call"))
    assert np.isnan(bs.implied_vol(upper + 0.01, S, K, T, r, q, "call"))
    assert np.isnan(bs.implied_vol(5.0, S, K, 0.0, r, q, "call"))  # T = 0
    assert np.isnan(bs.implied_vol(np.nan, S, K, T, r, q, "call"))


def test_kind_non_valido():
    with pytest.raises(ValueError):
        bs.price(100, 100, 1, 0.01, 0.0, 0.2, "straddle")


def test_iv_array():
    K = np.array([90.0, 100.0, 110.0, 100.0])
    kind = np.array(["put", "call", "call", "put"])
    p = np.array([bs.price(100, k, 0.5, 0.03, 0.01, 0.2, t) for k, t in zip(K, kind)])
    iv = bs.implied_vol_array(p, 100, K, 0.5, 0.03, 0.01, kind)
    assert np.allclose(iv, 0.2, atol=1e-8)


def test_limiti_no_arbitraggio():
    S, T, r, q = 100.0, 0.5, 0.03, 0.01
    fs, fk = S * np.exp(-q * T), 100.0 * np.exp(-r * T)
    # call ATM: limite inferiore = max(fs - fk, 0), superiore = fs
    lo, up = bs.price_bounds(S, 100.0, T, r, q, "call")
    assert lo == max(fs - fk, 0.0) and up == fs
    # call molto OTM: il limite inferiore non può essere negativo
    lo, _ = bs.price_bounds(S, 200.0, T, r, q, "call")
    assert lo == 0.0
    # put molto OTM: idem
    lo, up = bs.price_bounds(S, 50.0, T, r, q, "put")
    assert lo == 0.0 and up == 50.0 * np.exp(-r * T)
    # prezzi nulli o negativi non hanno una volatilità implicita
    assert np.isnan(bs.implied_vol(0.0, S, 200.0, T, r, q, "call"))
    assert np.isnan(bs.implied_vol(-1.0, S, 200.0, T, r, q, "call"))
