import numpy as np
import pytest

from volsurf import bs, heston

S, T, r, q = 100.0, 1.0, 0.03, 0.01
P = heston.HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)


def test_funzione_caratteristica_martingala():
    # E[S_T / F_T] = 1  <=>  phi(-i) = 1
    assert abs(heston.char_fn(-1j, 1.0, P) - 1) < 1e-12
    assert abs(heston.char_fn(-1j, 5.0, P) - 1) < 1e-12


def test_limite_black_scholes():
    # con xi -> 0, rho = 0 e v0 = theta la varianza è costante: Heston = BS con sigma = sqrt(v0)
    p = heston.HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.01, rho=0.0)
    for K in (80, 100, 120):
        h = heston.call_price(S, K, T, r, q, p)
        b = bs.price(S, K, T, r, q, 0.2, "call")
        assert abs(h - b) < 5e-4


def test_correzione_lineare_in_xi():
    # con rho != 0 la differenza da BS è del primo ordine in xi (è lo skew che nasce):
    # dimezzando xi la differenza deve dimezzarsi
    def diff(xi, K):
        p = heston.HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=xi, rho=-0.7)
        return heston.call_price(S, K, T, r, q, p) - bs.price(S, K, T, r, q, 0.2, "call")

    for K in (80, 120):
        ratio = diff(0.01, K) / diff(0.005, K)
        assert 1.9 < ratio < 2.1


def test_put_call_parity():
    for K in (85, 100, 115):
        c = heston.price(S, K, T, r, q, P, "call")
        pt = heston.price(S, K, T, r, q, P, "put")
        assert abs((c - pt) - (S * np.exp(-q * T) - K * np.exp(-r * T))) < 1e-10


def test_limiti_no_arbitraggio():
    for kind in ("call", "put"):
        for K in (70, 90, 100, 110, 140):
            v = heston.price(S, K, T, r, q, P, kind)
            lo, up = bs.price_bounds(S, K, T, r, q, kind)
            assert lo <= v <= up


def test_convessita_in_strike():
    # niente arbitraggio butterfly: la derivata seconda del prezzo rispetto a K è >= 0
    K = np.arange(60.0, 141.0, 5.0)
    c = np.array([heston.call_price(S, k, T, r, q, P) for k in K])
    assert np.all(c[:-2] - 2 * c[1:-1] + c[2:] > 0)
    assert np.all(np.diff(c) < 0)  # decrescente in K


def test_skew_con_rho_negativo():
    # rho < 0: la IV delle put OTM (strike bassi) supera quella delle call OTM
    def iv(K):
        return bs.implied_vol(heston.call_price(S, K, T, r, q, P), S, K, T, r, q, "call")

    assert iv(85) > iv(100) > iv(115)
    # con rho > 0 lo skew si inverte
    p_pos = heston.HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=0.7)
    v85 = bs.implied_vol(heston.call_price(S, 85, T, r, q, p_pos), S, 85, T, r, q, "call")
    v115 = bs.implied_vol(heston.call_price(S, 115, T, r, q, p_pos), S, 115, T, r, q, "call")
    assert v85 < v115


def _mc_call(p, K_list, T, n_paths=200_000, n_steps=250, seed=42):
    """Prezzi call per simulazione di Heston (Eulero con troncamento della varianza)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    logS = np.full(n_paths, np.log(S))
    v = np.full(n_paths, p.v0)
    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = p.rho * z1 + np.sqrt(1 - p.rho**2) * rng.standard_normal(n_paths)
        vp = np.maximum(v, 0.0)
        logS += (r - q - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
        v += p.kappa * (p.theta - vp) * dt + p.xi * np.sqrt(vp * dt) * z2
    ST = np.exp(logS)
    out = []
    for K in K_list:
        payoff = np.exp(-r * T) * np.maximum(ST - K, 0.0)
        out.append((payoff.mean(), payoff.std() / np.sqrt(n_paths)))
    return out


# v0 = theta, e poi v0 diverso da theta (altrimenti i due parametri non si distinguono)
MC_CASI = [
    (heston.HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.35, rho=-0.7), 1.0),
    (heston.HestonParams(v0=0.02, kappa=2.5, theta=0.06, xi=0.40, rho=-0.6), 0.5),
    (heston.HestonParams(v0=0.02, kappa=2.5, theta=0.06, xi=0.40, rho=-0.6), 2.0),
]


@pytest.mark.parametrize("p,T_", MC_CASI)
def test_monte_carlo(p, T_):
    # verifica indipendente dalla formula: il prezzo di Heston deve coincidere con la simulazione
    strikes = (85.0, 100.0, 115.0)
    for K, (mc, se) in zip(strikes, _mc_call(p, strikes, T_), strict=True):
        assert abs(mc - heston.call_price(S, K, T_, r, q, p)) < 4 * se + 0.03


def test_prezzo_sempre_dentro_i_limiti_di_non_arbitraggio():
    p = heston.HestonParams(0.04, 4, 0.0359, 0.48, -0.7)
    S, r, q = 100.0, 0.04, 0.013
    for T, K in [(1 / 252, 140), (1 / 252, 60), (2, 300), (0.01, 200), (0.5, 100)]:
        c = heston.call_price(S, K, T, r, q, p)
        assert max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) <= c <= S * np.exp(-q * T)
        assert heston.put_price(S, K, T, r, q, p) >= -1e-12
