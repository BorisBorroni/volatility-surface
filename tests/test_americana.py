import numpy as np
import pytest

from volsurf import americana as am
from volsurf import bs


def lsm_put(S0, K, T, r, q, sig, n_paths=100_000, n_steps=50, seed=0):
    """Put americana con Monte Carlo di Longstaff-Schwartz (metodo indipendente dall'albero)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_steps, n_paths))
    S = S0 * np.exp(np.cumsum((r - q - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * z, axis=0))
    cf = np.maximum(K - S[-1], 0.0)
    for t in range(n_steps - 2, -1, -1):
        cf *= np.exp(-r * dt)
        itm = K - S[t] > 0
        A = np.vander(S[t][itm] / K, 4)
        cont = A @ np.linalg.lstsq(A, cf[itm], rcond=None)[0]
        esercita = np.where(itm)[0][K - S[t][itm] > cont]
        cf[esercita] = K - S[t][esercita]
    return np.exp(-r * dt) * cf.mean()


def test_put_di_riferimento():
    # S = K = 100, T = 1, r = 5%, sigma = 20%: l'albero converge a circa 6,0905
    # (la put europea vale 5,5735, il resto è il premio di esercizio anticipato)
    assert am.prezzo(100, 100, 1, 0.05, 0, 0.2, "put", 1000)[0] == pytest.approx(6.0905, abs=1.5e-3)


def test_confronto_con_longstaff_schwartz():
    # Monte Carlo: errore statistico e leggero bias verso il basso, quindi tolleranza larga
    lsm = lsm_put(100, 100, 1, 0.05, 0, 0.2)
    assert lsm == pytest.approx(am.prezzo(100, 100, 1, 0.05, 0, 0.2, "put", 1000)[0], abs=0.12)
    lsm2 = lsm_put(100, 110, 0.5, 0.04, 0.013, 0.25)
    assert lsm2 == pytest.approx(am.prezzo(100, 110, 0.5, 0.04, 0.013, 0.25, "put", 1000)[0], abs=0.12)


def test_convergenza_nei_passi():
    ref = am.prezzo(100, 100, 1, 0.05, 0, 0.2, "put", 4000)[0]
    errori = [abs(am.prezzo(100, 100, 1, 0.05, 0, 0.2, "put", n)[0] - ref) for n in (50, 150, 500)]
    assert errori[2] < errori[1] < errori[0] and errori[1] < 4e-3


def test_simmetria_call_put():
    # McDonald-Schroder: call(S, K, r, q) = put(K, S, q, r)
    c = am.prezzo(100, 110, 0.5, 0.04, 0.013, 0.25, "call", 300)[0]
    p = am.prezzo(110, 100, 0.5, 0.013, 0.04, 0.25, "put", 300)[0]
    assert c == pytest.approx(p, abs=1e-9)


def test_casi_in_cui_coincide_con_black_scholes():
    # call senza dividendi e put con tasso nullo non si esercitano mai prima della scadenza
    for S, K in [(90, 100), (100, 100), (120, 100)]:
        assert am.prezzo(S, K, 1, 0.05, 0.0, 0.3, "call", 300)[0] == pytest.approx(
            bs.price(S, K, 1, 0.05, 0.0, 0.3, "call"), abs=5e-3)
        assert am.prezzo(S, K, 1, 0.0, 0.0, 0.3, "put", 300)[0] == pytest.approx(
            bs.price(S, K, 1, 0.0, 0.0, 0.3, "put"), abs=5e-3)


def test_esercizio_immediato():
    assert am.prezzo(50, 100, 1, 0.05, 0.0, 0.2, "put", 300)[0] == pytest.approx(50.0, abs=1e-12)
    # call molto in the money con dividendo alto e tasso basso: vale almeno l'intrinseco
    americana = am.prezzo(150, 100, 1, 0.02, 0.20, 0.2, "call", 300)[0]
    assert americana >= 50 and americana > bs.price(150, 100, 1, 0.02, 0.20, 0.2, "call") + 1


@pytest.mark.parametrize("S,K,T,r,q,sig", [(100, 100, 0.5, 0.04, 0.013, 0.2), (100, 90, 1.0, 0.05, 0.0, 0.3),
                                             (100, 120, 0.25, 0.03, 0.04, 0.15), (80, 100, 2.0, 0.06, 0.02, 0.4)])
def test_proprieta_di_non_arbitraggio(S, K, T, r, q, sig):
    c = am.prezzo(S, K, T, r, q, sig, "call", 300)[0]
    p = am.prezzo(S, K, T, r, q, sig, "put", 300)[0]
    assert c >= max(S - K, 0) - 1e-12 and p >= max(K - S, 0) - 1e-12
    assert c >= bs.price(S, K, T, r, q, sig, "call") - 3e-3      # il diritto in più non fa perdere valore
    assert p >= bs.price(S, K, T, r, q, sig, "put") - 3e-3
    assert S * np.exp(-q * T) - K - 3e-3 <= c - p <= S - K * np.exp(-r * T) + 3e-3   # parity americana


def test_monotonia_in_sigma_e_in_T():
    sig = np.linspace(0.05, 1.0, 20)
    for kind in ("call", "put"):
        assert np.all(np.diff(am.prezzo(100, 100, 0.5, 0.04, 0.013, sig, kind, 200)) > 0)
        T = [am.prezzo(100, 100, t, 0.04, 0.013, 0.2, kind, 200)[0] for t in (0.1, 0.3, 0.6, 1.0)]
        assert np.all(np.diff(T) > 0)


def test_vettoriale_uguale_a_scalare_anche_con_tipi_misti():
    K = np.array([90.0, 100.0, 110.0, 100.0])
    sig = np.array([0.2, 0.25, 0.3, 0.35])
    kind = np.array(["put", "call", "put", "call"])
    batch = am.prezzo(100, K, 0.5, 0.04, 0.013, sig, kind, 100)
    uno = [am.prezzo(100, K[i], 0.5, 0.04, 0.013, sig[i], kind[i], 100)[0] for i in range(4)]
    assert np.allclose(batch, uno, atol=1e-12)


def test_sigma_troppo_piccolo_per_l_albero():
    with pytest.raises(ValueError):
        am.prezzo(100, 100, 1, 0.10, 0.0, 0.0005, "call", 100)


def test_kind_non_valido():
    with pytest.raises(ValueError):
        am.prezzo(100, 100, 1, 0.04, 0.0, 0.2, "straddle")


def test_iv_andata_e_ritorno():
    K = np.linspace(80, 120, 9)
    sig = 0.15 + 0.002 * (K - 80)
    kind = np.where(K < 100, "put", "call")
    p = am.prezzo(100, K, 0.5, 0.04, 0.013, sig, kind, 200)
    assert np.allclose(am.iv(p, 100, K, 0.5, 0.04, 0.013, kind, 200), sig, atol=1e-7)


def test_iv_coincide_con_black_scholes_dove_non_c_e_esercizio_anticipato():
    K = np.array([90.0, 100.0, 110.0])
    p = bs.price(100, K, 0.5, 0.04, 0.0, 0.22, "call")
    # niente dividendi: la call americana vale come la europea (a meno dell'errore dell'albero)
    assert np.allclose(am.iv(p, 100, K, 0.5, 0.04, 0.0, "call", 400), 0.22, atol=2e-4)


def test_iv_nan_fuori_dai_limiti():
    p = np.array([0.5, 10.0, 105.0, np.nan, 3.0])
    K = np.array([100.0] * 5)
    out = am.iv(p, 100, K, 0.5, 0.04, 0.013, "call", 100)
    assert np.isnan(out[0])                      # 0,5 sotto il valore minimo possibile con sigma >= 2%
    assert np.isnan(out[2]) and np.isnan(out[3]) and np.isfinite(out[1]) and np.isfinite(out[4])
    # prezzo sotto il valore intrinseco
    assert np.isnan(am.iv(5.0, 100, 90.0, 0.5, 0.04, 0.013, "call", 100)[0])
    assert np.isnan(am.iv(1.0, 100, 110.0, 0.0, 0.04, 0.013, "put", 100)[0])
