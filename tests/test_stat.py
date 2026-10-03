import numpy as np

from volsurf import stat


def _ar1(n, phi, rng):
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def test_senza_ritardi_coincide_con_l_errore_standard_classico():
    x = np.random.default_rng(0).standard_normal(500)
    m, se, t = stat.media_hac(x, 0)
    assert np.isclose(m, x.mean()) and np.isclose(se, x.std() / np.sqrt(500)) and np.isclose(t, m / se)


def test_serie_autocorrelata_errore_standard_piu_grande_e_giusto():
    # AR(1) con phi=0.8: errore standard vero della media = sigma_x / sqrt(n) * sqrt((1+phi)/(1-phi))
    rng = np.random.default_rng(1)
    phi, n = 0.8, 4000
    ses = [stat.media_hac(_ar1(n, phi, rng), 60)[1] for _ in range(30)]
    vero = 1 / np.sqrt(1 - phi**2) / np.sqrt(n) * np.sqrt((1 + phi) / (1 - phi))
    assert abs(np.mean(ses) / vero - 1) < 0.1
    assert vero > 2.5 / np.sqrt(n) / np.sqrt(1 - phi**2) * 0.99    # molto più del classico


def test_pendenza_hac_ritrova_la_pendenza_e_l_errore_standard_classico_senza_ritardi():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(2000)
    y = 0.5 * x + rng.standard_normal(2000)
    b, se = stat.pendenza_hac(y, x, 0)
    assert abs(b - 0.5) < 3 * se
    assert np.isclose(se, 1 / np.sqrt(2000), rtol=0.1)


def test_pendenza_hac_con_finestre_sovrapposte_e_piu_incerta():
    rng = np.random.default_rng(3)
    z = rng.standard_normal(3000 + 20)
    x = np.convolve(z, np.ones(21) / 21, "valid")        # medie mobili: autocorrelazione alta
    y = np.convolve(rng.standard_normal(3020), np.ones(21) / 21, "valid")
    _, se0 = stat.pendenza_hac(y, x, 0)
    _, se20 = stat.pendenza_hac(y, x, 21)
    assert se20 > 2 * se0
