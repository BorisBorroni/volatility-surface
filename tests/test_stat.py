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
