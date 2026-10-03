"""Superficie di volatilità implicita continua, costruita dai fit SVI delle singole scadenze.

A log-moneyness k fissato la varianza totale è interpolata linearmente in T tra le due scadenze
vicine (così non c'è arbitraggio calendar). Fuori dall'intervallo delle scadenze la IV è
mantenuta costante (w proporzionale a T). Ogni scadenza conosce il range di log-moneyness dei
suoi dati: fuori da quel dominio `iv` restituisce nan, perché lì lo SVI è solo un'estrapolazione.
"""
import numpy as np

from . import svi


class Superficie:
    def __init__(self, fits, S, r, q, carry=None):
        """carry: r - q per ogni scadenza di fits (se manca vale r - q per tutte)."""
        self.T = fits.index.values.astype(float)
        self.slices = [svi.SVI(x.a, x.b, x.rho, x.m, x.sigma) for x in fits.itertuples()]
        self.S, self.r, self.q = S, r, q
        self.carry = np.full(len(self.T), r - q) if carry is None else np.asarray(carry, dtype=float)
        # range di log-moneyness coperto dai dati di ogni scadenza (se noto): fuori, lo SVI è
        # solo un'estrapolazione
        self.k_min = fits["k_min"].values if "k_min" in fits else np.full(len(self.T), -np.inf)
        self.k_max = fits["k_max"].values if "k_max" in fits else np.full(len(self.T), np.inf)

    @classmethod
    def da_chain(cls, df):
        """df: chain con colonna iv (vedi chain.add_iv). r e q possono cambiare per scadenza
        (sui dati reali q è un valore implicito per scadenza)."""
        fits = svi.fit_chain(df)
        per_T = df.groupby("T")[["r", "q"]].first().reindex(fits.index)
        return cls(fits, df["S"].iloc[0], per_T["r"].iloc[0], per_T["q"].iloc[0],
                   carry=(per_T["r"] - per_T["q"]).values)

    def forward(self, T):
        """Forward a scadenza T: il carry r - q è interpolato tra le scadenze (costante fuori)."""
        return self.S * np.exp(np.interp(T, self.T, self.carry) * T)

    def _pesi(self, T):
        # restituisce (indice sinistro, indice destro, peso del destro, fattore di scala)
        if T <= 0:
            raise ValueError("T deve essere positivo")
        if T <= self.T[0]:
            return 0, 0, 0.0, T / self.T[0]
        if T >= self.T[-1]:
            n = len(self.T) - 1
            return n, n, 0.0, T / self.T[-1]
        j = int(np.searchsorted(self.T, T, side="right") - 1)
        lam = (T - self.T[j]) / (self.T[j + 1] - self.T[j])
        return j, j + 1, lam, 1.0

    def _comb(self, f, k, T):
        i, j, lam, scala = self._pesi(T)
        return scala * ((1 - lam) * f(k, self.slices[i]) + lam * f(k, self.slices[j]))

    def dominio(self, T):
        """(k_min, k_max) dove la superficie a scadenza T poggia su dati: l'intersezione dei
        range delle due scadenze vicine (quello del bordo fuori dall'intervallo)."""
        i, j, _, _ = self._pesi(T)
        return max(self.k_min[i], self.k_min[j]), min(self.k_max[i], self.k_max[j])

    def w(self, k, T):
        """Varianza totale a log-moneyness k e scadenza T."""
        return self._comb(svi.w, np.asarray(k, dtype=float), T)

    def g(self, k, T):
        """Condizione di Durrleman sulla curva interpolata a scadenza T."""
        k = np.asarray(k, dtype=float)
        return svi.durrleman_wdd(k, self.w(k, T), self._comb(svi.dw, k, T),
                                 self._comb(svi.d2w, k, T))

    def iv(self, K, T, fuori="nan"):
        """IV a strike K e scadenza T. Fuori dal dominio dei dati restituisce nan, a meno che
        fuori="estrapola" (allora si usano le ali dello SVI, non affidabili)."""
        if fuori not in ("nan", "estrapola"):
            raise ValueError('fuori deve essere "nan" o "estrapola"')
        k = np.log(np.asarray(K, dtype=float) / self.forward(T))
        iv = np.sqrt(self.w(k, T) / T)
        if fuori == "nan":
            lo, hi = self.dominio(T)
            iv = np.where((k >= lo) & (k <= hi), iv, np.nan)
        return iv
