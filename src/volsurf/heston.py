"""Modello di Heston: prezzi europei con la funzione caratteristica (formula di Lewis).

Sotto la misura di prezzo Q:
    dS = (r - q) S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + xi sqrt(v) dW2,    corr(dW1, dW2) = rho
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad


@dataclass(frozen=True)
class HestonParams:
    v0: float     # varianza iniziale
    kappa: float  # velocità di ritorno alla media
    theta: float  # varianza di lungo periodo
    xi: float     # volatilità della varianza
    rho: float    # correlazione tra prezzo e varianza

    def feller_ok(self):
        """Se 2 kappa theta >= xi^2 la varianza non tocca mai lo zero."""
        return 2 * self.kappa * self.theta >= self.xi**2


def char_fn(u, T, p):
    """E[exp(i u X_T)] con X_T = ln(S_T / F_T), F_T prezzo forward.

    Forma "little trap" (Albrecher et al.), stabile per T grandi.
    """
    b = p.kappa - 1j * p.rho * p.xi * u
    d = np.sqrt(b * b + p.xi**2 * (1j * u + u * u))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = p.kappa * p.theta / p.xi**2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    D = (b - d) / p.xi**2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * p.v0)


def call_price(S, K, T, r, q, p):
    F = S * np.exp((r - q) * T)
    k = np.log(F / K)

    def integrand(u):
        return (np.exp(1j * u * k) * char_fn(u - 0.5j, T, p)).real / (u * u + 0.25)

    integrale, _ = quad(integrand, 0, np.inf, limit=200)
    c = np.exp(-r * T) * (F - np.sqrt(F * K) / np.pi * integrale)
    # l'integrale numerico sbaglia di ~1e-8 nelle code: si riporta il prezzo nei limiti di non-arbitraggio
    return min(max(c, S * np.exp(-q * T) - K * np.exp(-r * T), 0.0), S * np.exp(-q * T))


def put_price(S, K, T, r, q, p):
    # put-call parity
    return call_price(S, K, T, r, q, p) - (S * np.exp(-q * T) - K * np.exp(-r * T))


def price(S, K, T, r, q, p, kind="call"):
    if kind == "call":
        return call_price(S, K, T, r, q, p)
    if kind == "put":
        return put_price(S, K, T, r, q, p)
    raise ValueError("kind deve essere 'call' o 'put'")
