"""Call venduta e coperta in delta (delta-hedging), con volatilità di copertura fissa.

Si vende una call al prezzo di Black-Scholes con la volatilità implicita sigma di inizio
periodo e ogni giorno si ribilancia il sottostante con la delta di Black-Scholes calcolata
con lo stesso sigma. Con copertura continua il guadagno a scadenza è (Derman, Carr):
    P&L = integrale di e^{r(T-t)} * 1/2 * Gamma * S^2 * (sigma_imp^2 - sigma_reale(t)^2) dt
quindi dipende solo da quanto la varianza realizzata sta sotto quella implicita. Qui la
copertura è giornaliera, e la formula si scrive con i rendimenti veri del giorno:
    sum_i e^{r(T-t_i)} * 1/2 * Gamma_i * (sigma^2 S_i^2 dt - (S_{i+1} - S_i)^2).
"""
import numpy as np

from . import bs

DT = 1.0 / 252


def call_venduta_coperta(S, K, r, q, sigma, ogni=1, costo=0.0):
    """P&L a scadenza di una call venduta a prezzo BS(sigma) e coperta ogni `ogni` giorni.

    S: prezzi giornalieri da inizio a scadenza, array (n + 1) o (n + 1, percorsi); la
    scadenza è T = n/252. K e sigma: scalari o array (uno per percorso).
    costo: costo proporzionale di ogni acquisto o vendita di azioni (0.0005 = 5 punti base del
    controvalore), pagato anche sull'acquisto iniziale e sulla chiusura finale della copertura.
    Restituisce (pnl, teoria): il P&L simulato in valore a scadenza e la somma gamma sopra
    (la teoria non include i costi).
    """
    S = np.asarray(S, dtype=float)
    n = len(S) - 1
    T = n * DT
    delta = bs.delta(S[0], K, T, r, q, sigma)
    cassa = bs.price(S[0], K, T, r, q, sigma) - delta * S[0]    # incasso del premio meno azioni
    cassa = cassa - costo * np.abs(delta) * S[0]
    teoria = np.zeros_like(cassa)
    for i in range(n):
        tau = T - i * DT
        gam = bs.gamma(S[i], K, tau, r, q, sigma)
        dS = S[i + 1] - S[i]
        teoria += np.exp(r * tau) * 0.5 * gam * (sigma**2 * S[i] ** 2 * DT - dS**2)
        cassa = cassa * np.exp(r * DT) + delta * S[i] * (np.exp(q * DT) - 1)   # interesse e dividendi
        if i + 1 < n and (i + 1) % ogni == 0:                     # ribilancio a fine giornata
            nuova = bs.delta(S[i + 1], K, T - (i + 1) * DT, r, q, sigma)
            cassa = cassa - (nuova - delta) * S[i + 1] - costo * np.abs(nuova - delta) * S[i + 1]
            delta = nuova
    pnl = cassa + delta * S[n] - costo * np.abs(delta) * S[n] - np.maximum(S[n] - K, 0.0)
    return pnl, teoria
