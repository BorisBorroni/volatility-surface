# volatility-surface

Progetto individuale di finanza quantitativa. Parto dalle quotazioni delle opzioni su SPY, ne ricavo la volatilità implicita, la sintetizzo in una superficie (strike × scadenza), la confronto con la volatilità realizzata e verifico con un backtest se vendere opzioni e coprirle in delta ha prodotto un guadagno.

Il codice è in `src/volsurf`, i risultati sono in `notebooks/analisi.ipynb`.

**Risultato in breve.** La superficie si costruisce bene e senza arbitraggi evidenti nell'intervallo dei dati. Sul confronto implicita contro realizzata il premio c'è per il VIX (circa 3.5 punti di volatilità), ma per una singola opzione at-the-money a 30 giorni è di circa 0.4 punti e non è distinguibile da zero. Il backtest storico su 8 anni di SPY dà un guadagno medio per operazione di +0.07% dello spot (t = 1.24), che sparisce con costi di pochi punti base o con una IV più bassa di 0.7 punti. Quindi un vantaggio non è dimostrato.

## Indice

1. Struttura del progetto
2. Installazione e uso
3. Convenzioni
4. Black-Scholes e volatilità implicita
5. Opzioni americane
6. Dalla chain scaricata alla chain pulita
7. Il modello SVI
8. La superficie
9. Il mondo sintetico (Heston)
10. Volatilità realizzata
11. Delta hedging
12. Statistica: perché Newey-West
13. Dati storici
14. Risultati
15. Limiti
16. Test

## 1. Struttura del progetto

```
src/volsurf/      libreria: un modulo per argomento
scripts/          comandi da lanciare (scaricare i dati, costruire la superficie, backtest)
notebooks/        analisi.ipynb (risultati con grafici); genera_notebook.py lo rigenera
tests/            test automatici (pytest)
data/             serie storiche (prezzi, VIX, IV); data/corrente/ (non versionata) contiene la chain scaricata all'ultimo uso
output/           file derivati, non versionato
```

I moduli, in ordine di dipendenza: `bs` (Black-Scholes), `heston`, `synthetic` (mercato sintetico), `chain` (IV su una chain), `americana` (albero binomiale), `real` (pulizia dei dati reali), `svi`, `surface`, `realized`, `hedge`, `backtest`, `segnale`, `stat`, `dati`.

## 2. Installazione e uso

Serve Python 3.10 o superiore. Dalla cartella del progetto:

```
pip install -e ".[dev,dati]"
```

`dev` installa pytest, `dati` installa yfinance (serve per scaricare i dati correnti), `notebook` installa quello che serve per rigenerare il notebook (`pip install -e ".[notebook]"`). Le serie storiche incluse in `data/` bastano per rifare l'analisi storica; la superficie richiede di scaricare prima la chain (primo comando qui sotto).

Comandi, nell'ordine in cui li uso:

```
python scripts/scarica_dati.py --solo-chain          # chain di SPY e curva dei tassi dell'ultima seduta
python scripts/scarica_dati.py --solo-storico --anni 8   # aggiorna prezzi e VIX
python scripts/costruisci_superficie.py            # pulizia, IV, fit SVI (circa 1 minuto)
python scripts/backtest_storico.py                 # premio IV - RV e backtest su 8 anni
python scripts/sensibilita.py                      # backtest sintetico: premio di rischio, frequenza, costi
python scripts/controlla_iv.py                     # confronto tra la mia IV e quella del fornitore
pytest                                             # 132 test (pytest -m "not slow" salta i più lenti)
```

`scarica_dati.py` prende la chain da Yahoo Finance (yfinance), la curva dei tassi dal sito FRED (Treasury a 1, 3, 6 mesi, 1 e 2 anni) e i prezzi giornalieri di SPY e del VIX. Le chain non si possono scaricare per date passate: Yahoo dà solo la seduta corrente. Per questo la superficie è sempre quella dell'ultima seduta disponibile, mentre la storia viene da un'altra fonte (sezione 13).

**Analisi corrente e analisi storica.** L'analisi del presente (la superficie) si calcola sui dati scaricati nel momento in cui si usa il progetto: `scarica_dati.py --solo-chain` salva la chain e la curva dei tassi in `data/corrente/` (cartella non versionata) e `costruisci_superficie.py` la elabora, quindi nel repository non c'è nessuna chain salvata. L'analisi storica (premio implicita-realizzata e backtest) usa invece le serie in `data/`: prezzi e VIX si aggiornano con `scarica_dati.py --solo-storico --anni 8`, mentre la serie di IV di DoltHub va riesportata (sezione 13). Il notebook ricalcola la parte corrente a ogni esecuzione. I numeri della sezione 14 relativi alla superficie sono quelli di un'esecuzione di esempio e cambiano a ogni giorno di mercato; quelli storici sono riproducibili sui dati inclusi.

## 3. Convenzioni

- Tempo delle opzioni: giorni di calendario / 365. Tempo della volatilità realizzata e del mondo sintetico: giorni di borsa / 252. Sono due convenzioni diverse e le tengo separate; nel backtest una call a "21 giorni" ha T = 21/252.
- Tassi e dividendi sono continui: r è il tasso privo di rischio, q il dividend yield.
- Volatilità sempre annua in decimali (0.20 = 20%).
- k = ln(K/F) è la log-moneyness rispetto al forward F; w = IV² · T è la varianza totale.

## 4. Black-Scholes e volatilità implicita (`bs.py`)

Prezzo europeo di Black-Scholes-Merton con dividend yield continuo, più delta, gamma e vega. La volatilità implicita è il valore di sigma per cui il prezzo del modello eguaglia il prezzo di mercato; la ricavo con il metodo di Brent sull'intervallo [1e-6, 5], dopo aver controllato che il prezzo stia nei limiti di non arbitraggio (altrimenti la IV non esiste e il risultato è NaN). Il prezzo è strettamente crescente in sigma, quindi se la soluzione esiste è unica; un metodo che non usa la vega funziona anche vicino alla scadenza, dove la vega è quasi nulla.

## 5. Opzioni americane (`americana.py`)

Le opzioni su SPY sono americane, quindi Black-Scholes non basta: il prezzo di un'opzione americana è sempre almeno quello europeo. Uso un albero binomiale di Cox-Ross-Rubinstein: a ogni nodo il valore è il massimo tra continuazione ed esercizio immediato. Per ridurre l'oscillazione dei prezzi al variare dei passi, all'ultimo passo prima della scadenza uso il prezzo di Black-Scholes come valore di continuazione (variante Broadie-Detemple, "BBS"). Ho usato 300 passi per le IV.

Le funzioni lavorano su array: la IV americana è una bisezione (su [0.02, 3.0]) fatta in parallelo su tutte le opzioni di una scadenza, non una alla volta, e per questo una chain intera si calcola in pochi secondi.

## 6. Dalla chain scaricata alla chain pulita (`real.py`)

Questo è il passaggio più delicato: i dati grezzi contengono quote inutilizzabili e mancano due informazioni che servono per prezzare, cioè il tasso e i dividendi. I passi, in ordine (lo script `costruisci_superficie.py` stampa quante quote restano dopo ciascuno):

1. **Quote valide.** Tengo le scadenze tra 14 giorni e 2 anni (sotto i 14 giorni le quote hanno poco valore temporale e spread relativi alti, quindi la IV è poco informativa), scarto le quote con bid nullo o incrociate (ask < bid), quelle con spread relativo oltre il 50% e quelle con prezzo medio sotto 0.10. Il prezzo di riferimento è il mid, (bid + ask)/2.
2. **Tasso per scadenza.** Interpolo linearmente la curva del Treasury americano (FRED) alla scadenza di ciascuna opzione, invece di usare un tasso unico. Con un tasso fisso il dividend yield che ne esce risulta distorto, quindi la curva è necessaria.
3. **Dividend yield implicito.** Con opzioni americane la put-call parity vale solo come disuguaglianza, quindi non posso ricavare il forward da C − P. Per ogni scadenza cerco allora il q per cui la IV americana della call e quella della put coincidono sui 5 strike più vicini allo spot (se q è sbagliato le due curve si separano). Uso 150 passi per questa stima invece di 300: su una chain con q noto (test `test_q_con_150_passi_come_con_300`) e su SPY la differenza in q è sotto 1e-4 e il calcolo costa la metà. Se non trovo un q (nessun cambio di segno) la scadenza viene scartata e lo dichiaro nell'output. Nota importante: questo q non è il dividend yield dichiarato di SPY. Assorbe anche il costo di finanziamento e qualsiasi differenza tra il tasso Treasury e quello effettivo, quindi lo chiamo **carry implicito** (r − q). I valori ottenuti sono sotto quelli che mi aspetto da SPY (circa 1–1.3% annuo): da −1.5% a 14 giorni a +0.1–0.5% oltre i tre mesi. Parte della differenza è strutturale: SPY paga dividendi trimestrali discreti (di norma con stacco il terzo venerdì di marzo, giugno, settembre e dicembre), quindi le scadenze brevi, che non contengono nessuno stacco, non hanno dividendi da scontare. Il resto, circa 0.6–1 punto, è compatibile con un costo di finanziamento implicito sopra il rendimento dei Treasury, ma non l'ho verificato. Il punto da tenere a mente è che il carry è stimato **per scadenza**, quindi il forward di ogni scadenza è coerente con le quote di quella scadenza qualunque ne sia la causa; ma i singoli valori di q (specie quelli negativi sulle scadenze brevi) non vanno letti come stime di dividendi.
4. **Un lato per strike.** Tengo le put sotto il forward e le call sopra (opzioni out-of-the-money), perché sono le più liquide e hanno meno componente di esercizio anticipato.
5. **Moneyness.** Tengo gli strike con |ln(K/F)| ≤ 0.6·√T, una fascia che si allarga con la scadenza. Le code lontanissime hanno prezzi quasi nulli e IV poco informative e deformano il fit.
6. **IV americana** di ogni quota e **rimozione degli outlier**: per ogni scadenza confronto la IV di una quota con la retta tra i due vicini a sinistra e a destra (gli smile sono lisci), e scarto la quota peggiore se si discosta più di 4 volte lo scarto robusto (MAD) e comunque più di 0.5 punti. Ripeto fino a 15 volte, togliendo una sola quota per scadenza alla volta, perché un outlier fa sembrare sbagliati anche i vicini.

In un'esecuzione di esempio: 8959 quote grezze, 2467 quote finali, 17 scadenze (da 14 a 441 giorni).

## 7. Il modello SVI (`svi.py`)

Per ogni scadenza interpolo la varianza totale con la parametrizzazione SVI "raw":

    w(k) = a + b ( rho (k − m) + sqrt((k − m)² + sigma²) )

Cinque parametri: a è il livello, b l'angolo dello smile, rho l'asimmetria (skew), m lo spostamento, sigma la curvatura al minimo. Ho scelto SVI perché è lo standard di mercato per questo problema, ha pochi parametri, e le sue ali sono lineari, in accordo con il limite di Lee (pendenza asintotica della varianza totale al massimo 2).

Il fit segue Zeliade: fissati m e sigma, il problema diventa lineare nei restanti parametri (con vincoli di segno), quindi cerco solo su (m, sigma): una griglia 15×15 e poi un affinamento con Nelder-Mead sui tre punti migliori. Pesi: la vega di ogni quota, così le quote vicino all'ATM contano di più di quelle in coda. Servono almeno 5 quote per scadenza.

Controlli che il codice calcola e riporta:

- **rmse_iv**: errore del fit in punti di IV (qui tra 5 e 21 punti base).
- **Butterfly** (Durrleman, condizione g(k) ≥ 0, equivale a densità implicita non negativa): `g_dati` sul solo intervallo dei dati, `g_min` su [−1.5, 1.5]. Sui dati è soddisfatta per tutte le scadenze; fuori dai dati lo SVI sulle scadenze brevi la viola.
- **Calendar** (la varianza totale non deve diminuire con la scadenza): `calendar_min` guarda il minimo di w(T₂) − w(T₁) sull'intervallo di k coperto da entrambe le scadenze. In un'esecuzione di esempio vale +0.00014: positivo ma molto vicino a zero.
- **al_limite**: segnala le scadenze in cui una pendenza delle ali è sul limite di Lee (b(1 ± rho) ≤ 2). Succede sulle prime quattro scadenze (fino a 35 giorni), dove i dati sono su un solo lato dello smile e l'altra ala non è identificata.

## 8. La superficie (`surface.py`)

La classe `Superficie` mette insieme i fit: a un k fissato interpola linearmente la varianza totale tra le due scadenze vicine (interpolare in varianza totale e non in IV mantiene l'ordinamento delle varianze nel tempo, quindi niente arbitraggio calendar se le scadenze quotate lo rispettano; l'assenza di butterfly sulle curve interpolate è verificata numericamente nei test, non garantita in teoria); fuori dall'intervallo delle scadenze la IV resta costante. Il forward a una scadenza qualsiasi usa il carry r − q interpolato.

Scelta importante: ogni scadenza ricorda l'intervallo di k dei suoi dati e `iv(K, T)` restituisce **NaN** fuori da quel dominio, perché lì lo SVI è solo un'estrapolazione. Si può chiedere comunque l'estrapolazione con `fuori="estrapola"`, ma non è affidabile.

## 9. Il mondo sintetico (`heston.py`, `synthetic.py`)

Prima di passare ai dati reali volevo un posto dove la risposta giusta fosse nota, per controllare che tutti i pezzi (IV, SVI, hedging, statistica) funzionino. Uso il modello di Heston:

    dS = (r − q) S dt + sqrt(v) S dW1
    dv = kappa (theta − v) dt + xi sqrt(v) dW2,    corr(dW1, dW2) = rho

I prezzi europei vengono dalla funzione caratteristica (formula di Lewis), con il risultato limitato ai limiti di non arbitraggio per evitare errori numerici.

Il modello vive in due mondi collegati dal premio per il rischio di varianza `lam`: sotto **P** (il mondo reale) governa come si muove il titolo; sotto **Q** (il mondo di prezzo) governa i prezzi delle opzioni. Sotto Q la varianza torna alla media più lentamente e verso un livello più alto (kappa_Q = kappa − lam), quindi la volatilità implicita sta sopra quella realizzata: è la versione sintetica del premio che si osserva sul mercato. Parametri: r = 4%, q = 1.3%, mu = 10%, kappa = 4, theta = 0.0359, xi = 0.48, rho = −0.7, lam = 2. I percorsi si simulano con uno schema di Eulero a 10 sottopassi al giorno, troncando la varianza a zero.

## 10. Volatilità realizzata (`realized.py`)

Annualizzata con 252 giorni. Due stimatori:

- **Close-to-close**: radice di 252 volte la media dei rendimenti logaritmici al quadrato su n giorni. Semplice, ed è l'unico possibile sui dati sintetici (che hanno solo la chiusura).
- **Yang-Zhang**: usa apertura, massimo, minimo e chiusura, combinando il salto notturno, il movimento apertura-chiusura e lo stimatore di Rogers-Satchell, con il peso che minimizza la varianza. Lo uso come controllo sui dati reali.

## 11. Delta hedging (`hedge.py`, `backtest.py`)

L'operazione che testo: vendo una call at-the-forward al prezzo di Black-Scholes con la IV del giorno, e ogni giorno ribilancio le azioni con la delta di Black-Scholes calcolata con la stessa IV, fino a scadenza. Il P&L include interessi sulla cassa, dividendi e, se richiesti, costi proporzionali (in punti base del controvalore scambiato).

Con copertura continua il guadagno è

    P&L = ∫ e^{r(T−t)} · ½ Γ S² · (sigma_implicita² − sigma_realizzata(t)²) dt

cioè dipende solo da quanto la varianza realizzata sta sotto quella implicita. Con copertura giornaliera la stessa formula diventa una somma: `call_venduta_coperta` la calcola accanto al P&L vero (colonna `teoria`) e i test verificano che i due numeri siano molto correlati. È la ragione per cui questa strategia ha senso: il guadagno non dipende dalla direzione del mercato, ma dalla differenza tra implicita e realizzata.

Sul mondo sintetico (`backtest`) la IV di vendita è la IV ATM del modello e i percorsi sono quelli reali (sotto P). Per non risolvere Heston per ogni giorno e percorso, calcolo la IV ATM su una griglia di varianze e interpolo con una spline (`segnale.py`). Sui dati reali (`backtest_reale`) la IV di vendita è quella a 30 giorni del fornitore di quel giorno, con r = 3% e q = 1.3% costanti.

Conversione della base temporale. la IV a 30 giorni è annualizzata sul calendario (30/365 di anno), mentre il backtest e la vol realizzata lavorano in giorni di borsa (21/252 di anno). Le due durate non coincidono (30 giorni di calendario sono circa 20.6 giorni di borsa) e usare la IV tale e quale venderebbe l'opzione a una varianza totale dell'1.4% più alta di quella di mercato. `realized.iv_in_tempo_di_borsa` converte la IV conservando la varianza totale (fattore 0.9931, cioè circa −0.12 punti a IV 17%), e lo applico a IV e VIX prima di ogni confronto con la vol realizzata e prima del backtest.

## 12. Statistica: perché Newey-West (`stat.py`)

Il backtest apre una posizione ogni giorno, e ognuna dura 21 giorni di borsa: due operazioni consecutive condividono 20 giorni su 21. I risultati sono quindi fortemente correlati e l'errore standard classico, che li tratta come indipendenti, sottostima di molto l'incertezza (e gonfia la statistica t). Uso lo stimatore di Newey-West con pesi di Bartlett e 21 ritardi (`media_hac`). Nota: fino al 2023 le osservazioni della IV non sono giornaliere (lunedì, mercoledì e venerdì), quindi 21 ritardi in numero di osservazioni coprono più di 21 giorni di borsa: la sovrapposizione reale è di circa 9 osservazioni. Un numero di ritardi più grande del necessario tende a rendere l'errore standard più prudente, non meno, e preferisco questo a una scelta più stretta. Sul mondo sintetico invece le operazioni dello stesso percorso sono legate tra loro ma i percorsi sono indipendenti: faccio la media per percorso e calcolo l'errore standard tra percorsi (`media_con_ic`).

## 13. Dati storici

Le chain storiche non si scaricano da Yahoo. Uso il database pubblico DoltHub `post-no-preference/options`, che contiene la IV a 30 giorni (`iv_current`) e le quote di opzioni dal 2019. Il database pesa circa 8 GB, quindi lo clono una volta, ne esporto due file piccoli per SPY e poi lo posso cancellare:

```
dolt clone post-no-preference/options
cd options
dolt sql -r csv -q "select date, iv_current, hv_current from volatility_history where act_symbol='SPY' order by date" > vol_SPY.csv
dolt sql -r csv -q "select date, expiration, strike, call_put, bid, ask, vol from option_chain where act_symbol='SPY' order by date, expiration, strike" > chain_SPY.csv
```

Metto i due file in `data/dolthub/`. Su Windows PowerShell scrive i file con un BOM iniziale; `volsurf.dati.leggi` lo gestisce. Il file della serie di IV (`vol_SPY.csv`) è nel repository; il file delle quote (`chain_SPY.csv`, circa 9 MB) è escluso da git e serve solo a `controlla_iv.py`.

Limiti di questi dati, che pesano sull'interpretazione:

- Non sono chain giornaliere complete: fino al 2023 circa 150 giorni all'anno (lunedì, mercoledì, venerdì), poi 183 nel 2024 e 259 nel 2025; il 2019 ha solo 48 giorni.
- Ogni chain ha solo 3–4 scadenze, perciò non si può costruire una superficie storica: uso solo la IV a 30 giorni.
- Per le quote, qualche percento ha bid nullo.

Dopo il download calcolo la mia IV a 30 giorni su 62 giorni di prova (`controlla_iv.py`) e la confronto con quella del fornitore: la mia è più bassa di 0.73 punti in media (mediana 0.71), con correlazione 0.997. La differenza è sistematica, quindi non è rumore; non ne conosco la causa (convenzioni del fornitore, r e q costanti, scelta degli strike). È il motivo per cui nel backtest mostro la sensibilità a uno spostamento della IV.

**Fonti e uso dei dati.** Prezzi, VIX e chain provengono da Yahoo Finance (tramite `yfinance`), la curva dei tassi da FRED, le serie di IV da DoltHub (`post-no-preference/options`). I file in `data/` sono estratti piccoli inclusi a scopo didattico, per rendere riproducibile l'analisi storica; per un uso diverso vanno rispettati i termini delle singole fonti.

## 14. Risultati

Tutti i numeri e i grafici sono in `notebooks/analisi.ipynb`. In sintesi:

**Superficie (esecuzione di esempio)** (spot 769.64). IV ATM a 14 giorni 11.2%, a 30 giorni 12.5%, a 91 giorni 13.7%, a 182 giorni 14.7%, a 441 giorni 16.4%. Lo smile ha lo skew tipico di un indice azionario (IV più alta per gli strike bassi).

**Premio implicita − realizzata** (8 anni, errori Newey-West con 21 ritardi):

| confronto | media (punti di vol) | t |
|---|---|---|
| VIX − RV close-to-close (1989 giorni) | +3.54 | 4.6 |
| VIX − RV Yang-Zhang | +3.25 | 4.5 |
| VIX − RV, solo i 1157 giorni con IV del fornitore | +3.66 | 4.0 |
| IV 30 giorni − RV close-to-close (1157 giorni) | +0.42 | 0.5 |

Il VIX non è la IV di una call ATM: stima il prezzo di uno swap sulla varianza, che include gli strike bassi dove il premio di rischio è maggiore. Il rapporto IV/VIX ha mediana 0.83. Per questo il premio che conta per la mia strategia è l'ultimo, e non è significativo.

**Backtest su SPY** (call ATM a 21 giorni di borsa, copertura giornaliera, 1157 operazioni, P&L in % dello spot):

| caso | P&L medio | t |
|---|---|---|
| IV del fornitore (convertita), senza costi | +0.070% | 1.24 |
| con 1 punto base di costo | +0.045% | 0.79 |
| con 3 punti base | −0.006% | −0.11 |
| IV spostata di −0.73 punti | −0.014% | −0.26 |
| r = 0% / r = 5% (q = 1.3%) | +0.052% / +0.082% | 0.92 / 1.45 |
| q = 0% / q = 2% (r = 3%) | +0.078% / +0.066% | 1.38 / 1.17 |

Per anno: 2020 +0.02, 2021 +0.16, 2022 −0.13, 2023 +0.19, 2024 −0.03, 2025 +0.06, 2026 +0.21. Il segno non è stabile: l'anno peggiore è il 2022, e il 2020 è quello con la dispersione maggiore (deviazione standard 1.1%).

**Mondo sintetico** (300 percorsi). Con lam = 0 il premio è leggermente negativo (−0.02%, intervallo [−0.03, −0.01]), e cresce con lam: +0.06% con lam = 2 a 21 giorni. Ribilanciare meno spesso non cambia la media ma ne alza la dispersione (a 21 giorni la deviazione standard passa da 0.53 a 1.19 ribilanciando ogni 10 giorni invece che ogni giorno). Il premio sintetico a 21 giorni scompare con circa 2 punti base di costo. Il −0.02% a lam = 0 non è un errore del codice: è l'effetto della correlazione negativa fra prezzo e varianza (rho = −0.7), che rende la copertura di Black-Scholes imperfetta su un'opzione at-the-money; ponendo rho = 0 la media diventa −0.007% con intervallo [−0.02, +0.005], compatibile con zero.

**Conclusione.** Un premio medio positivo esiste in tutto il campione, ma è dello stesso ordine dell'incertezza sulla stessa IV (0.7 punti) e dei costi di esecuzione, e non è stabile negli anni. Non posso affermare che la strategia abbia un vantaggio.

## 15. Limiti

- Una sola superficie, quella dell'ultima seduta, senza validazione fuori campione; la storia ha solo la IV a 30 giorni, non superfici.
- r e q costanti nel backtest reale (3% e 1.3%). Ho provato r tra 0% e 5% e q tra 0% e 2%: la media cambia di al più 0.02 punti percentuali dello spot, sempre dentro l'errore standard (0.056). q nella superficie è un carry implicito per scadenza, non il dividend yield (sezione 6).
- Il prezzo di vendita nel backtest è teorico, Black-Scholes alla IV a 30 giorni del fornitore, non una quota realmente eseguibile; non c'è lo spread bid-ask dell'opzione. La IV del fornitore (`iv_current`) è un dato di cui non conosco il metodo di calcolo, e la mia IV differisce di 0.7 punti senza causa nota: non so quale delle due sia più vicina al prezzo di mercato.
- I costi di copertura sono un parametro, non una misura: 1 punto base del controvalore scambiato è prudente per SPY (lo spread tipico è di un centesimo su circa 770 dollari, cioè molto meno di 1 punto base), quindi il caso con 3 punti base è pessimistico.
- Esecuzione alla chiusura, copertura sempre giornaliera, nessun costo di prestito o margine.
- Un solo titolo (SPY) e un solo schema (ATM, 21 giorni); nessun test fuori campione. Ho guardato più varianti (costi, r, q, spostamenti della IV): i p-value non sono corretti per i confronti multipli, ma poiché nessun risultato è significativo la conclusione non ne risente.
- Il mondo sintetico è Heston senza salti e con opzioni europee; Durrleman è calcolato sulle IV americane come approssimazione.
- La superficie sulle scadenze fino a 35 giorni è poco affidabile (dati su un solo lato, ala al limite).

## 16. Test

`pytest` esegue 132 test (82 senza quelli segnati `slow`). Verificano tra l'altro: valori di riferimento, parità put-call e greche (contro differenze finite) di Black-Scholes; funzione caratteristica di Heston come martingala, limite verso Black-Scholes, limiti di non arbitraggio e confronto con Monte Carlo; albero americano contro Longstaff-Schwartz, convergenza nei passi e coincidenza con Black-Scholes dove l'esercizio anticipato non conviene; stima di q su chain costruite a q noto (dove la parity europea sbaglia); fit SVI che ritrova parametri noti, anche con rumore; assenza di arbitraggio calendar della superficie; coerenza tra P&L simulato e teoria gamma; errori Newey-West su serie con autocorrelazione nota. La CI su GitHub Actions li lancia a ogni push.
