# DFIG — Dispensa di laboratorio

> Materiale di supporto per il corso di Macchine Elettriche e Conversione dell'Energia.
> Accompagna il simulatore interattivo `dfig.py`.

---

## Indice

1. [Cos'è una DFIG e perché ci interessa](#1-cosè-una-dfig-e-perché-ci-interessa)
2. [Parametri della macchina simulata](#2-parametri-della-macchina-simulata)
3. [Modello in coordinate di fase: perché lo abbandoniamo](#3-modello-in-coordinate-di-fase-perché-lo-abbandoniamo)
4. [Trasformata di Clarke (`abc → αβ`)](#4-trasformata-di-clarke-abc--αβ)
5. [Trasformata di Park (`αβ → dq`)](#5-trasformata-di-park-αβ--dq)
6. [Modello DFIG nel riferimento sincrono](#6-modello-dfig-nel-riferimento-sincrono)
7. [Coppia elettromagnetica e bilancio meccanico](#7-coppia-elettromagnetica-e-bilancio-meccanico)
8. [Cosa fa il simulatore: mappa codice ↔ equazioni](#8-cosa-fa-il-simulatore-mappa-codice--equazioni)
9. [Lettura della GUI: piani d-q, gauge, diagnostica numerica](#9-lettura-della-gui-piani-d-q-gauge-diagnostica-numerica)
10. [Casi di studio guidati](#10-casi-di-studio-guidati)
11. [Sull'integrazione numerica](#11-sullintegrazione-numerica)
12. [Limiti del modello e cosa non vedi](#12-limiti-del-modello-e-cosa-non-vedi)
13. [Esercizi proposti](#13-esercizi-proposti)
14. [Bibliografia minima](#14-bibliografia-minima)

---

## 1. Cos'è una DFIG e perché ci interessa

DFIG sta per *Doubly-Fed Induction Generator* — generatore asincrono a doppia alimentazione. È
una macchina trifase a rotore avvolto in cui:

- **lo statore** è connesso direttamente alla rete (es. 690 V, 50 Hz);
- **il rotore** (avvolto, accessibile tramite anelli) è alimentato attraverso un convertitore
  elettronico **back-to-back** (rotor-side converter + grid-side converter, condivisi da un DC link).

Nel mondo reale è oggi la macchina dominante negli aerogeneratori a velocità variabile in classe
1.5–6 MW (≈ 50–60% del parco eolico installato). Il motivo è economico e di efficienza: il
convertitore deve dimensionare solo la potenza di slip (≈ ±30 % della potenza nominale)
invece dell'intera potenza, perché il **path principale di potenza è lo statore diretto-in-rete**.
Solo la quota legata allo slip transita per il convertitore.

```
            ┌──────────┐
   rete ────┤ statore  │
            │          │  ω_m  ← turbina
            │          │
            │  rotore  │ ←── tensione iniettata (V_r, f_r)
            └────┬─────┘
                 │ anelli
                 │
        ┌────────┴───────┐
        │   convertitore │
        │   back-to-back │
        └────────┬───────┘
                 │
                rete
```

**Punto chiave didattico**: agendo su modulo e fase della tensione iniettata al rotore, regoliamo
indipendentemente coppia (→ `P`) e magnetizzazione (→ `Q`), pur lavorando a velocità variabile.
Il simulatore vi permette di toccare con mano questo concetto senza dover costruire l'inverter.

---

## 2. Parametri della macchina simulata

| Simbolo | Valore | Unità | Significato |
|---:|---:|:---|:---|
| `R_s` | 4.8 | mΩ  | Resistenza di una fase di statore |
| `R_r` | 2.4 | mΩ  | Resistenza di una fase di rotore (riportata allo statore) |
| `L_s` | 6.8 | mH  | Induttanza propria di statore |
| `L_r` | 7.1 | mH  | Induttanza propria di rotore (riportata) |
| `M`   | 6.8 | mH  | Mutua statore–rotore (coefficiente di accoppiamento alto: `M ≈ √(L_s·L_r)`) |
| `n_p` | 3   | —   | Coppie polari (sincronismo a 1000 rpm meccanici @ 50 Hz) |
| `J`   | 50  | kg·m² | Momento d'inerzia complessivo (rotore + carico equivalente) |
| `b`   | 3.3 | N·m·s | Attrito viscoso meccanico equivalente |
| `V_n` | 690 | V (linea-linea, picco di fase ≈ 563 V) | Nominale |
| `S_n` | 1   | MVA | Potenza apparente nominale |

> Definiamo per comodità il *determinante induttivo*
> $D \equiv L_s L_r - M^2 = (6.8 \cdot 7.1 - 6.8^2)\cdot 10^{-6} \approx 2.04\cdot 10^{-6}$ H².
> Comparirà ovunque nelle inversioni `flusso → corrente`.

---

## 3. Modello in coordinate di fase: perché lo abbandoniamo

In coordinate naturali `a, b, c` (per statore e rotore: 6 fasi totali) le equazioni sono:

$$
\mathbf{v}_s = R_s\,\mathbf{i}_s + \frac{d\boldsymbol{\psi}_s}{dt}, \qquad
\mathbf{v}_r = R_r\,\mathbf{i}_r + \frac{d\boldsymbol{\psi}_r}{dt}
$$

con flussi concatenati legati alle correnti da

$$
\begin{bmatrix}\boldsymbol\psi_s\\ \boldsymbol\psi_r\end{bmatrix}
= \begin{bmatrix} L_{ss} & M_{sr}(\theta_e) \\ M_{rs}(\theta_e) & L_{rr}\end{bmatrix}
\begin{bmatrix}\mathbf{i}_s\\ \mathbf{i}_r\end{bmatrix}
$$

Il problema è che $M_{sr}(\theta_e) = M\,\mathbf{C}(\theta_e)$ dipende dall'angolo elettrico
$\theta_e = n_p\,\theta_m$. Le induttanze mutue si modificano sinusoidalmente mentre il rotore gira:
**il sistema è lineare ma a coefficienti tempo-varianti**. Disastroso da simulare e ancora peggio
da analizzare per il controllo.

La trasformata di Park (preceduta da Clarke) cancella esattamente questa dipendenza dall'angolo,
trasformando il modello in **lineare a coefficienti costanti** rispetto al riferimento scelto.

---

## 4. Trasformata di Clarke (`abc → αβ`)

Date tre grandezze trifase bilanciate ($x_a + x_b + x_c = 0$), Clarke proietta le tre fasi sui
**due assi ortogonali** $\alpha,\beta$ del piano:

$$
\begin{bmatrix} x_\alpha \\ x_\beta \\ x_0 \end{bmatrix}
= \frac{2}{3}
\begin{bmatrix} 1 & -\tfrac12 & -\tfrac12 \\ 0 & \tfrac{\sqrt 3}{2} & -\tfrac{\sqrt 3}{2} \\ \tfrac12 & \tfrac12 & \tfrac12\end{bmatrix}
\begin{bmatrix} x_a \\ x_b \\ x_c \end{bmatrix}
$$

Il coefficiente `2/3` è la convenzione *amplitude-invariant*: $|x_{\alpha\beta}| = |x_a|$ in regime
sinusoidale. La componente $x_0$ (omopolare) si annulla per terne bilanciate.

**Interpretazione geometrica**. Tre vettori unitari sfasati di 120° individuano tre direzioni nel
piano; sommando vettorialmente i contributi $x_a, x_b, x_c$ ottieni un singolo vettore in $\alpha\beta$,
che ruota a velocità angolare $\omega$ se le tre fasi sono sinusoidi a quella frequenza. Da tre
sinusoidi siamo passati a un **fasore rotante** in 2D.

In simulazione partiamo già da $\alpha\beta$ implicito: il simulatore non genera mai la terna
`a, b, c`, lavora direttamente con i fasori spaziali. Niente di perso, perché le grandezze sono
ricostruibili invertendo Clarke.

---

## 5. Trasformata di Park (`αβ → dq`)

Park è una **rotazione** di un angolo $\theta$ del piano $\alpha\beta$:

$$
\begin{bmatrix} x_d \\ x_q \end{bmatrix}
= \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix}
\begin{bmatrix} x_\alpha \\ x_\beta \end{bmatrix}
$$

Se $\theta = \omega_s\,t$, ovvero ruotiamo a velocità angolare $\omega_s$ dello stesso fasore, allora
**un fasore che ruotava a $\omega_s$ in $\alpha\beta$ diventa fermo in `dq`**. È la magia: in regime
le sinusoidi diventano costanti.

Tre scelte tipiche del riferimento:

| Riferimento | $\theta$ | Quando è utile |
|---|---|---|
| Stazionario ($\alpha\beta$) | $\theta = 0$ | Analisi a piccoli segnali, FOC scalar |
| **Sincrono di rete** | $\theta = \omega_s t$ | DFIG, FOC, controllo P/Q indipendenti |
| Rotorico (rotor-flux) | $\theta = \omega_s t - n_p\theta_m$ | Macchine asincrone con orientamento di flusso rotorico |

Il simulatore usa il **sincrono di rete**: tutte le `d, q` di statore/rotore sono valutate in un
sistema che ruota a $\omega_s$ rispetto allo statore fisico. Significa che, in regime, i due piani
`(d,q)` mostrati nella GUI restano **stazionari** (i vettori non ruotano: si fermano su un punto).

**Chiarimento sulla tensione rotorica**. Il convertitore lato rotore inietta nelle terne fisiche di
rotore. Per portarla nel riferimento sincrono dobbiamo ruotare di $\omega_s\,t - n_p\theta_m$. Se la
tensione iniettata ha modulo $V_r$ e pulsazione $\omega_r$ nel riferimento di rotore, l'angolo
risultante visto in `dq` sincroni è

$$
a(t) = \omega_r t - \omega_s t + n_p\theta_m(t)
$$

ed è esattamente quello che troviamo nella riga 49 del codice:

```python
a = wr * t - ws * t + NP * thm
vrd, vrq = Vr * cos(a), Vr * sin(a)
```

> 💡 **Insight**. Per ottenere un punto di lavoro stazionario nel sincrono basta che $a$ sia
> costante, ovvero $\omega_r = \omega_s - n_p\omega_m = \omega_{sl}$. È esattamente la pulsazione
> di slip: il convertitore deve iniettare a $f_{slip}$ per "bloccare" il vettore di tensione
> rotorica in `dq`. Con `f_r = f_slip` la macchina ha un punto di equilibrio; nello slider `f_r`
> imposta intuitivamente questo a mano per vedere quando la macchina entra in regime stazionario.

---

## 6. Modello DFIG nel riferimento sincrono

Scriviamo le quattro equazioni di tensione (statore d/q, rotore d/q) e l'equazione meccanica.
Scegliamo come **variabili di stato i flussi concatenati**, perché rendono il sistema diagonale
nelle derivate temporali.

### 6.1 Equazioni di tensione

$$
\begin{aligned}
\dot\psi_{sd} &= v_{sd} - R_s i_{sd} + \omega_s\,\psi_{sq}\\
\dot\psi_{sq} &= v_{sq} - R_s i_{sq} - \omega_s\,\psi_{sd}\\
\dot\psi_{rd} &= v_{rd} - R_r i_{rd} + \omega_{sl}\,\psi_{rq}\\
\dot\psi_{rq} &= v_{rq} - R_r i_{rq} - \omega_{sl}\,\psi_{rd}
\end{aligned}
$$

con $\omega_{sl} \equiv \omega_s - n_p\,\omega_m$ (pulsazione di slip elettrica).

I termini $\pm\omega \psi$ sono i contributi della **rotazione del riferimento**: quando ruoti il
sistema di riferimento alla velocità $\omega$, una grandezza ferma nel laboratorio appare cambiare
proiezione lungo `d, q`. Sono spesso detti *back-EMF* o *cross-coupling terms*.

> Convenzione nel simulatore: scegliamo l'asse `d` allineato con la **tensione di rete** di statore,
> quindi $v_{sd} = V_s$ e $v_{sq} = 0$. È la convenzione *grid-voltage-oriented*, la più diffusa
> nel controllo DFIG.

### 6.2 Relazioni flusso ↔ corrente

Le relazioni costitutive (matrice di induttanze nel sincrono) sono **indipendenti dall'angolo**:

$$
\begin{bmatrix}\psi_{sd}\\ \psi_{sq}\\ \psi_{rd}\\ \psi_{rq}\end{bmatrix} =
\begin{bmatrix}
L_s & 0 & M & 0\\
0 & L_s & 0 & M\\
M & 0 & L_r & 0\\
0 & M & 0 & L_r
\end{bmatrix}
\begin{bmatrix} i_{sd}\\ i_{sq}\\ i_{rd}\\ i_{rq}\end{bmatrix}
$$

Invertendo (asse per asse, perché d e q non si mescolano):

$$
\boxed{\;\;
\begin{aligned}
i_{sd} &= \frac{L_r\,\psi_{sd} - M\,\psi_{rd}}{D}\\
i_{sq} &= \frac{L_r\,\psi_{sq} - M\,\psi_{rq}}{D}\\
i_{rd} &= \frac{L_s\,\psi_{rd} - M\,\psi_{sd}}{D}\\
i_{rq} &= \frac{L_s\,\psi_{rq} - M\,\psi_{sq}}{D}
\end{aligned}\;\;}\qquad D = L_s L_r - M^2
$$

Trovi questo blocco nelle righe 51-54 del codice. È **la** formula da memorizzare.

> 💡 Il fatto che $D \approx 2\cdot 10^{-6}$ H² (cinque ordini di grandezza meno di $L_s L_r$) ci
> dice che l'accoppiamento $M$ è **strettissimo** — il flusso disperso è poca cosa. È il caso
> tipico delle DFIG: `M ≈ √(L_s L_r)` quasi unitario.

### 6.3 Forma di stato finale

Sostituendo le correnti nelle equazioni di tensione otteniamo un sistema ODE in `(ψ_sd, ψ_sq, ψ_rd, ψ_rq)` con
matrice di stato dipendente da $\omega_m$ (tramite $\omega_{sl}$). È **non lineare** solo a causa
dell'accoppiamento elettromeccanico (l'eq. di $\omega_m$ sotto), non per la trasformata.

---

## 7. Coppia elettromagnetica e bilancio meccanico

### 7.1 Coppia EM

Dal lavoro virtuale o dalla potenza di traferro si arriva a (forma cross-product nel piano `dq`):

$$
C_e \;=\; n_p\,M\,\bigl(i_{sq}\,i_{rd} - i_{sd}\,i_{rq}\bigr)
$$

> **Lettura**: la coppia è il prodotto vettoriale tra il fasore di corrente di statore e quello di
> rotore, scalato dal coefficiente di accoppiamento $M$. Massima per fasori in quadratura.

Espressione equivalente in termini di flusso e corrente: $C_e = \tfrac{3}{2}n_p\,(\psi_{sd}i_{sq} -
\psi_{sq}i_{sd})$ (dipende dalla normalizzazione di Clarke; quella usata qui non ha il $3/2$
perché lavoriamo già con grandezze "fasoriali" amplitude-invariant).

### 7.2 Equazione meccanica

$$
J\,\dot\omega_m = C_e - b\,\omega_m - C_l, \qquad \dot\theta_m = \omega_m
$$

`J` integra la coppia netta; `b` rappresenta tutte le perdite proporzionali alla velocità (ventilazione,
attrito viscoso, perdite di rame *aggiuntive* non incluse altrove); `C_l` è la coppia di carico imposta
dall'esterno (slider `C_load` nella GUI). Convenzione: `C_l > 0` significa coppia *resistente* (motore),
`C_l < 0` significa coppia *motrice* esterna (es. una turbina che spinge — funzionamento
generatorico).

---

## 8. Cosa fa il simulatore: mappa codice ↔ equazioni

Lo stato è un vettore `s ∈ ℝ⁷`:

| `s[i]` | Variabile | Unità |
|---:|---|:---|
| `s[0]` | $\psi_{sd}$ | Wb |
| `s[1]` | $\psi_{sq}$ | Wb |
| `s[2]` | $\psi_{rd}$ | Wb |
| `s[3]` | $\psi_{rq}$ | Wb |
| `s[4]` | $\omega_m$ | rad/s meccanici |
| `s[5]` | $\theta_m$ | rad meccanici |
| `s[6]` | $t$ | s |

La funzione `deriv(s, V_s, ω_s, V_r, ω_r, C_l, out)` (riga 46–67) calcola le sette derivate
$\dot s_i$ implementando esattamente le equazioni di §6 e §7. Nient'altro.

Il loop di integrazione vive dentro `advance_rk45(...)` (riga 137 e seguenti), un Dormand-Prince
5(4) classico con error control e step-size adattativo. La funzione è compilata JIT da
**Numba**: il loop interno gira a velocità nativa C, senza overhead Python. Il simulatore esegue
~2 milioni di step/s di RK45 su singolo core moderno.

### 8.1 Architettura runtime

```
┌──────────────────────┐         ┌─────────────────────────┐
│  Sim thread          │         │  GUI thread (main loop) │
│ (Numba @njit, nogil) │         │  (GTK4 + Cairo)         │
│                      │         │                         │
│  while True:         │         │  add_tick_callback:     │
│    snapshot(ctrl)    │         │    snapshot(state)      │
│    advance_rk45(...) │ ─lock─→ │    redraw plot          │
│    push history      │         │    update labels        │
│    sleep_to_wall()   │         │                         │
└──────────────────────┘         └─────────────────────────┘
       fa il lavoro                  vede il lavoro
```

Le due esecuzioni sono disaccoppiate: il sim thread può ciclare a 200 kHz mentre il rendering gira
a 120 Hz (vsync del monitor). La comunicazione è una `threading.Lock` su due strutture:
`state` (snapshot dello stato corrente) e `hist` (deque sotto-campionate ogni 5 ms simulati).

### 8.2 Real-time lock e diagnostica

Il sim thread *aspetta* il wall clock per non correre più veloce della realtà. La metrica nell'header
"**RT factor**" è il rapporto vero `sim_time / wall_time`: in regime di lock, deve stare a 1.00. La
metrica `headroom` (in DIAGNOSTICA NUMERICA) è invece quanto la sim girerebbe se non fosse limitata —
oggi tipicamente 100–200×, ovvero **abbiamo da spendere 100× la CPU che ci serve per il real-time**.
Quel margine viene usato per sceglie `rtol=1e-9` invece di tolleranze grossolane.

---

## 9. Lettura della GUI: piani d-q, gauge, diagnostica numerica

### 9.1 Piani `(d, q)`

I due piani d-q (correnti e flussi) mostrano la traiettoria del fasore spaziale negli **assi
sincroni**. In regime stazionario il pallino non ruota: si ferma su un punto. Le tracce in
trasparenza sono la "scia" recente — sono utili per leggere transienti e oscillazioni elettriche.

- Se l'asse **d** è allineato con la tensione di statore, allora:
  - $i_{sd}$ è proporzionale alla **potenza attiva** $P_s = V_s\,i_{sd}$
  - $i_{sq}$ è proporzionale alla **potenza reattiva** $Q_s = -V_s\,i_{sq}$
- Per il fasore di flusso di statore, in regime senza resistenza, $\psi_{sq} \approx -V_s/\omega_s$
  e $\psi_{sd} \approx 0$: il fasore di flusso è 90° in ritardo rispetto alla tensione.

### 9.2 Sezione SEGNALI

14 stati primari (correnti d/q, flussi d/q, ω_m, C_em, P/Q/Pr, slip) con valore istantaneo e
checkbox per assegnarli ai 4 plot temporali. Sotto, i 5 derivati `|i_s|, |i_r|, |φ_s|, |φ_r|, f_slip`
non sono plottabili (sono moduli, non grandezze loggate) ma si leggono in tempo reale.

### 9.3 DIAGNOSTICA NUMERICA

| Voce | Lettura |
|:---|:---|
| `RT factor` | sim/wall reale. Deve stare a 1.00× con il lock attivo. |
| `lag` | quanto stiamo indietro rispetto al wall (ms). Verde sotto 5 ms. |
| `headroom` | margine CPU. >>1 vuol dire CPU ampiamente disponibile. |
| `step/s` | quante chiamate `deriv()` al secondo. |
| `dt_eff` | passo medio scelto da RK45 (μs). Crolla in transitorio, sale in regime. |
| `iter wall` | ms wall per iterazione del sim thread. |
| `step rejet.` | percentuale di step rifiutati dal controllo errore. Se >10% c'è qualcosa di strano. |
| `‖err‖∞` | norma infinity dell'errore stimato per step (scalata su rtol). |
| `resync` | quante volte il lock real-time è stato riazzerato. >0 = abbiamo perso terreno. |
| `UI fps` | frame al secondo del rendering. |

---

## 10. Casi di studio guidati

Tutti partono dal default (`V_s = 690 V`, `f_s = 50 Hz`, `V_r = 0`, `f_r = 0`, `C_load = 0`).

### 10.1 Motore asincrono semplice (rotore in corto)

Configurazione default: rotore in corto (`V_r = 0`), nessun carico. Premi RUN e osserva:

- `ω_m → 1000 rpm` ma non esattamente: `slip ≈ 0.018%` permanente. È il piccolo slip necessario
  a generare una coppia EM che vinca l'attrito viscoso `b·ω_m`. Senza slip, non c'è coppia.
- `P_s ≈ 36 kW` assorbiti: sono le perdite ($P_{Rs} + P_{Rr} + b\,\omega_m^2$).
- `Q_s ≈ 222 kVAR` assorbiti: questa è la **magnetizzazione** della macchina. Una IM/DFIG senza
  rotore alimentato è un consumatore di reattivo.
- `C_em ≈ 345 N·m`: pareggia $b\cdot\omega_m = 3.3\cdot 104.7$.

Esercizio: aumenta `C_load` a 5000 N·m e osserva slip e potenze.

### 10.2 Funzionamento generatorico subsincrono

Imposta `C_load = -5000 N·m` (turbina che spinge in modo modesto).

- $\omega_m$ sale poco sopra il sincronismo? **No**, scende sotto. Qual è il punto di equilibrio?
  Soluzione: ${C_e} = b\omega_m + C_l$ con $C_l < 0$ richiede $C_e < b\omega_m$, ovvero coppia EM
  positiva ridotta → slip più piccolo o negativo a seconda dei segni.
- $P_s$ diventa **negativa**: la macchina cede potenza alla rete dallo statore.

### 10.3 DFIG vera: iniezione al rotore

Imposta `V_r = 50 V`, `f_r = 5 Hz`, `C_load = -10000 N·m`.

Domanda da porre agli studenti: **quale `f_r` rende il punto di lavoro stazionario nel sincrono?**
Risposta attesa: $f_r = (\omega_s - n_p\omega_m)/(2\pi) = f_{slip}$. Confronta col valore mostrato
nel pannello ROTORE (`f_slip`). Se imposti `f_r ≠ f_slip`, le grandezze nel sincrono iniziano a
oscillare a $f_r - f_{slip}$ (non si fermano nei piani d-q). Confronta visivamente.

### 10.4 Iniezione e fasore di tensione rotorica

Mantieni `f_r ≈ f_slip`. Variando solo l'**ampiezza** $V_r$ e tenendo $f_r$ fissa, sposti il punto
di lavoro lungo una direzione precisa nei piani d-q (la direzione dipende dalla fase di iniezione,
qui implicitamente data dalla scelta `vrd = V_r cos a, vrq = V_r sin a`). In una DFIG reale il
controllore vector-control modula sia ampiezza che **fase relativa** della tensione di iniezione
per gestire indipendentemente $C_e$ e $Q_s$.

### 10.5 Transitorio di tensione di rete (LVRT)

Riduci di colpo `V_s` da 690 a 200 V (simulando un buco di rete). Cosa succede?

- I flussi statorici (che erano "agganciati" a `V_s/ω_s`) non possono cambiare istantaneamente:
  generano una componente DC transitoria che gira nel sistema rotorico inducendo correnti elevate.
- La coppia EM oscilla violentemente.
- `dt_eff` nella diagnostica crolla a pochi μs: RK45 sta lavorando duro per tracciare il transitorio.
- `‖err‖∞` resta sotto 1 (errore controllato), `step rejet.` può salire al 5–10% nei primi
  millisecondi.

È una buona occasione per discutere di *crowbar* e *chopper* (protezioni reali del convertitore)
e dei requisiti grid-code di Low Voltage Ride Through.

---

## 11. Sull'integrazione numerica

### 11.1 Perché RK45 adattativo

Il sistema ha tempi caratteristici molto diversi:

| Modo | Costante di tempo | Note |
|:---|:---|:---|
| Elettrico (sincrono) | $1/\omega_s \approx 3$ ms | quasi-tempo di rotazione |
| Elettrico (smorzamento) | $L/R \approx 1$ s | smorzamento dei transitori di flusso |
| Meccanico | $J/b \approx 15$ s | inerzia / attrito |

In regime stazionario tutto è praticamente fermo nel sincrono → un integratore rigido può fare
passi grossi (fino a $dt_{max}$). Durante un transitorio rapido (es. step di tensione),
$\omega_s\Delta t$ deve restare piccolo per non perdere la dinamica → passi micro-secondo.

Un integratore a passo fisso (Eulero/RK4) ti costringe a scegliere il **peggior caso**: passi
piccolissimi sempre, sprecando CPU. Adattativo (RK45 Dormand-Prince) sceglie il passo in base
all'errore stimato per step: lavora di fino solo dove serve.

### 11.2 La trasformazione di Park come "demodulazione"

Un punto pedagogico spesso sottovalutato: lavorare nel sincrono è **anche un trick numerico**.
In `αβ` (stazionario) le grandezze sono sinusoidi a 50 Hz, quindi $\dot{x} = \omega \cdot x_\perp$
è grande; servirebbero passi <1 ms per non distorcerlo. Nel sincrono le grandezze sono costanti in
regime, quindi $\dot{x}$ è prossimo a zero e l'integratore vola. Park *demodula* la portante a 50 Hz.

### 11.3 Conservazione di flussi e energia

Il modello **non** è simplettico: non conserva esattamente energia. Ma con `rtol = 1e-9` la deriva
è invisibile su scale di ore di simulazione. Verifica empirica: con rotore in corto e `C_load = 0`,
osserva $\omega_m$ dopo 60 s; non drifta.

---

## 12. Limiti del modello e cosa non vedi

| Aspetto reale | Nel simulatore |
|:---|:---|
| Saturazione magnetica | Lineare (M, L_s, L_r costanti) |
| Armoniche di tensione (PWM) | Tensioni puramente sinusoidali |
| Distribuzione spaziale degli avvolgimenti | Sinusoidale ideale |
| Effetto pelle nelle barre di rotore | Resistenze costanti |
| Termica e variazione delle resistenze | Niente |
| Inerzia distribuita / torsione albero | Massa concentrata |
| Squilibri di rete | Trifase puramente bilanciata |
| Dinamica del DC link, del filtro LCL, del chopper | Assenti |
| Perdite nel ferro | Confluite in `b` (modo grossolano) |

In un corso di seconda laurea questi punti sono lo spunto naturale per estensioni del modello.
Nessuno di questi rende il modello *sbagliato*: lo rende **adeguato per studiare fenomeni
elettro-meccanici a banda larga** (≈ 0–500 Hz), inadeguato per studi armonici o di EMI.

---

## 13. Esercizi proposti

1. **Punto di lavoro analitico**. Con `V_s = 690 V, f_s = 50 Hz, V_r = 0, C_l = 0`, ricavare
   analiticamente il valore di $\omega_m$, $i_{sd}$, $i_{sq}$ in regime, e verificare con la
   simulazione (i valori istantanei nel pannello SEGNALI). Suggerimento: imporre $\dot\psi = 0$
   nelle 4 eq. di tensione e $\dot\omega_m = 0$ nell'eq. meccanica, ottenendo un sistema algebrico
   di 5 equazioni in 5 incognite.

2. **Magnetizzazione e $Q_s$**. Mostrare che, con rotore in corto e trascurando le resistenze,
   $Q_s \approx V_s^2/(\omega_s L_s)$. Confrontare il valore col simulatore (≈ 222 kVAR per default).
   Discutere come cambia se il rotore *fornisce* magnetizzazione (immettendo $i_{rd}$).

3. **Slittamento e potenza di slip**. In regime stazionario, mostrare che la potenza che attraversa
   il rotore (verso il convertitore) è $P_r = -s\cdot P_{airgap}$ con $s = (\omega_s - n_p\omega_m)/\omega_s$.
   Verificare il segno e l'ordine di grandezza nel simulatore con punti di lavoro super-sincroni.

4. **Stima dei tempi caratteristici**. Calcolare le costanti di tempo statoriche $\tau_s = L_s/R_s$
   e rotoriche $\tau_r = L_r/R_r$. Indurre uno step di `V_s` di pochi volt in più rispetto al
   nominale, e misurare graficamente il tempo di assestamento di $i_{sd}$. Confrontare.

5. **Sensitività al passo**. Cambiare `rtol` da `1e-9` a `1e-4` (modificando il default in
   `SimEngine.__init__`) e osservare nei plot se appaiono differenze visibili. Nei piani d-q?
   In `‖err‖∞`?

6. **Integratore equivalente**. Implementare RK4 a passo fisso (`integrator = INT_RK4`,
   `logdt = -4`) e quantificare la differenza di costo CPU rispetto a RK45 adattativo. È il
   trade-off classico accuracy/cost.

7. **DFIG come VAR compensator**. Imporre $C_e = 0$ regolando $V_r, f_r$. La macchina può fornire
   reattivo *senza* assorbire o erogare attivo? Se sì, è un STATCOM rotante.

---

## 14. Bibliografia minima

- B.K. Bose, *Modern Power Electronics and AC Drives*, Prentice Hall, 2002 — capitoli 2 e 8 sul
  modello dq e il vector control.
- N. Mohan, *Advanced Electric Drives*, Wiley, 2014 — derivazione pulita di Clarke/Park e dei
  modelli d-q.
- O. Anaya-Lara, N. Jenkins et al., *Wind Energy Generation: Modelling and Control*, Wiley, 2009 —
  capitolo 4 dedicato alla DFIG.
- W. Leonhard, *Control of Electrical Drives*, Springer, 2001 — il classico, denso ma definitivo.
- M.Y. Worku, *Doubly Fed Induction Generator-Based Wind Turbines*, IntechOpen, 2020 — open access,
  utile per panoramica moderna.

---

> **Disclaimer didattico**. Il simulatore è uno strumento di laboratorio, non un sostituto della
> derivazione su carta delle equazioni. La prima cosa che si chiede agli studenti è di *ritrovare*
> ogni numero che compare nella GUI con un calcolo a mano, prima di interrogare la macchina.
> Solo allora la simulazione smette di essere una scatola nera e diventa un **microscopio** sul
> modello.
