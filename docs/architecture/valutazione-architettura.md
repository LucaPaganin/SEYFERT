# Valutazione architetturale del backend SEYFERT

> Stato analizzato: `master` @ `330fac3` (v1.3.8) — package `seyfert/` (~14.3k righe Python), `bin/`, `tests/`.

## 1. Che tipo di software è

SEYFERT è una **pipeline di calcolo scientifico batch**:

```
config JSON ──► parametri fisici ──► cosmologia + P(k) (CAMB/CLASS)
            ──► densità n(z) ──► funzioni peso ──► C_ℓ fiduciali
            ──► C_ℓ variati ──► derivate dC_ℓ/dθ ──► ΔC_ℓ (covarianza)
            ──► matrici di Fisher ──► risultati finali / tabelle / plot
```

Caratteristiche che contano per la scelta del pattern:

| Aspetto | Situazione | Implicazione |
|---|---|---|
| Dominio | Matematica/fisica (numpy), molto ricco e stabile | Il dominio va protetto: è il valore del software |
| Sistemi esterni | CAMB, CLASS, HDF5, JSON, pickle, file di riferimento IST, scheduler LSF (`bsub`), filesystem del run | **Molti "bordi" intercambiabili** |
| Punti d'ingresso | 17 script in `bin/`, GUI PyQt, notebook, inizio di app Streamlit | **Molti "driver"** dello stesso caso d'uso |
| Persistenza | Cache su disco a step (se il file esiste lo ricarica) | Il "repository" è il run directory |
| Team / ciclo di vita | Progetto di ricerca, pochi sviluppatori | Niente cerimonie enterprise |

## 2. Quale pattern: esagonale sì, ma "lite"

**Verdetto: l'architettura esagonale (Ports & Adapters) non è overkill qui — ma la versione completa sì.**

Perché ha senso: l'esagonale serve quando hai un nucleo da proteggere *e* tanti adattatori su entrambi i lati.
SEYFERT ha esattamente questo profilo: 2 solver Boltzmann, 3+ formati di storage, 2 modalità di esecuzione
(interattiva / LSF), 4 interfacce utente. Oggi ogni cambiamento su un bordo (es. passare da LSF a SLURM,
aggiungere un emulatore di P(k), sostituire pickle) tocca il dominio.

Cosa invece è overkill e **sconsiglio**:

- DTO separati per ogni layer e mapping manuale ovunque → gli `np.ndarray` sono già un ottimo "linguaggio comune".
- Un use case-classe per ogni micro-operazione, command bus, container di Dependency Injection.
- Interfacce per librerie numeriche stabili (numpy, scipy, numba): non sono "bordi", sono parte del linguaggio del dominio.

Il pattern raccomandato è quindi **"Functional core, imperative shell" organizzato come esagonale leggero**:

- **Core** puro: funzioni e classi che prendono array/valori e restituiscono array/valori. Zero I/O.
- **Pochi port** (4) solo dove c'è davvero variabilità o I/O.
- **Un solo application layer** con una pipeline esplicita.
- **Adapter** per tutto il resto.

## 3. Score di manutenibilità e robustezza ai cambiamenti

Scala 0–10, misurata *rispetto al pattern target sopra*.

| # | Criterio | Peso | Score | Evidenze principali |
|---|---|---|---|---|
| 1 | Isolamento del dominio dall'I/O | 20% | **3** | 13 moduli espongono `fromHDF5`/`saveToHDF5`; `Cosmology`, `PowerSpectrum`, `c_ells`, `bias`, `weight_functions`… importano `AbstractH5FileIO`. Merito parziale: i mapper `H5Cl`, `H5ClCollection` ecc. sono già classi separate. |
| 2 | Direzione delle dipendenze | 15% | **3** | Il dominio importa infrastruttura e applicazione: `cosmology/*` → `config.main_config`, `file_io`, `utils.filesystem_utils`; `fisher/delta_cl.py`, `derivatives/*` → `utils.workspace.WorkSpace`; `c_ells.py` importa matplotlib (`plot_utils.cl_plot`). `utils` è un hub con 92 import entranti. |
| 3 | Astrazione dei sistemi esterni | 15% | **5** | Buono: `ExternalBoltzmannSolver` (ABC) con `CAMBBoltzmannSolver`/`CLASSBoltzmannSolver` è *già* un port+adapter. Ma la scelta è un `if/elif` dentro `PowerSpectrum.evaluateLinearAndNonLinearPowerSpectra` (`power_spectrum.py:75`), gli import falliti di camb/classy sono solo loggati, e `BsubInterface` è usata direttamente senza astrazione. |
| 4 | Application layer / casi d'uso chiari | 15% | **3** | Tre orchestratori paralleli: `main/seyfert_core.do_forecast`, la classe `seyfert/seyfert.py::Seyfert` (con `prepareWorkspace` copiato quasi riga per riga da `prepare_workspace`), e `main/task_runners.py` + `orchestration.py` per il flusso a job. Input come `args_dict` non tipizzato. |
| 5 | Configurazione | 10% | **4** | Esistono classi di config tipizzate (bene), ma leggono file nel costruttore, hanno default hardcoded (`fsu.config_files_dir()`, `basic_forecast_config.json`), e file "nascosti" letti dal dominio (`selected_datavectors.txt`, `scaledmeanlum-E2Sa.dat`). |
| 6 | Testabilità | 15% | **5** | 26 file di test (~2.2k righe) sulla matematica: buona base. Però quasi nessun test usa fake/mock (solo 2 file), perché il dominio non si può istanziare senza file o CAMB; nessun test della pipeline end-to-end se non l'integration test via Docker. |
| 7 | Igiene / robustezza | 10% | **4** | Cache delle derivate in `pickle` (fragile tra versioni, non sicura); cache "esiste il file → riusa" senza verifica della config (rischio risultati stantii); `from sympy import comp` inutilizzato in `cl_derivative_core.py` con sympy non dichiarato (funziona solo come dipendenza transitiva); `mytest_debug/` e `.DS_Store` versionati; dipendenze runtime e dev mischiate (pytest, jupyterlab, PyQt5, streamlit tutte obbligatorie). |

**Score complessivo pesato: ≈ 3.8 / 10**

Lettura: il **dominio scientifico è di buona qualità** (classi coese, docstring, tipizzazione, test numerici),
il problema è l'**architettura attorno**: I/O e orchestrazione sono "spalmati" dentro il dominio,
quindi la robustezza ai cambiamenti di infrastruttura è bassa. È un caso tipico in cui il refactoring
architetturale rende molto con poco rischio, perché la matematica non va toccata.

### Scenari di cambiamento (stress test)

| Cambiamento plausibile | Oggi | Con il pattern target |
|---|---|---|
| Aggiungere un emulatore di P(k) (es. CosmoPower) | Modificare `PowerSpectrum` (if/elif) + `Cosmology` | Nuovo adapter `PowerSpectrumProvider`, registrato nel composition root |
| Passare da LSF a SLURM | Toccare `bsub_utils`, `task_runners`, `seyfert_core`, CLI | Nuovo adapter `JobExecutor` |
| Sostituire pickle con HDF5 per le derivate | Toccare `seyfert_core`, `cl_derivative`, loader | Solo l'adapter del repository |
| Esporre la pipeline in Streamlit/GUI | Ri-scrivere orchestrazione (terza/quarta copia) | Chiamare lo stesso use case `RunForecast` |
| Testare il calcolo Fisher senza CAMB né disco | Difficile | Fake in-memory dei port |

## 4. Come dovrebbe essere: panoramica target

### 4.1 Regola delle dipendenze

```
          driving adapters                         driven adapters
   (CLI, GUI PyQt, Streamlit, notebook)    (CAMB, CLASS, HDF5, LSF, file IST)
                 │                                    ▲
                 ▼                                    │ implementano
        ┌────────────────── application ──────────────┴───┐
        │  use case / pipeline   ──usa──►   ports (Protocol) │
        └──────────────────────┬───────────────────────────┘
                               ▼
                        ┌── domain ──┐
                        │ numpy puro │   (nessun import da application/adapters)
                        └────────────┘
```

Le frecce vanno **solo verso l'interno**. Il dominio non sa che esistono file, HDF5, CAMB o LSF.

### 4.2 Struttura di package proposta

```
seyfert/
├── domain/                      # core puro: input array → output array
│   ├── parameters.py            # PhysicalParameter, PhysicalParametersCollection
│   ├── cosmology.py             # E(z), r(z), check parametri (riceve P(k) già calcolato)
│   ├── power_spectrum.py        # dataclass PowerSpectrum(k, z, p_lin, p_nonlin)
│   ├── redshift_density.py
│   ├── bias.py, weight_functions.py, kernel_functions.py
│   ├── c_ells.py                # calcolo C_ℓ (senza plot, senza HDF5)
│   ├── derivatives.py           # SteM differentiator
│   ├── covariance.py            # ΔC_ℓ
│   ├── fisher.py                # FisherMatrix, datavector, linalg
│   └── numeric/                 # integrazione, utilità numeriche
│
├── application/
│   ├── ports.py                 # Protocol: PowerSpectrumProvider, RunRepository,
│   │                            #           JobExecutor, ReferenceDataProvider
│   ├── config.py                # dataclass immutabili: ForecastSettings, AngularSettings, ...
│   ├── use_cases.py             # compute_fiducial_cls, compute_derivatives,
│   │                            # compute_fishers, run_forecast (UNA pipeline)
│   └── caching.py               # logica "già calcolato?" basata su hash della config
│
├── adapters/
│   ├── boltzmann/camb.py, classy.py          # → PowerSpectrumProvider
│   ├── storage/hdf5_run_repository.py         # → RunRepository (ex WorkSpace + H5*)
│   ├── storage/json_config_loader.py          # file JSON → dataclass di config
│   ├── reference/ist_files.py                 # → ReferenceDataProvider
│   ├── execution/local.py, lsf.py             # → JobExecutor
│   └── reporting/plots.py, tables.py, tex.py  # output "presentazionali"
│
├── entrypoints/                 # driving adapters
│   ├── cli/                     # gli script di bin/ diventano thin wrapper
│   ├── gui/                     # PyQt
│   └── notebook/                # notebook_helpers
│
└── bootstrap.py                 # composition root: sceglie e collega gli adapter
```

### 4.3 I quattro port (e solo quattro)

```python
# application/ports.py
from typing import Protocol
import numpy as np
from seyfert.domain.power_spectrum import PowerSpectrum
from seyfert.domain.parameters import PhysicalParametersCollection
from seyfert.domain.c_ells import ClCollection
from seyfert.domain.fisher import FisherMatrix


class PowerSpectrumProvider(Protocol):
    """CAMB, CLASS, un emulatore, o un fake per i test."""
    def compute(self, cosmo_params: dict[str, float], z_grid: np.ndarray) -> PowerSpectrum: ...


class RunRepository(Protocol):
    """Il run directory, astratto. Oggi: HDF5 su disco. Nei test: dict in memoria."""
    def load_cls(self, dvar: str, step: int) -> ClCollection | None: ...
    def save_cls(self, dvar: str, step: int, cls: ClCollection) -> None: ...
    def load_derivatives(self) -> dict[str, "ClDerivatives"] | None: ...
    def save_derivatives(self, ders: dict[str, "ClDerivatives"]) -> None: ...
    def save_fisher(self, name: str, fisher: FisherMatrix) -> None: ...


class ReferenceDataProvider(Protocol):
    """File ausiliari oggi letti 'di nascosto' dal dominio."""
    def selected_datavectors(self) -> list[str]: ...
    def ist_gcsp_pk_fisher(self, scenario: str) -> FisherMatrix: ...
    def scaled_mean_luminosity(self) -> tuple[np.ndarray, np.ndarray]: ...


class JobExecutor(Protocol):
    """Esecuzione locale o batch (LSF, domani SLURM)."""
    def submit(self, command: str, resources: "JobResources") -> str: ...
    def wait(self, job_ids: list[str]) -> None: ...
```

### 4.4 Un solo use case di pipeline

```python
# application/use_cases.py
def run_forecast(settings: ForecastSettings,
                 pk: PowerSpectrumProvider,
                 repo: RunRepository,
                 refs: ReferenceDataProvider) -> list[FisherMatrix]:
    fid_cosmo = domain.cosmology.build(settings.params.fiducials(), pk.compute(...))
    densities = domain.redshift_density.evaluate_all(settings.probes, settings.z_grid)
    fid_cls = repo.load_cls("central", 0) or _compute_and_store_cls(...)
    ders = repo.load_derivatives() or _compute_and_store_derivatives(...)
    delta_cls = domain.covariance.delta_cls(fid_cls, settings)
    fishers = [domain.fisher.compute(dv, ders, delta_cls)
               for dv in refs.selected_datavectors()]
    for f in fishers:
        repo.save_fisher(f.name, f)
    return fishers
```

`seyfert_core.do_forecast`, la classe `Seyfert` e il flusso di `task_runners` convergono qui:
la CLI, la GUI e Streamlit chiamano **la stessa funzione** con adapter diversi.

### 4.5 Composition root

```python
# bootstrap.py
def build(settings: ForecastSettings, run_dir: Path):
    pk = {"CAMB": CambProvider, "CLASS": ClassyProvider}[settings.boltzmann_code](settings.camb)
    repo = Hdf5RunRepository(run_dir)
    refs = IstFilesReferenceData(settings.input_data_dir)
    return partial(run_forecast, settings=settings, pk=pk, repo=repo, refs=refs)
```

È l'**unico** posto che conosce le classi concrete. L'`if/elif` di `PowerSpectrum` sparisce dal dominio.

### 4.6 Come si mappa il codice attuale

| Oggi | Destinazione |
|---|---|
| `cosmology/*` (parte matematica) | `domain/` |
| `fromHDF5` / `saveToHDF5` / classi `H5*` | `adapters/storage/` (sono già quasi dei Data Mapper: vanno solo spostati) |
| `cosmology/boltzmann_solver.py` | `adapters/boltzmann/` (l'ABC diventa il port) |
| `utils/workspace.py` | `adapters/storage/hdf5_run_repository.py` |
| `config/*` | dataclass in `application/config.py` + loader JSON in `adapters/storage/` |
| `main/seyfert_core.py`, `seyfert.py`, `main/orchestration.py` | `application/use_cases.py` (unificati) |
| `main/bsub_utils.py`, `main/task_runners.py` | `adapters/execution/` |
| `fisher/fisher_plot.py`, `plot_utils/`, `utils/tex_utils.py`, `table_utils.py` | `adapters/reporting/` |
| `gui/`, `notebook_helpers/`, `bin/` | `entrypoints/` |
| `utils/` (hub generico) | da smembrare: ciò che è numerico → `domain/numeric`, ciò che è filesystem → adapter |

## 5. Piano di migrazione incrementale (strangler)

Ogni passo è rilasciabile da solo e non cambia i risultati numerici (verificabile con l'integration test esistente).

1. **Rete di sicurezza**: congelare un run di riferimento (C_ℓ, derivate, Fisher) e aggiungere un test di regressione
   numerica che lo confronta. Rimuovere `from sympy import comp`, `mytest_debug/`, `.DS_Store`; separare dipendenze dev/opzionali.
2. **Unificare l'orchestrazione**: far diventare `Seyfert` un thin wrapper di `seyfert_core` (o viceversa) eliminando la copia di `prepare_workspace`.
3. **Introdurre `PowerSpectrumProvider`**: promuovere `ExternalBoltzmannSolver` a `Protocol`, spostare la scelta CAMB/CLASS nel composition root.
4. **Introdurre `RunRepository`**: avvolgere `WorkSpace` + mapper `H5*`; sostituire pickle con HDF5; chiave di cache = hash della config.
5. **Estrarre `ReferenceDataProvider`**: togliere `fsu.*` dal dominio (`weight_functions.py:191`, `fisher_utils.py:86,99`).
6. **Svuotare il dominio dall'I/O**: deprecare `fromHDF5`/`saveToHDF5` sulle classi di dominio (prima delegano al mapper, poi spariscono); togliere il plot da `c_ells.py`.
7. **`JobExecutor`**: astrarre `BsubInterface`; `task_runners` diventa adapter.
8. **Far rispettare la regola**: aggiungere [import-linter](https://github.com/seddonym/import-linter) in CI con un contratto
   `domain` ✗→ `application`, `adapters`, `entrypoints`, h5py, matplotlib, camb.

## 6. Nota didattica sull'esagonale

Cosa ti insegna bene questo progetto rispetto al pattern:

- **Port ≠ interfaccia per ogni classe.** Un port nasce dove c'è un *bordo* (I/O, sistema esterno, scelta intercambiabile). numpy non è un bordo.
- **Hai già un esempio di adapter "giusto"** (`ExternalBoltzmannSolver`): il difetto è solo *chi* lo sceglie (il dominio) e *dove* vive (dentro `cosmology/`).
- **Il "repository" non deve per forza essere un database**: qui è il run directory HDF5, e astrarlo ti dà test in memoria gratis.
- **Driving vs driven**: CLI/GUI/Streamlit *guidano* l'applicazione; CAMB/HDF5/LSF *sono guidati*. Il valore pratico è che aggiungere un nuovo driver (la tua app Streamlit) diventa banale.
