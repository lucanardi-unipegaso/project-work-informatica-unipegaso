# Triage automatico dei ticket con Machine Learning

## Descrizione del progetto
Questo repository contiene il codice sorgente del progetto di **triage automatico dei ticket**
sviluppato nell’ambito del project work per il corso di laurea L-31.

Il sistema realizza un prototipo di classificazione automatica dei ticket di supporto aziendale,
assegnando a ciascuna richiesta:
- una **categoria** (Tecnico, Amministrazione, Commerciale);
- una **priorità operativa** (bassa, media, alta).

L’obiettivo è mostrare come tecniche di **Machine Learning supervisionato**, combinate con **regole di dominio**, 
possano supportare il processo di smistamento iniziale dei ticket in modo semplice, riproducibile e interpretabile.

---

## Funzionalità principali
- **Predizione singolo ticket** tramite dashboard Streamlit  
- **Predizione batch da CSV** (colonne richieste: `title`, `body`)  
- **Dataset sintetico** generato automaticamente (1000 ticket)  
- **Approccio ibrido**: ML + Rule Engine (override condizionato per casi critici)  
- **Explainability**: top 5 parole chiave (TF IDF + keyword di dominio)  
- **Valutazione prestazioni**: Accuracy, F1 score, confusion matrix  
- **Persistenza modelli** (joblib + metadati JSON) con controllo di compatibilità
- **Confronto con baseline** (DummyClassifier) tramite script dedicato  


---

## Tecnologie utilizzate
- Python 3.11
- Streamlit  
- scikit‑learn  
- pandas, numpy  
- matplotlib / seaborn


---

## Struttura del repository
```text
project-work-informatica-unipegaso/
├── app.py                       # Dashboard Streamlit (entrypoint principale)
├── requirements.txt             # Dipendenze Python
├── ticket_sintetici.csv         # Dataset sintetico (1000 ticket)
├── predizione.csv               # Storico predizioni (generato automaticamente)
├── models/                      # Modelli e vettorizzatori serializzati
│   ├── model_cat.joblib
│   ├── vectorizer_cat.joblib
│   ├── model_pri.joblib
│   ├── vectorizer_pri.joblib
│   └── metadata.json
└── scripts/
    ├── evaluate_baseline.py     # Confronto modelli vs DummyClassifier
    └── README.md                # Istruzioni e riproducibilità baseline
```
## Deploy online
Per una prova rapida senza setup locale è disponibile un'istanza **già deployata** della dashboard:

- **URL**: https://lndev.santannapisa.it

> Nota: l'istanza online è destinata alla valutazione rapida dell'interfaccia e del flusso; la **riproducibilità completa** (training, valutazione, baseline) è garantita dalle istruzioni di esecuzione locale riportate di seguito.

---

## Installazione ed Esecuzione Locale

### Requisiti
- Python 3.11
- Dipendenze: `requirements.txt`

### Setup

#### 1) Clona il repository
```bash
git clone https://github.com/lucanardi-unipegaso/project-work-informatica-unipegaso.git
cd project-work-informatica-unipegaso
```

### Crea l’ambiente virtuale (venv) Windows
```bash
python -m venv venv
```
### Crea l’ambiente virtuale (venv) macOS / Linux
```bash
python3 -m venv venv
```

### Attiva il venv

### Windows
```bash
venv\Scripts\activate
```

### macOS / Linux
```bash
source venv/bin/activate
```

### Installa dipendenze
```bash
pip install -r requirements.txt
```

### Esegui dashboard Streamlit
```bash
streamlit run app.py
```

### L’app sarà disponibile su:
```bash
http://localhost:8501
```

### Primo Avvio
Al primo avvio il sistema:
1. valida i metadati e, se necessario, rigenera i modelli;
2. genera/ricarica il dataset sintetico;
3. carica modello e vettorizzatori;
4. rende disponibile la dashboard completa.

### Predizione batch (CSV)
Il file CSV deve contenere almeno:

- title → titolo del ticket
- body → descrizione testuale

Il sistema rileva automaticamente:

- encoding (UTF‑8, UTF‑8‑SIG, CP1252, Latin‑1…)
- separatore (virgola, ;, tab)

Al termine viene generato un CSV contenente categoria, priorità

### Valutazione del modello

Le prestazioni del sistema vengono valutate tramite una procedura di
train/test split 80/20, applicata separatamente alla classificazione di
categoria e di priorità.

Le metriche utilizzate includono:

Accuracy (accuratezza complessiva),
F1-score per classe e F1 macro,
matrici di confusione, utili per analizzare le tipologie di errore.

Il confronto con baseline statistiche (DummyClassifier) mostra:

Categoria → LinearSVC nettamente superiore (F1 ≈ 0.99 vs 0.27–0.34)
Priorità → Logistic Regression balanced supera le baseline (F1 ≈ 0.76 vs 0.21–0.40)

Lo script di valutazione è disponibile in:
scripts/evaluate_baseline.py

### Limiti

- Il sistema è addestrato su un dataset sintetico: servono test su casi reali.
- Maggiore incertezza nella distinzione bassa↔media (grey zone).
- Evoluzioni del lessico o dei processi possono ridurre la performance (richiede manutenzione).

### Riproducibilità Comparazione (valutazione e baseline)

Il repository contiene tutti gli script necessari per replicare:

- addestramento modelli
- valutazione
- baseline (DummyClassifier)
- pipeline Streamlit

I risultati di comparazione riportati nell’elaborato sono ottenibili rieseguendo gli script presenti in scripts/.

➡️ Istruzioni: vedere la cartella [scripts](scripts/)
``

