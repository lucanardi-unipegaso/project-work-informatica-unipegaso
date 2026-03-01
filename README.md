# Triage automatico dei ticket con Machine Learning

## Descrizione del progetto
Questo repository contiene il codice sorgente del progetto di **triage automatico dei ticket**
sviluppato nell’ambito del project work per il corso di laurea L-31.

Il sistema realizza un prototipo di classificazione automatica dei ticket di supporto aziendale,
assegnando a ciascuna richiesta:
- una **categoria** (Tecnico, Amministrazione, Commerciale);
- una **priorità operativa** (bassa, media, alta).

L’obiettivo del progetto è dimostrare come tecniche di **Machine Learning supervisionato**,
applicate a testi brevi, possano supportare i processi di gestione dei ticket in modo semplice,
riproducibile e interpretabile, con particolare attenzione alla valutazione delle prestazioni.

---

## Funzionalità principali
- Predizione singola dei ticket tramite interfaccia web basata su Streamlit.
- Predizione batch a partire da file CSV.
- Dataset sintetico generato automaticamente per l’addestramento dei modelli.
- Valutazione delle prestazioni tramite **Accuracy**, **F1-score** e **matrici di confusione**.
- Approccio ibrido: modelli di Machine Learning combinati con regole basate su parole chiave.
- Dashboard interattiva per l’analisi dei risultati e delle metriche.

---

## Tecnologie utilizzate
- Python 3
- Streamlit
- scikit-learn
- pandas
- numpy
- matplotlib / seaborn

---

## Struttura del repository
```text
triage-ticket-ml/
├── triage_dashboard_v99_12am_fix812a.py   # Applicazione Streamlit principale
├── models/                               # Modelli ML persistenti (auto-generati)
│   ├── vectorizer_cat.joblib             # TF-IDF per categoria
│   ├── model_cat.joblib                  # LinearSVC (categoria)
│   ├── vectorizer_pri.joblib             # TF-IDF per priorità
│   ├── model_pri.joblib                  # Logistic Regression (priorità)
│   └── metadata.json                     # Metadati di training e versioning
├── predizione.csv                        # Storico ticket analizzati
├── dataset.csv                           # Dataset sintetico (1000 ticket)
├── requirements.txt                     # Dipendenze Python
└── README.md                             # Questo file


---

## Installazione ed Esecuzione

### Requisiti
- Python 3.9+
- Dipendenze: `requirements.txt`

### Setup
```bash
# Clone repository
git clone https://github.com/lucanardi-unipegaso/project-work-informatica-unipegaso.git
cd triage-ticket-ml

# Installa dipendenze
pip install -r requirements.txt

# Esegui dashboard
streamlit run triage_dashboard_v99_12am_fix812a.py

```

### Primo Avvio
Al primo avvio il sistema:
1. Genera un dataset sintetico di circa 1000 ticket.
2. Addestra i modelli di Machine Learning (LinearSVC e Logistic Regression).
3. Salva i modelli addestrati nella cartella models/ per utilizzi successivi.
4. Avvia la dashboard web accessibile all’indirizzo http://localhost:8501

```

### Valutazione del modello

Le prestazioni del sistema vengono valutate tramite una procedura di
train/test split 80/20, applicata separatamente alla classificazione di
categoria e di priorità.

Le metriche utilizzate includono:

Accuracy (accuratezza complessiva),
F1-score per classe e F1 macro,
matrici di confusione, utili per analizzare le tipologie di errore.

I risultati mostrano una buona capacità di distinguere le classi,
in particolare per i ticket con priorità alta, critici dal punto di vista operativo.

### Limiti

Il sistema è addestrato su un dataset sintetico e rappresenta un prototipo
dimostrativo. Le prestazioni osservate sono valide nel contesto sperimentale
considerato e non garantiscono una generalizzazione immediata su dati reali
senza ulteriori test e adattamenti.

### Riproducibilità Comparazione (valutazione e baseline)

Per replicare i risultati di confronto riportati nell’elaborato (baseline DummyClassifier vs modelli scelti),
è disponibile lo script di valutazione in `scripts/`.

➡️ Istruzioni: vedere [`scripts/README.md`](scripts/README.md)
