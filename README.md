# Triage automatico dei ticket con Machine Learning

## Descrizione del progetto
Questo repository contiene il codice sorgente del progetto di **triage automatico dei ticket**
sviluppato nell’ambito del project work per il corso di laurea L-31.

Il sistema realizza un prototipo di classificazione automatica dei ticket di supporto aziendale,
assegnando a ciascuna richiesta:
- una **categoria** (Tecnico, Amministrazione, Commerciale);
- una **priorità operativa** (bassa, media, alta).

L’obiettivo è dimostrare come tecniche di **Machine Learning supervisionato**, applicate a testi brevi,
possano supportare i processi di gestione dei ticket in modo semplice, riproducibile e interpretabile.

---

## Funzionalità principali
- Predizione singola dei ticket tramite interfaccia web (Streamlit).
- Predizione batch a partire da file CSV.
- Dataset sintetico generato automaticamente per l’addestramento.
- Valutazione delle prestazioni tramite Accuracy, F1-score e matrici di confusione.
- Approccio ibrido: modelli ML + regole basate su parole chiave.
- Dashboard interattiva per l’analisi dei risultati.

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
.
├── app.py                 # Applicazione Streamlit principale
├── requirements.txt       # Dipendenze Python
├── models/                # Modelli ML e vettorializzatori serializzati
├── predizione.csv         # Storico delle predizioni salvate
└── README.md              # Descrizione del progetto
