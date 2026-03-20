# Scripts di valutazione

Questa cartella contiene gli script utilizzati per replicare le valutazioni di confronto nel project work,
con particolare riferimento al confronto tra il modello di Machine Learning e baseline statistiche.

## evaluate_baseline.py — Confronto con baseline (solo ML, senza regole)

Lo script confronta i modelli scelti con due baseline basate su (`DummyClassifier`):
- `most_frequent` (predice sempre la classe più rappresentata )
- `stratified` (genera predizioni casuali rispettando la distribuzione delle classi)

Il confronto è eseguito **solo ML** (senza regole) per isolare il contributo del componente di Machine Learning.
Vengono utilizzati **due split distinti**:
- uno stratificato su **categoria**
- uno stratificato su **priorità**


Lo script produce in output (a terminale):

- **Accuracy** per baseline e modelli scelti  
- **F1‑Macro** per baseline e modelli  
- dimensione del test set utilizzato  
- un riepilogo in tabella leggibile  

I valori ottenuti corrispondono alle **Tabelle 1–2** riportate nell’elaborato.


### Requisiti
Dalla root del repository:
```bash
pip install -r requirements.txt
```

### Esecuzione dello script
```bash
python scripts/evaluate_baseline.py --data ticket_sintetici.csv
```


