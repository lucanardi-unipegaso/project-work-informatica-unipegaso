# Scripts di valutazione

Questa cartella contiene script utili per replicare le valutazioni riportate nel project work.

## evaluate_baseline.py — Confronto con baseline (solo ML, senza regole)

Lo script confronta i modelli scelti con due baseline "vere" (`DummyClassifier`):
- `most_frequent` (predice sempre la classe più frequente)
- `stratified` (predice in modo casuale rispettando la distribuzione delle classi)

Il confronto è eseguito **solo ML** (senza regole) per isolare il contributo del componente di Machine Learning.
Vengono utilizzati **due split distinti**:
- uno stratificato su **categoria**
- uno stratificato su **priorità**

### Requisiti
Dalla root del repository:
```bash
pip install -r requirements.txt