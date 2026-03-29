# -*- coding: utf-8 -*-
"""
=================================================================================
TRIAGE AUTOMATICO TICKET DEI TICKET CON MACHINE LEARNING 
=================================================================================
"""

# =============================================================================
# IMPORTAZIONE LIBRERIE
# =============================================================================
import os
import io
import json
import string
from datetime import datetime, UTC

# Data science & ML
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit per UI web
import streamlit as st

# Scikit-learn per ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import joblib  # Per serializzazione modelli


# =============================================================================
# INTERFACCIA UTENTE STREAMLIT
# =============================================================================

st.set_page_config(page_title="Triage Ticket", layout="wide")

CSS_PROFESSIONAL = """
<style>
/* === BADGE E CARD === */
.badge {
    display:inline-block; 
    padding:4px 8px; 
    border-radius:6px; 
    font-weight:600;
}

/* Badge categoria (blu) */
.badge-cat {
    background:#EEF3FE; 
    color:#1f3b66;
}

/* Badge priorità con codici colore semantici */
.badge-pri-alta {
    background:#FFE8E8; 
    color:#a61e1e;
}
.badge-pri-media {
    background:#FFF3CD; 
    color:#6b4e00;
}
.badge-pri-bassa {
    background:#E8F7EE; 
    color:#1b5e20;
}

/* Chip per parole chiave in stile code */
.codechip {
    background:#f4f6f8; 
    border:1px solid #e1e5ea; 
    padding:2px 6px; 
    border-radius:6px; 
    font-family:monospace;
}

/* Card generiche per contenuti */
.card {
    padding:10px;
    border:1px solid #e2e6ea;
    border-radius:8px;
    background:#f8f9fb;
}
.card-title {
    font-weight:600;
    margin-bottom:6px;
}
.card-body {
    font-family:monospace;
}

/* === SPACING E LAYOUT === */
/* Sezione risultati con separatore superiore */
.results-section {
    margin-top: 2rem;
    padding-top: 1.5rem;
    border-top: 2px solid #e9ecef;
}

/* Pulsante nuovo ticket con spacing */
.new-ticket-button {
    margin-top: 2rem !important;
    padding-top: 1rem;
}

/* Sezione parole chiave */
.keywords-section {
    margin-bottom: 2rem;
}

/* === STYLING PULSANTE SUBMIT === */
/* Gradiente viola con ombra e hover effect */
div[data-testid="column"] button[type="submit"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
    transition: all 0.3s ease !important;
}

div[data-testid="column"] button[type="submit"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4) !important;
}
</style>
"""
st.markdown(CSS_PROFESSIONAL, unsafe_allow_html=True)


# =============================================================================
# MESSAGGI UI
# =============================================================================

# Newline character per costruzione messaggi multilinea
NL = chr(10)

# Messaggi tab Predizione
MSG_PRED_TITLE = "Triage automatico ticket"
MSG_PRED_CAPTION = "Classificazione categoria e priorità con modelli ML"
MSG_PLACEHOLDER_TITLE = "Es.: Errore durante la chiusura contabile"
MSG_PLACEHOLDER_BODY = "Descrivi il problema o la richiesta…"
MSG_FORM_SUBMIT = "Nuova predizione"
MSG_INSERT_FIELDS = "Inserisci titolo e descrizione prima di analizzare."
MSG_ANALYSIS_SPINNER = "Analisi in corso..."
MSG_NEW_TICKET_BTN = "Nuovo ticket"

# Messaggi tab Ticket Salvati
MSG_SAVED_TITLE = "Ticket salvati"
MSG_NO_TICKETS = "Nessun ticket salvato al momento."
MSG_FILTERS_SUBTITLE = "Filtri rapidi"
MSG_FILTER_DATE = "Data"
MSG_FILTER_CATS = "Categoria"
MSG_FILTER_PRI = "Priorità"
MSG_ALL_DATES = "Tutte"

# Ordine priorità
PRI_ORDER = ['bassa', 'media', 'alta']

# Messaggi tab Batch
MSG_BATCH_TITLE = "Batch prediction & Dataset"
MSG_DOWNLOAD_DATASET = "Download dataset sintetico"
MSG_PREP_DOWNLOAD = "Preparazione download dataset sintetico"
MSG_DOWNLOAD_BTN = "Scarica dataset sintetico (CSV)"
MSG_DOWNLOAD_PRED_SAVED = "Download predizioni salvate"
MSG_NO_PRED_SAVED = "Nessuna predizione salvata ancora."
MSG_BATCH_UPLOAD = "Predizione batch da CSV"
MSG_UPLOAD_LABEL = "Carica file CSV"
MSG_UPLOAD_OK = "CSV caricato (encoding rilevato: {enc})."
MSG_COLS_DETECTED = "Colonne rilevate: {cols}"
MSG_NEED_COLS = "Il CSV deve contenere almeno le colonne: 'title'/'titolo' e 'body'/'descrizione'."
MSG_ERR_READ_CSV = "Impossibile leggere il CSV: problema di encoding o formattazione."
MSG_DETAILS_ATTEMPTS = "Dettagli tentativi:"
MSG_TIPS = "Suggerimenti:"
MSG_TIP_UTF8 = "• Se usi Excel, salva il CSV come UTF-8 (File → Salva con nome → CSV UTF-8)."
MSG_TIP_HELP = "• In alternativa, inviami il file e posso indicarti l'encoding corretto."
MSG_TICKET_SAVED_CSV_BTN = "Scarica ticket salvati (CSV)"
MSG_PRED_CSV_BTN = "Scarica predizione.csv"
MSG_PREDICTED_CSV_BTN = "Scarica CSV con predizioni"


# =============================================================================
# GESTIONE FILE CSV PREDIZIONI
# =============================================================================

# File CSV con predizioni effettuate
PRED_FILE = "predizione.csv"

# Inizializza file 
if not os.path.exists(PRED_FILE):
    # Crea file vuoto con struttura corretta
    pd.DataFrame(columns=["id", "date", "title", "body", "category", "priority"]).to_csv(
        PRED_FILE, index=False, encoding="utf-8"
    )
else:
    # Verifica e aggiungi colonna date se assente
    try:
        tmp = pd.read_csv(PRED_FILE)
        if 'date' not in tmp.columns:
            tmp.insert(1, 'date', '')  
        tmp.to_csv(PRED_FILE, index=False, encoding="utf-8")
    except Exception:
        pass  


# =============================================================================
# FUNZIONI PREPROCESSING TESTO
# =============================================================================

def clean_text(text: str) -> str:
  
    text = (text or "").lower() 
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


# =============================================================================
# CONFIGURAZIONE TF-IDF
# =============================================================================

ITALIAN_STOPWORDS = [
    "di","a","da","in","con","su","per","tra","fra","il","lo","la","i","gli","le",
    "un","uno","una","e","ed","che","non","come","anche","ma","o",
    "al","ai","agli","alle","all","dal","dai","dagli","dalle","dell","dei","degli","delle",
    "mi","ti","ci","vi","si","questo","quello","questa","quella","quelli","quelle",
    "sono","sei","siamo","siete","era","ero","erano","essere","stato","stata"
]

# Parametri TF-IDF Vectorizer
VEC_KWARGS = dict(
    strip_accents="unicode",     # Normalizza accenti (è -> e)
    ngram_range=(1, 2),          # Unigrammi e bigrammi (cattura frasi brevi)
    min_df=1,                    # Minimo 1 documento (mantiene termini rari)
    max_df=0.95,                 # Filtra termini troppo comuni
    stop_words=ITALIAN_STOPWORDS # Stopwords custom italiane
)


# =============================================================================
# DIZIONARI KEYWORDS PER REGOLE PRIORITÀ
# =============================================================================

priorities_keywords = {
    # PRIORITÀ ALTA: blocchi critici, urgenze, scadenze immediate
    "alta": [
        "bloccato", "bloccante", "non risponde",
        "impossibile accedere", "impossibile stampare", "down", "offline",
        "schermata blu", "schermate blu", "bsod", "crash", "errore critico", 
        
        "urgente", "urgenza", "scadenza oggi", "entro oggi",
        
        "impossibilita di stampare", "impossibilità di stampare", 
        "non si riesce a stampare", "blocco stampa", "stampa bloccata", 
        
        "documenti fiscali", "documenti contabili",
        "scadenza", "chiusure amministrative", "chiusura amministrativa",
        
        "non accessibile", "non è accessibile", "accesso non disponibile", 
        "servizio non disponibile", "server non disponibile", "server down",
        
        "da nessuna postazione", "tutte le postazioni", 
        "non accessibile da nessuna postazione",
        "tutto il reparto", "su nessuna postazione", "intero reparto",
    ],
    
    # PRIORITÀ MEDIA: problemi funzionali non bloccanti
    "media": [
        "problema", "malfunzionamento", "funziona male", 
        "lento", "prestazioni basse",
        "anomalia", "errore sporadico", "errore occasionale"
    ],
    
    # PRIORITÀ BASSA: richieste informative, configurazioni non urgenti
    "bassa": [
        "richiesta installazione", "installazione software", 
        "richiesta permessi", "permessi di installazione",
        "richiesta abilitazione", "abilitazione", 
        "richiesta configurazione", "configurazione",
        
        "richiesta informazioni", "informazioni", "info", 
        "richiesta dettagli", "dettagli",
        "richiesta chiarimento", "chiarimento", 
        "domanda", "domanda generica",
        
        "richiesta assistenza", "richiesta supporto",
        "ho bisogno di assistenza", "necessito assistenza",
        
        "documentazione", "richiesta documentazione", 
        "modulo", "richiesta modulo", "modulistica",
        
        "preventivo", "richiesta preventivo", "listino", 
        "richiesta listino", "offerta", "richiesta offerta", 
        "promozione", "richiesta promozione",
        "contatto", "richiesta contatto", 
        "appuntamento", "richiesta appuntamento",
        
        "nota spese", "info nota spese", "informazioni nota spese", 
        "rimborso spese", "rendicontazione", "rendiconto spese", 
        "procedura nota spese", "modulo nota spese", "modulistica nota spese",
        
        "verificare aggiornamento", "verifica aggiornamento", 
        "aggiornamento recepito", "verificare recepimento", 
        "richiesta verifica", "richiesta controllo",
        "controllare stato", "verifica stato"

        "formazione", "training", "corso", "sessione formativa", "affiancamento", "tutorial"
    ],
}


# =============================================================================
# DIZIONARI KEYWORDS PER REGOLE CATEGORIA
# =============================================================================

# CATEGORIA TECNICO - Keywords forti
CATEGORY_RULES_TECNICO_STRONG = {
    "crash", "schermata blu", "bsod", "non risponde",
    "impossibile accedere", "impossibile stampare",

    "non accessibile", "non è accessibile", 
    "accesso non disponibile", "servizio non disponibile",
    "server non disponibile", "server down", "down", "offline",
    
    "non riesco ad aprire", "file non si apre", 
    "errore apertura", "errore di sistema",
    
    "server condiviso", "cartella di rete", "percorso di rete", 
    "share di rete", "smb", "permessi cartella", "permessi di condivisione",
    
    "vpn", "wifi", "rete", "timeout", "bloccato", 
    "errore applicativo", "errore tecnico"
}

# CATEGORIA TECNICO - Keywords generali
CATEGORY_RULES_TECNICO = {
    "smartphone", "cellulare", "telefono", "mobile", 
    "dispositivo", "device", "android", "iphone", "ios", "tablet",
    
    "app", "applicazione", "wifi", "wi-fi", "vpn", 
    "browser", "password", "account",
    
    "stampante", "stampa", "spooler", "coda di stampa", 
    "driver di stampa", "inceppamento", "toner",
}

# CATEGORIA TECNICO - Installazioni software
CATEGORY_RULES_INSTALL_SOFTWARE = {
    "installazione software", "richiesta installazione", "installare",
    "setup", "set up", "istanza software", "deploy software",
    "configurazione software", "configurare software",
    "installazione del software", "installazione applicativo",
    "installare il software", "installazione programma"
}

# CATEGORIA AMMINISTRAZIONE
CATEGORY_RULES_AMMINISTRAZIONE = {
    "fatturazione", "dati di fatturazione", "dati fatturazione",
    "dati anagrafici", "anagrafica", "anagrafiche",
    "anagrafica clienti", "anagrafica fornitori",
    
    "fattura", "fatture", "pagamento", "pagamenti",
    "note di credito", "documenti contabili", "documenti fiscali",
    
    "estratto conto", "iban", "modifica dati", "aggiornamento dati",
    "variazione dati", "duplicato documento fiscale", 
    "copia documento fiscale", "duplicato fattura", "copia fattura",
    
    "chiusura contabile", "chiusura contabile mensile",
    "chiusura fiscale", "chiusura fiscale mensile",
    "chiusure amministrative", "chiusure contabili",
    "nota di credito", "documento di accredito", 
    "portale amministrativo", "scadenza fiscale", "scadenze fiscali",
    
    "contratto", "contratti", "fornitore", "fornitori",
    "contratto fornitore", "contratti fornitori",
    "rinnovo contratto", "scadenza contratto"
}

# CATEGORIA COMMERCIALE
CATEGORY_RULES_COMMERCIALE = {
    "preventivo", "offerta", "offerta commerciale", 
    "proposta commerciale", "ordine", "listino", 
    "promozione", "catalogo", "disponibilita", "disponibilità",
    
    "cliente", "crm", "anagrafica cliente", "anagrafiche clienti",
    "duplicazione anagrafica", "duplicazione cliente",
    "anagrafica commerciale", "dati cliente (crm)", 
    
    "portale vendite", "condizioni economiche", 
    "modifica offerte", "gestione offerte",
    
    "vendite", "statistiche di vendita", "report vendite",
    "portale commerciale", "area vendite", "report commerciale",
    "dati di vendita", "dati vendite", "dashboard vendite", 
    "kpi vendite", "analisi vendite", "performance vendite",
    
    "conferma ordine", "conferme d ordine", "sistema commerciale", 
    "email conferma ordine"
}


# =============================================================================
# FUNZIONE REGOLE CATEGORIA
# =============================================================================

def rule_based_category(text: str):
  
    t = clean_text(text)
    
    # REGOLA 1: Commerciale critici
    commerciale_critical_terms = {
        "conferme d ordine", "conferme ordine", "conferma ordine", 
        "email conferma ordine", "invio conferme", "conferme automatiche",
        "sistema commerciale", "ordine cliente", "ordini clienti",
        "processamento ordini", "gestione ordini"
    }
    if any(tok in t for tok in commerciale_critical_terms):
        return "Commerciale"

    # REGOLA 2: Commerciale portali e configurazioni
    if any(tok in t for tok in ["portale vendite", "condizioni economiche", 
                                 "offerte", "listino", "catalogo"]):
        return "Commerciale"
    
    # REGOLA 3: Commerciale statistiche/reporting vendite
    sales_terms = {
        "statistiche di vendita", "report vendite", "dashboard vendite",
        "analisi vendite", "performance vendite", "kpi vendite",
        "dati di vendita", "dati vendite", "portale commerciale", 
        "area vendite", "report commerciale"
    }
    if any(tok in t for tok in sales_terms):
        return "Commerciale"
    
    # REGOLA 4: Commerciale CRM
    crm_terms = {
        "crm", "anagrafica cliente", "anagrafiche clienti", 
        "duplicazione anagrafica", "duplicazione cliente"
    }
    if any(tok in t for tok in crm_terms):
        return "Commerciale"
    
    # REGOLA 5: Tecnico supporto tecnico generico (PRIORITÀ ALTA)
    supporto_tecnico_terms = {
        "supporto tecnico", "assistenza tecnica", 
        "richiesta supporto tecnico", "richiesta assistenza tecnica"
    }
    if any(tok in t for tok in supporto_tecnico_terms):
        return "Tecnico"
    
    # REGOLA 5b: Tecnico stampa (PRIORITÀ ALTA su amministrazione)
    stampa_terms = {
        "stampante", "stampa", "spooler", "coda di stampa", 
        "driver di stampa", "inceppamento", "toner"
    }
    if any(tok in t for tok in stampa_terms):
        return "Tecnico"
    
    # REGOLA 6: Tecnico installazioni software
    if any(tok in t for tok in CATEGORY_RULES_INSTALL_SOFTWARE):
        return "Tecnico"
    
    # REGOLA 7: Amministrazione (fatture, contabilità, chiusure)
    if any(tok in t for tok in CATEGORY_RULES_AMMINISTRAZIONE):
        return "Amministrazione"
    
    # REGOLA 8: Tecnico strong (crash, blocchi critici)
    if any(tok in t for tok in CATEGORY_RULES_TECNICO_STRONG):
        return "Tecnico"
    
    # REGOLA 9: Commerciale generale
    if any(tok in t for tok in CATEGORY_RULES_COMMERCIALE):
        return "Commerciale"
    
    # REGOLA 10: Tecnico generale (dispositivi, app, connettività)
    if any(tok in t for tok in CATEGORY_RULES_TECNICO):
        return "Tecnico"
    
    # Nessuna regola applicabile → lascia decidere al modello ML
    return None


# =============================================================================
# FUNZIONE REGOLE PRIORITÀ
# =============================================================================

def rule_based_priority(text: str) -> str | None:

    t = clean_text(text)
    
    # REGOLA 1: COMMERCIALE CRITICO ALTA
    commerciale_blocker_terms = [
        "conferme d ordine", "conferme ordine", "conferma ordine",
        "email conferma ordine", "invio conferme", "email ordine",
        "non invia", "non vengono inviate", "non arrivano", "non partono",
        "mancato invio", "invio fallito"
    ]
    commerciale_context = [
        "sistema commerciale", "ordine", "ordini", "cliente", "clienti"
    ]
    
    has_blocker = any(b in t for b in commerciale_blocker_terms)
    has_context = any(c in t for c in commerciale_context)
    
    if has_blocker and has_context:
        return "alta"

    # REGOLA 2: PROBLEMI AUDIO/CUFFIE NON BLOCCANTI MEDIA
    audio_terms = ["cuffie", "audio", "microfono", "altoparlante", "speaker"]
    non_blocking_terms = [
        "intermittente", "funziona male", "problema", 
        "malfunzionamento", "non funziona correttamente"
    ]
    urgency_terms = [
        "bloccato", "impossibile usare", "non risponde", 
        "non si avvia", "blocco totale", "schermo nero",
        "urgente", "urgenza", "entro oggi", "scadenza oggi", "oggi"
    ]
    
    if any(a in t for a in audio_terms):
        if not any(u in t for u in urgency_terms):
            return "media"

    # REGOLA 3: COMMERCIALE IMPOSSIBILE CREARE DOC ALTA
    commerciale_terms = [
        "preventivo", "ordine", "offerta", 
        "proposta commerciale", "listino"
    ]
    blocker_terms = [
        "impossibile creare", "non riesco a creare", 
        "procedura si blocca", "bloccato", "bloccata", "bloccante",
        "non si completa", "non va avanti", "non prosegue", "non si conclude"
    ]
    
    if any(c in t for c in commerciale_terms) and any(b in t for b in blocker_terms):
        return "alta"

    # REGOLA 4: AMMINISTRAZIONE BLOCCO CHIUSURE ALTA
    admin_block_terms = [
        "chiusura contabile", "chiusura contabile mensile",
        "chiusura fiscale", "chiusura fiscale mensile",
        "chiusure amministrative", "chiusure contabili"
    ]
    block_terms = [
        "non permette di completare", "non consente di completare",
        "impossibile completare", "bloccata la procedura",
        "blocca l'operazione", "operazione bloccata", "procedura bloccata"
    ]
    
    if any(a in t for a in admin_block_terms) and any(b in t for b in block_terms):
        return "alta"

    # REGOLA 5: AMMINISTRAZIONE ERRORI DOC SENZA URGENZA MEDIA
    admin_tx_terms = {
        "nota di credito", "note di credito",
        "fattura", "fatture",
        "pagamento", "pagamenti",
        "documenti contabili", "documenti fiscali",
        "dati di fatturazione", "dati fatturazione"
    }
    has_admin_tx = any(a in t for a in admin_tx_terms)
    has_error = ("errore" in t) or ("errore critico" in t)
    has_urgency = any(u in t for u in ["urgente", "urgenza", "entro oggi", 
                                        "scadenza oggi", "oggi"])
    
    # Errore su documenti amministrativi senza urgenza MEDIA
    if has_admin_tx and has_error and not has_urgency:
        return "media"

    # REGOLA 6: OUTAGE SERVIZI SU POSTAZIONI ALTA
    outage_terms = [
        "non accessibile", "non è accessibile", "accesso non disponibile",
        "servizio non disponibile", "server non disponibile", "server down",
        "down", "offline", "non si avvia", "non parte", "avvio bloccato",
        "avvio fallito", "errore in avvio", "blocco applicazione", 
        "applicazione bloccata", "impossibile avviare", "impossibile aprire"
    ]
    ampiezza_terms = [
        "da nessuna postazione", "tutte le postazioni", "intero ufficio",
        "nessuna postazione", "tutto il reparto", "intero reparto",
        "più postazioni", "diverse postazioni", "alcune postazioni", 
        "vari utenti", "più utenti", "reparto", "su più pc", "su più computer"
    ]
    
    if any(o in t for o in outage_terms) and any(a in t for a in ampiezza_terms):
        return "alta"

    # REGOLA 7: PC NON SI ACCENDE/AVVIA ALTA
    blocker_single_terms = [
        "non si accende", "non parte il pc", "non parte il computer", 
        "schermo nero", "nessuna schermata", 
        "non si avvia il pc", "non si avvia il computer",
        "avvio bloccato", "avvio fallito", "errore in avvio"
    ]
    
    if any(b in t for b in blocker_single_terms):
        return "alta"

    # REGOLA 8: COMMERCIALE DATA QUALITY SENZA URGENZA MEDIA
    sales_dq_terms = {
        "statistiche di vendita", "report vendite", "dashboard vendite",
        "analisi vendite", "performance vendite", "kpi vendite", 
        "portale commerciale", "dati di vendita", "dati vendite", 
        "portale vendite", "report commerciale", "errore nel portale"
    }
    not_updated_terms = {
        "non aggiornate", "non aggiornato", "non è aggiornato",
        "valori non coerenti", "valori incoerenti", "dati incoerenti",
        "dati non coerenti", "dati non allineati", "non in linea", "non allineati"
    }
    urgency_terms = [
        "urgente", "urgenza", "entro oggi", "scadenza oggi", "oggi"
    ]
    
    has_sales_context = any(s in t for s in sales_dq_terms)
    has_dq_issue = any(n in t for n in not_updated_terms)
    
    if has_sales_context and has_dq_issue:
        if not any(u in t for u in urgency_terms):
            return "media"

    # REGOLA 9: DUPLICAZIONE ANAGRAFICA CON/SENZA IMPATTO
    dq_terms = [
        "duplicazione anagrafica", "duplicazione cliente", "dati duplicati",
        "anagrafica duplicata", "inconsistenza dati", "dati incoerenti"
    ]
    impact_terms = [
        "ordine bloccato", "non è possibile creare l'ordine", 
        "impossibile evadere ordine", "workflow bloccato", 
        "integrazione bloccata", "sincronizzazione fallita",
        "urgente", "urgenza", "entro oggi"
    ]
    
    if any(dq in t for dq in dq_terms):
        if any(imp in t for imp in impact_terms):
            return "media"  
        else:
            return "bassa"  

    # REGOLA 10: DUBBI/DOMANDE SU CONTRATTI/AMMINISTRAZIONE BASSA
    admin_terms = ["contratto", "contratti", "fornitore", "fornitori", 
                   "fattura", "fatture", "pagamento", "documento fiscale"]
    question_terms = ["dubbi", "dubbio", "domanda", "chiarimento", 
                      "informazioni", "info", "dettagli"]
    
    if any(a in t for a in admin_terms) and any(q in t for q in question_terms):
        return "bassa"
    
    # REGOLE 11: KEYWORDS GENERALI PRIORITÀ
    # Fallback su dizionari keywords generali
    
    # Keywords ALTA priorità
    if any(kw in t for kw in priorities_keywords.get("alta", [])):
        return "alta"
    
    # Keywords BASSA priorità (verifiche, installazioni)
    if any(kw in t for kw in [
        "verificare aggiornamento", "verifica aggiornamento", 
        "aggiornamento recepito", "verificare recepimento", 
        "richiesta verifica", "richiesta controllo",
        "controllare stato", "verifica stato", "informazioni", "info"
    ]):
        return "bassa"
    
    if any(kw in t for kw in [
        "richiesta installazione", "installazione software", 
        "richiesta permessi", "permessi di installazione",
        "richiesta abilitazione", "abilitazione", 
        "richiesta configurazione", "configurazione"
    ]):
        return "bassa"
    
    # Keywords MEDIA priorità
    if any(kw in t for kw in priorities_keywords.get("media", [])):
        return "media"
    
    # Keywords BASSA priorità (richieste generali)
    if any(kw in t for kw in priorities_keywords.get("bassa", [])):
        return "bassa"
    
    # Nessuna regola applicabile → lascia decidere al modello ML
    return None

# =============================================================================
# GENERAZIONE DATASET SINTETICO
# =============================================================================

# ---------------------------------------------------------------------------
# UTILITY: post-processing linguistico
# ---------------------------------------------------------------------------
import re as _re

# Nomi propri e acronimi 
_PROPER_NOUNS = frozenset({
    'IT', 'PC', 'VPN', 'CRM', 'PDF', 'OK', 'LAN', 'WAN', 'HR', 'ERP', 'SAP',
    'SMTP', 'CPU', 'RAM', 'SSD', 'UPS', 'NAS', 'DNS', 'IP', 'USB',
    'Windows', 'Outlook', 'Chrome', 'Edge', 'Excel', 'Word', 'Teams', 'Zoom',
    'Linux', 'Android', 'Mac', 'iOS', 'Adobe', 'Acrobat', 'SharePoint',
    'Notepad', 'Power', 'Desktop',  
    'Office', 'AutoCAD', 'Auto', 'Photoshop', 'Illustrator', 'InDesign',
    'Skype', 'Slack', 'GitHub', 'GitLab', 'Jira', 'Confluence', 'Salesforce',
})

def _fix_italian(text: str) -> str:
    """Corregge automaticamente incongruenze grammaticali comuni nell'italiano."""
    fixes = [
        # Preposizioni articolate
        (r'\bsu il\b',   'sul'),
        (r'\bsu lo\b',   'sullo'),
        (r'\bsu la\b',   'sulla'),
        (r'\bsu i\b',    'sui'),
        (r'\bsu gli\b',  'sugli'),
        (r'\bsu le\b',   'sulle'),
        (r'\ba il\b',    'al'),
        (r'\ba lo\b',    'allo'),
        (r'\ba la\b',    'alla'),
        (r'\ba i\b',     'ai'),
        (r'\ba gli\b',   'agli'),
        (r'\ba le\b',    'alle'),
        (r'\bdi il\b',   'del'),
        (r'\bdi lo\b',   'dello'),
        (r'\bdi la\b',   'della'),
        (r'\bdi i\b',    'dei'),
        (r'\bdi gli\b',  'degli'),
        (r'\bdi le\b',   'delle'),
        (r'\bda il\b',   'dal'),
        (r'\bda lo\b',   'dallo'),
        (r'\bda la\b',   'dalla'),
        (r'\bda i\b',    'dai'),
        (r'\bda gli\b',  'dagli'),
        (r'\bda le\b',   'dalle'),
        (r'\bin il\b',   'nel'),
        (r'\bin lo\b',   'nello'),
        (r'\bin la\b',   'nella'),
        (r'\bin i\b',    'nei'),
        (r'\bin gli\b',  'negli'),
        (r'\bin le\b',   'nelle'),
        # Ridondanze
        (r'\bprocedura per la procedura\b',   'procedura'),
        (r'\bprocedura per il processo\b',    'processo'),
        (r'\bil cliente il cliente\b',         'il cliente'),
        (r'\bl\'utente l\'utente\b',           "l'utente"),
        (r'\b(\w+) \1\b',                      r'\1'),   # parola duplicata generica
    ]
    for pattern, replacement in fixes:
        text = _re.sub(pattern, replacement, text, flags=_re.IGNORECASE)

    # Abbassa maiuscole spurie mid-sentence.
    # Lookbehind: char non-punteggiatura (lettera/cifra/virgola) + spazio + Maiuscola.
    # Preserva acronimi puri (≥2 char tutti-maiuscolo) e nomi propri in _PROPER_NOUNS.
    def _lower_cap(m):
        word = m.group(1)
        if word.isupper() or word in _PROPER_NOUNS:
            return m.group(0)
        return m.group(0)[:-len(word)] + word[0].lower() + word[1:]

    text = _re.sub(r'(?<=[a-zà-ùA-Z0-9,])\s([A-ZÀ-Ù][a-zà-ù][a-zà-ùA-Z]*)', _lower_cap, text)
    return text



def _join_fragment(base: str, fragment: str) -> str:
    """
    Appende `fragment` a `base` garantendo capitalizzazione corretta.

    - Se `base` termina con punteggiatura forte (.!?) → `fragment` inizia
      con maiuscola (è una nuova frase): lo lasciamo com'è.
    - Altrimenti (mid-sentence) → primo carattere di `fragment` in minuscolo,
      salvo che sia un acronimo o nome proprio in _PROPER_NOUNS.

    Usare questa funzione ovunque si concateni body + suffisso variabile.
    """
    base = base.rstrip()
    if not fragment:
        return base
    frag = fragment.lstrip()
    # Determina se stiamo iniziando una nuova frase
    new_sentence = base and base[-1] in '.!?'
    if new_sentence:
        # Nuova frase: assicura spazio e maiuscola
        first_word = _re.match(r'[A-ZÀ-Üa-zà-ü]+', frag)
        if first_word:
            w = first_word.group()
            if w not in _PROPER_NOUNS and not w.isupper():
                frag = w[0].upper() + w[1:] + frag[len(w):]
        return base + ' ' + frag
    else:
        # Mid-sentence: abbassa il primo carattere se non è nome proprio/acronimo
        first_word = _re.match(r'[A-ZÀ-Üa-zà-ü]+', frag)
        if first_word:
            w = first_word.group()
            if w not in _PROPER_NOUNS and not w.isupper():
                frag = w[0].lower() + w[1:] + frag[len(w):]
        return base + ' ' + frag


# ---------------------------------------------------------------------------
# costruttore ticket con coerenza titolo↔body
# ---------------------------------------------------------------------------
def _make_ticket(title_tpl: str, body_tpl: str, **kwargs) -> tuple[str, str]:
    """
    Espande titolo e body con gli STESSI kwargs, poi applica fix grammaticali.
    Garantisce coerenza semantica titolo↔body.
    """
    title = _fix_italian(title_tpl.format(**kwargs))
    body  = _fix_italian(body_tpl.format(**kwargs))
    return title, body


# ---------------------------------------------------------------------------
# VARIABILI CONTESTUALI 
# ---------------------------------------------------------------------------
VAR_TEMPI = [
    "da stamattina", "da ieri pomeriggio", "da circa un'ora",
    "dall'aggiornamento di ieri sera", "dal rientro dalla pausa pranzo",
    "dal cambio turno di questa mattina", "da quando è stato fatto l'aggiornamento",
    "dall'ultimo riavvio del server", "da questa mattina presto"
]

VAR_TENTATIVI = [
    "Ho già provato a riavviare ma niente.",
    "Ho riavviato due volte, il problema persiste.",
    "Già controllato cavi e alimentazione, tutto ok.",
    "Ho provato da un'altra postazione, stesso errore.",
    "Riavvio del servizio non ha risolto.",
    "Ho svuotato la cache ma non cambia nulla.",
    "Ho provato a reinstallare il driver senza successo.",
    "Già segnalato la settimana scorsa, si è ripresentato.",
    "Ho aspettato 10 minuti ma non si risolve da solo."
]

VAR_IMPATTO = [
    "Tutto il reparto è fermo.",
    "Non riusciamo a lavorare.",
    "Stiamo accumulando ritardo sulle consegne.",
    "I colleghi stanno aspettando questo per procedere.",
    "Abbiamo una riunione importante tra un'ora.",
    "C'è una scadenza oggi pomeriggio.",
    "Il cliente sta aspettando una risposta.",
    "Siamo bloccati su una commessa urgente.",
    "Questo blocca tutta la pipeline del reparto."
]

# frasi di chiusura per ticket a bassa/media urgenza.
VAR_TONO_MILD = [
    "Non è urgentissimo, ma prima si risolve meglio è.",
    "Quando potete, grazie.",
    "Non è bloccante ma rallenta il lavoro.",
    "Appena avete tempo.",
    "Non è prioritario, ma va sistemato entro fine settimana.",
    "Senza fretta, ma sarebbe utile risolvere.",
    "Non è critico, solo scomodo.",
    "Fatemi sapere come procedere.",
    "Non è un'emergenza, ma se potete dateci un'occhiata.",
    "Ho già trovato un workaround temporaneo.",
    "Non blocca il lavoro ma sarebbe utile sistemarlo.",
    "Prima o poi va risolto.",
    "Grazie per il supporto.",
    "Non è urgente, ma vorrei capire come mai accade.",
    "Resto a disposizione se serve altro.",
]

# Suffissi aggiuntivi per ticket bassa
VAR_BASSA_SUFFIX = [
    "",
    " Grazie in anticipo.",
    " Fammi sapere a chi mi devo rivolgere.",
    " Ho cercato sul portale ma non ho trovato niente.",
    " È una cosa nuova per me.",
    " Mi serve entro fine settimana se possibile.",
    " È da ieri che ci provo senza successo.",
    " Ho chiesto a un collega ma non sapeva nemmeno lui.",
    " Resto a disposizione per ulteriori dettagli.",
    " Se serve posso fornire ulteriori informazioni.",
    " Ho provato a cercare nella documentazione interna ma non sono riuscito a trovare nulla.",
    " È la prima volta che me lo chiedono.",
    " Grazie per il supporto.",
]

# Variabili contestuali per la grey zone
VAR_GZ_CONTEXT = [
    " Uso Windows 11 sulla mia postazione.",
    " È la prima volta che mi succede.",
    " Ho già riavviato il PC senza risultati.",
    " Succede sia da ufficio che da remoto.",
    " Ho notato questo da dopo l'ultimo aggiornamento.",
    " Anche un collega ha lo stesso problema.",
    " Succede solo al mattino, non nel pomeriggio.",
    " Ho provato su due browser diversi, stessa cosa.",
    " L'IT lo sa già ma non si è ancora risolto.",
    " È la seconda volta questa settimana.",
    " Sembra peggiore nei momenti di picco.",
    " Non succedeva prima delle vacanze.",
    " Ho fatto uno screenshot dell'errore se serve.",
    " Succede anche alla collega della scrivania accanto.",
    " Ho aggiornato il software ieri, forse è collegato.",
    " Non so se è solo io o anche altri.",
    " Ho contattato il collega esperto ma non sa.",
    " Problema presente da circa 3 giorni.",
    " Si manifesta soprattutto nel pomeriggio.",
    " Ho già aperto un ticket simile il mese scorso.",
    " Uso la versione web, non quella desktop.",
    " Sto lavorando da casa oggi.",
    " Sono in ufficio e anche il collega accanto ha lo stesso problema.",
    " Non è cambiato nulla sul mio PC di recente.",
    " Ho provato a disconnettermi e riconnettermi.",
    " Il problema è comparso dopo il riavvio di stamattina.",
    " Uso questo strumento quotidianamente e prima funzionava.",
    " Ho verificato con il responsabile: non sa da che dipende.",
    " Mi succede solo su questo specifico computer.",
    " Ho controllato e non ci sono aggiornamenti in sospeso.",
]

# Suffissi anti-duplicati per ticket senza variabili {tempo}/{tentativo}/{impatto}/{ctx}.
_ANTI_DUP_SUFFIXES = [
    " Uso la versione desktop.",
    " Lavoro in smart working oggi.",
    " Sono alla postazione fissa in ufficio.",
    " Ho già fatto uno screenshot se serve.",
    " È la prima volta che mi succede questo.",
    " Succede anche al mio collega di scrivania.",
    " Ho aggiornato il browser ieri.",
    " Ho provato sia da Chrome che da Edge.",
    " Il mio PC è Windows 10.",
    " Sono rientrato oggi da una settimana di ferie.",
    " Ho già chiesto al responsabile e mi ha detto di aprire ticket.",
    " Ho verificato con un collega e lui non ha lo stesso problema.",
    " Ho riprovato stamattina, stessa situazione.",
    " Lo segnalo anche a nome di un altro collega che ha lo stesso problema.",
    " Se serve posso venire di persona all'IT.",
    " Ho controllato se ci sono aggiornamenti pendenti ma non ne trovo.",
    " È successo anche ieri ma pensavo si risolvesse da solo.",
    " Uso questo strumento ogni giorno, quindi noto la differenza.",
    " L'ho segnalato anche verbalmente all'IT venerdì scorso.",
    " Ho cercato online ma non ho trovato nulla di utile.",
]

# Variabili  ogni istanza prende un dettaglio diverso
VAR_MISTI_CTX = [
    "da stamattina",
    "da ieri pomeriggio",
    "dall'aggiornamento di lunedì",
    "da quando ho cambiato postazione",
    "da dopo l'intervento IT della scorsa settimana",
    "da questa mattina presto",
    "dal rientro dalle vacanze",
    "da circa due ore",
    "da ieri sera",
    "da quando abbiamo aggiornato il sistema",
    "da stamattina, senza preavviso",
    "dal rinnovo del contratto software",
    "da ieri, non so perché",
    "da questa settimana",
    "da quando è arrivato il nuovo collega",
    "dall'aggiornamento di venerdì scorso",
    "da oggi pomeriggio",
    "da stamattina, dopo il riavvio",
    "dall'installazione dell'ultimo aggiornamento",
    "da due giorni a questa parte",
]




# ---------------------------------------------------------------------------
# LESSICO TECNICO - ACCENSIONE / HARDWARE (ALTA)
# ---------------------------------------------------------------------------
_POWER_PAIRS = [
    ("PC non si accende",
     "Premendo il pulsante non appare nulla. {tentativo} {impatto}"),
    ("Il computer non parte all'accensione",
     "Il monitor resta nero {tempo}: nessun beep, nessun logo di avvio. {tentativo}"),
    ("Schermo nero all'avvio del PC",
     "All'avvio schermo completamente nero, nessun segnale. {tentativo}"),
    ("La postazione non si avvia",
     "La postazione è ferma {tempo}: non si avvia e non appare alcuna schermata. {impatto}"),
    ("Il PC non dà segni di vita",
     "Ventole ferme, nessuna luce, schermo nero. {tentativo} {impatto}"),
    ("Computer bloccato, non riparte",
     "Si è spento all'improvviso {tempo} e ora non si riaccende più. {tentativo}"),
    ("Postazione ferma, impossibile lavorare",
     "Il PC non si avvia {tempo}. {tentativo} Ho bisogno di assistenza urgente."),
    ("PC spento di colpo, ora non riparte",
     "Si è spento all'improvviso durante l'uso. {tentativo} Schermo nero."),
    ("Monitor nero, pc completamente fermo",
     "Ho acceso il PC {tempo} ma schermo nero totale. {tentativo}"),
    ("Avvio fallito, postazione inutilizzabile",
     "Il sistema non si avvia {tempo}. {tentativo} {impatto}"),
    ("Errore all'avvio, PC bloccato prima del login",
     "Appena acceso si blocca prima della schermata di login. {tentativo}"),
    ("PC non si riaccende dopo aggiornamento",
     "Dopo l'aggiornamento automatico di stanotte il PC non si avvia più. {impatto}"),
]

BLOCKER_POWER_TERMS = [
    "non si accende", "non parte il pc", "non parte il computer",
    "schermo nero", "nessuna schermata",
    "non si avvia il pc", "non si avvia il computer",
    "avvio bloccato", "avvio fallito", "errore in avvio",
    "postazione ferma", "pc spento", "non riparte"
]


# ---------------------------------------------------------------------------
# LESSICO TECNICO - STAMPANTE ALTA
# ---------------------------------------------------------------------------
_STAMPA_ALTA_PAIRS = [
    ("Stampante bloccata, ho scadenze oggi",
     "La stampante non risponde {tempo}. Devo stampare le fatture entro oggi. {impatto}"),
    ("Impossibile stampare documenti fiscali urgenti",
     "Stampa completamente ferma. Ho documenti fiscali da consegnare entro le 17:00. {impatto}"),
    ("Stampa ferma: contratti da firmare entro oggi",
     "Coda di stampa bloccata {tempo}. Ho contratti urgenti da stampare. {tentativo}"),
    ("Urge sblocco stampante, chiusura contabile aperta",
     "La stampante condivisa è offline. Stiamo facendo la chiusura contabile. {impatto}"),
    ("Stampante offline nel reparto amministrazione",
     "Stampante offline {tempo}. Il reparto non riesce a produrre nessun documento. {tentativo}"),
    ("Blocco totale stampa, tutto il reparto fermo",
     "Non si stampa da nessuna postazione del piano {tempo}. {tentativo} {impatto}"),
    ("Coda di stampa bloccata, scadenza fiscale oggi",
     "Driver di stampa corrotto dopo l'aggiornamento. Non stampa nessuno. Scadenza oggi."),
    ("Nessuno riesce a stampare, sistema di stampa giù",
     "Il sistema di stampa non funziona {tempo}. Tutte le postazioni bloccate. {impatto}"),
    ("Stampa bloccata prima della chiusura mensile",
     "Non riesco a stampare i report di chiusura. La stampante dà errore e non risponde. {tentativo}"),
    ("Impossibile stampare da qualsiasi PC del piano",
     "Ho provato da tre postazioni diverse, la stampante non risponde su nessuna. {impatto}"),
]


# ---------------------------------------------------------------------------
# LESSICO TECNICO - STAMPANTE MEDIA
# ---------------------------------------------------------------------------
_STAMPA_MEDIA_PAIRS = [
    ("Stampante molto lenta, ci mette minuti per foglio",
     "La stampante funziona ma è lentissima {tempo}: ogni foglio ci mette quasi un minuto. Fatemi sapere."),
    ("La stampa esce storta, fogli inclinati",
     "I fogli escono leggermente inclinati. Non blocca ma è scomodo per i documenti formali. Non è prioritario."),
    ("Toner scarico: stampa sbiadita e illeggibile",
     "Il toner sembra agli sgoccioli: la stampa è pallida. Va sostituito presto. Quando potete, grazie."),
    ("Inceppamento carta ricorrente, 3-4 volte al giorno",
     "La carta si inceppa frequentemente {tempo}: circa 3 volte al giorno. {tentativo} Non è urgente."),
    ("Driver stampante obsoleto, errori sporadici",
     "Il driver sembra vecchio: ogni tanto va in errore ma poi riprende. {tentativo}"),
    ("Stampa solo in bianco e nero anche se chiedo a colori",
     "Ho impostato la stampa a colori ma escono sempre in B/N. Problema di configurazione? Appena avete tempo."),
    ("Rumore anomalo durante la stampa",
     "La stampante fa uno strano rumore durante la stampa {tempo}. Funziona, ma preoccupa."),
    ("Stampa fronte-retro non funziona sempre",
     "La fronte-retro funziona a volte sì e a volte no. {tentativo} Prima che diventi un problema."),
    ("Qualità stampa scaduta, righe orizzontali sui fogli",
     "Compaiono delle righe orizzontali su tutti i fogli stampati {tempo}. {tentativo}"),
    ("Stampante non comunicata su rete, appare offline a metà postazioni",
     "La stampante appare offline su alcune postazioni ma non su altre. {tentativo}"),
]


# ---------------------------------------------------------------------------
# LESSICO TECNICO - RETE ALTA
# ---------------------------------------------------------------------------
_RETE_ALTA_PAIRS = [
    ("Rete aziendale giù, nessuno si connette",
     "Da {tempo} nessuno riesce a connettersi alla rete aziendale. {impatto}"),
    ("VPN giù: tutto il team remoto bloccato",
     "La VPN è down {tempo}. Il team da remoto non riesce a lavorare. {tentativo}"),
    ("Internet offline in tutto l'ufficio",
     "Internet non funziona in tutto l'edificio {tempo}. Siamo completamente offline."),
    ("Server condiviso irraggiungibile, file non apribili",
     "Il server condiviso è irraggiungibile {tempo}. Non riusciamo ad aprire nessun file. {impatto}"),
    ("Rete aziendale down, lavoro completamente fermo",
     "La rete è down. {impatto} {tentativo}"),
    ("Cartelle di rete non accessibili da nessuna postazione",
     "Le cartelle di rete non sono accessibili da nessuna postazione del piano {tempo}. {impatto}"),
    ("Switch di piano offline, intero reparto senza rete",
     "Il switch del nostro piano sembra offline {tempo}. Nessuno ha connettività. {impatto}"),
    ("DNS aziendale non risolve, siti interni irraggiungibili",
     "Nessun sito interno risponde {tempo}. Probabile problema DNS. {tentativo}"),
    ("Proxy aziendale bloccato, impossibile navigare",
     "Il proxy non risponde {tempo}. Impossibile accedere a qualsiasi risorsa online. {impatto}"),
    ("Connettività assente su tutto il piano dopo aggiornamento",
     "Dopo l'aggiornamento notturno nessuno ha connettività. {tentativo} {impatto}"),
]


# ---------------------------------------------------------------------------
# LESSICO TECNICO - RETE MEDIA
# ---------------------------------------------------------------------------
_RETE_MEDIA_PAIRS = [
    ("Connessione VPN molto lenta ultimamente",
     "La VPN è lentissima {tempo}: ci vuole un'eternità ad aprire i file. Non è critico."),
    ("WiFi instabile sulla mia postazione",
     "Il WiFi si stacca e riconnette spesso {tempo}. {tentativo} Quando avete un momento."),
    ("Timeout occasionale sul server di rete",
     "Il server va in timeout circa una volta all'ora {tempo}, poi si risolve. Senza fretta."),
    ("Problemi intermittenti di rete, difficili da riprodurre",
     "Ho disconnessioni saltuarie {tempo}: funziona, poi smette, poi riprende. Difficile da catturare."),
    ("Rete a singhiozzo, rallentamenti continui",
     "La rete va a singhiozzo {tempo}. Non è sempre bloccante ma rallenta molto. Appena possibile, grazie."),
    ("Disconnessioni sporadiche dalla rete aziendale",
     "Mi disconnetto dalla rete 3-4 volte al giorno {tempo}. Non critico ma fastidioso. Non è un'emergenza."),
    ("VPN si disconnette ogni 20 minuti circa",
     "La VPN cade ogni 20 minuti circa {tempo}. Devo riconnettermi manualmente ogni volta."),
    ("Latenza alta sulla rete locale, operazioni lente",
     "La latenza sulla rete interna è molto alta {tempo}. Le operazioni sul gestionale si trascinano."),
    ("File server lento, apertura documenti richiede minuti",
     "L'apertura dei file dal server richiede 2-3 minuti {tempo}. Prima era immediata. {tentativo}"),
    ("Connessione instabile solo su alcune postazioni",
     "Solo alcune postazioni hanno problemi di rete {tempo}, le altre vanno bene. Strana selettività."),
]


# ---------------------------------------------------------------------------
# LESSICO TECNICO - APPLICAZIONI ALTA
# ---------------------------------------------------------------------------
_APP_ALTA_PAIRS = [
    ("Gestionale non si avvia su nessun PC del reparto",
     "Il gestionale non si apre su nessuna macchina {tempo}. {tentativo} {impatto}"),
    ("Applicativo va in crash appena avviato, dopo aggiornamento",
     "Dopo l'aggiornamento {tempo} il software crasha appena si tenta di aprirlo. {impatto}"),
    ("CRM irraggiungibile, team commerciale fermo",
     "Il CRM è irraggiungibile {tempo}. {impatto} {tentativo}"),
    ("Gestionale in crash ogni 5 minuti, impossibile lavorare",
     "Il gestionale crasha ogni pochi minuti {tempo}. Impossibile completare qualsiasi operazione."),
    ("Errore fatale all'avvio del software, si chiude subito",
     "All'avvio compare un errore fatale e il programma si chiude immediatamente. {tentativo} {impatto}"),
    ("Nessuno riesce ad aprire il gestionale, schermata bloccata",
     "Il gestionale si apre ma resta bloccato sulla schermata di caricamento {tempo}. {tentativo}"),
    ("Aggiornamento automatico ha rotto il software gestionale",
     "Dopo l'aggiornamento automatico di stanotte il software non parte più. {impatto}"),
    ("Licenza software scaduta, accesso negato a tutti",
     "Il software blocca l'avvio con errore di licenza {tempo}. {impatto}"),
    ("Software di contabilità non risponde, cursore a clessidra",
     "Il software di contabilità si apre ma poi non risponde più. Cursore a clessidra fisso. {tentativo}"),
    ("Errore di database all'avvio, dati non caricabili",
     "All'avvio compare un errore di connessione al database {tempo}. {tentativo} {impatto}"),
]


# ---------------------------------------------------------------------------
# LESSICO TECNICO - APPLICAZIONI MEDIA
# ---------------------------------------------------------------------------
_APP_MEDIA_PAIRS = [
    ("Gestionale molto più lento del solito",
     "Il gestionale è lentissimo {tempo}. Non è bloccante ma rallenta molto il lavoro. Grazie in anticipo."),
    ("Errore sporadico nell'applicativo, si risolve riaprendo",
     "Ogni tanto compare un errore, ma basta chiudere e riaprire. {tempo} Succede 2-3 volte al giorno."),
    ("Il software si blocca sulla schermata di reportistica",
     "Il software si blocca solo sul modulo report {tempo}. Le altre funzioni vanno bene. {tentativo}"),
    ("Bug nella funzione di esportazione CSV",
     "L'esportazione CSV fallisce una volta su tre {tempo}. Ho un workaround ma non è comodo."),
    ("L'applicativo non salva sempre i dati correttamente",
     "A volte i dati salvati non appaiono subito, bisogna ricaricare {tempo}. Problema di cache?"),
    ("Visualizzazione sballata del modulo ordini su widescreen",
     "Il modulo ordini è mal formattato su monitor widescreen {tempo}. Funziona ma è difficile da leggere."),
    ("Aggiornamento software ha introdotto un bug sulla ricerca",
     "Dopo l'ultimo aggiornamento la ricerca avanzata non restituisce risultati corretti. {tentativo}"),
    ("Notifiche email dell'applicativo non arrivano più",
     "Le notifiche automatiche del gestionale non arrivano più {tempo}. Gli altri workflow funzionano."),
    ("Stampa da applicativo produce file vuoto",
     "Quando stampo da dentro il gestionale il file PDF esce vuoto {tempo}. {tentativo}"),
    ("Sessione scade troppo presto, disconnessione forzata",
     "La sessione scade dopo pochi minuti di inattività {tempo}. Va rieffettuato il login troppo spesso."),
]


# ---------------------------------------------------------------------------
# LESSICO TECNICO - RICHIESTE BASSA
# ---------------------------------------------------------------------------
_SW_NAMES = [
    "Adobe Acrobat Reader", "Zoom", "Microsoft Teams", "Slack",
    "Office 365", "il gestionale ABC", "il software di contabilità",
    "AutoCAD LT", "Power BI Desktop", "il client VPN aziendale",
    "7-Zip", "Notepad++", "il software di firma digitale"
]

_SYSTEM_NAMES = [
    "il portale HR", "il CRM aziendale", "il gestionale",
    "l'area riservata del portale", "il drive condiviso del team",
    "il sistema documentale", "la VPN aziendale",
    "il portale fornitori", "la piattaforma e-learning"
]

_RICHIESTE_PAIRS = [
    ("Richiesta installazione {sw}",
     "Avrei bisogno di installare {sw} sulla mia postazione per le nuove attività assegnate. Non è bloccante."),
    ("Serve {sw} sul mio PC",
     "Potete installare {sw} sul mio computer? Mi serve per completare un progetto in corso. Prima o poi va sistemato."),
    ("Richiesta accesso a {system}",
     "Ho bisogno di accedere a {system}. Come posso fare richiesta formale? Non blocca il lavoro."),
    ("Configurazione accessi per nuovo collega",
     "È arrivato un nuovo collega e vanno configurati email e accesso a {system}. Fatemi sapere."),
    ("Permessi aggiuntivi su {system}",
     "Ho bisogno di permessi aggiuntivi su {system} per alcune funzioni che mi servono. Non è prioritario."),
    ("Richiesta licenza per {sw}",
     "Serve una licenza aggiuntiva per {sw} per il nuovo membro del team. Quando potete, grazie."),
    ("Setup postazione nuovo dipendente",
     "Abbiamo un nuovo dipendente da lunedì. Va configurata la postazione con accesso a {system}. Non è urgente."),
    ("Aggiunta stampante di piano alla postazione",
     "Vorrei aggiungere la stampante del corridoio alla mia postazione. Come si fa? Appena avete tempo."),
    ("Accesso alla cartella condivisa del progetto",
     "Vorrei accedere alla cartella condivisa del progetto corrente. Chi devo contattare? Prima che diventi un problema."),
    ("Firma email aziendale non configurata su Outlook",
     "La firma email standard non è configurata sul mio Outlook. Può sistemarmela? Non è critico."),
    ("Richiesta account su {system} per consulente esterno",
     "Abbiamo un consulente esterno che avrà bisogno di accesso a {system} per 3 mesi. Quando avete un momento."),
    ("Aggiornamento permessi dopo cambio ruolo",
     "Ho cambiato ruolo e ho bisogno di accedere a {system} con nuovi permessi. Senza fretta."),
]


# ---------------------------------------------------------------------------
# LESSICO AMMINISTRAZIONE - ALTA (chiusure bloccate)
# ---------------------------------------------------------------------------
_AMM_ALTA_PAIRS = [
    ("Errore critico nella chiusura contabile mensile",
     "Durante la chiusura contabile mensile il sistema dà errore e non permette di completare. Scadenza oggi. {impatto}"),
    ("Impossibile completare la chiusura fiscale, sistema bloccato",
     "Errore critico nella chiusura fiscale: la procedura si blocca all'ultimo step. Scadenza imminente."),
    ("Gestionale in errore durante chiusura contabile",
     "Il gestionale va in errore sempre allo stesso punto della chiusura mensile. {tentativo} {impatto}"),
    ("Chiusura mensile bloccata, ho già riprovato tre volte",
     "Devo chiudere il mese oggi ma il sistema continua a darmi errore. {tentativo} Urgente."),
    ("Scadenza fiscale oggi, chiusura impossibile da completare",
     "C'è una scadenza fiscale oggi pomeriggio e il gestionale blocca la chiusura. {impatto}"),
    ("Procedura di chiusura contabile si arresta sempre allo stesso punto",
     "La chiusura si blocca sempre nello stesso step {tempo}. {tentativo} Non riesco ad andare avanti."),
    ("Sistema contabile inaccessibile a fine mese, scadenze a rischio",
     "Il portale contabile è irraggiungibile proprio a ridosso della chiusura mensile. {impatto}"),
    ("Errore nel modulo di chiusura fiscale annuale",
     "Il modulo di chiusura fiscale annuale dà errore alla validazione finale. {tentativo} Urgente."),
]


# ---------------------------------------------------------------------------
# LESSICO AMMINISTRAZIONE - MEDIA (anomalie su documenti)
# ---------------------------------------------------------------------------
_AMM_MEDIA_PAIRS = [
    ("Dati di fatturazione non aggiornati nel gestionale",
     "Nel gestionale sono ancora visibili i dati di fatturazione precedenti {tempo}. L'aggiornamento è stato recepito?"),
    ("Anomalia su nota di credito: importo errato",
     "Una nota di credito risulta con importo errato {tempo}. Va sistemata prima della prossima chiusura. {tentativo}"),
    ("Fattura non rintracciabile nel sistema",
     "Non riesco a trovare una fattura nel sistema {tempo}. Probabile problema di inserimento o ricerca."),
    ("Pagamento risulta duplicato nell'estratto conto",
     "Un pagamento appare due volte nell'estratto conto {tempo}. Serve verifica prima del riconcilio."),
    ("Importo fattura non corrisponde all'ordine originale",
     "L'importo di una fattura non corrisponde all'ordine di riferimento {tempo}. Va verificato."),
    ("Discrepanza tra fattura emessa e importo concordato",
     "C'è una discrepanza tra l'importo fatturato e quello concordato con il fornitore {tempo}. Va chiarita prima della prossima scadenza."),
    ("Estratto conto con saldi non coerenti",
     "L'estratto conto mostra saldi che non tornano rispetto alle transazioni del periodo {tempo}."),
    ("Modifica dati anagrafici fornitore non recepita",
     "Ho modificato i dati anagrafici di un fornitore ma nel gestionale appaiono ancora quelli vecchi {tempo}."),
    ("Nota spese approvata ma rimborso non ancora accreditato",
     "Ho una nota spese approvata da {tempo} ma il rimborso non è ancora arrivato. Ho già controllato con l'ufficio pagamenti."),
    ("Errore nell'importazione del file contabile mensile",
     "L'importazione del file contabile mensile dà errore a metà processo {tempo}. {tentativo}"),
]


# ---------------------------------------------------------------------------
# LESSICO AMMINISTRAZIONE - BASSA (info, richieste doc)
# ---------------------------------------------------------------------------
_AMM_BASSA_TOPICS = [
    ("nota spese",
     "Come funziona la procedura nota spese?",
     "Vorrei capire come funziona la procedura di rimborso nota spese. Dove trovo la documentazione?"),
    ("rimborso spese",
     "Rimborso spese: come si compila la richiesta?",
     "Non ho mai compilato una richiesta di rimborso spese. C'è una guida o un modulo da seguire?"),
    ("fatturazione",
     "Informazioni sulla procedura di fatturazione",
     "Avrei qualche dubbio sul processo di fatturazione verso i clienti. Chi posso contattare?"),
    ("rinnovo contratto fornitore",
     "Rinnovo contratto fornitore: scadenze e procedura",
     "Ho un contratto fornitore in scadenza e non so come gestire il rinnovo. Come si procede?"),
    ("modulistica fornitori",
     "Dove trovo la modulistica per i nuovi fornitori?",
     "Sto inserendo un nuovo fornitore e non trovo i moduli necessari sul portale. Dove sono?"),
    ("chiusura fine anno",
     "Procedura di chiusura di fine anno: chiarimenti",
     "È la prima volta che partecipo alla chiusura di fine anno. C'è una guida con i passi da seguire?"),
    ("approvazione pagamenti",
     "Come funziona il processo di approvazione pagamenti?",
     "Ho bisogno di capire chi approva i pagamenti e in che tempistiche. Potete indicarmi la procedura?"),
    ("note di credito",
     "Informazioni sulla gestione delle note di credito",
     "Non ho mai emesso una nota di credito. C'è una procedura definita? A chi mi rivolgo?"),
    ("dati anagrafici aziendali",
     "Aggiornamento dati anagrafici: a chi si chiede?",
     "Devo aggiornare alcuni dati anagrafici dell'azienda nel portale. Chi è il referente?"),
    ("duplicato fattura",
     "Richiesta duplicato fattura già emessa",
     "Ho bisogno di un duplicato di una fattura emessa il mese scorso. Come lo richiedo?"),
    ("scadenze fiscali",
     "Conferma scadenze fiscali del prossimo trimestre",
     "Potete confermarmi le scadenze fiscali del prossimo trimestre? Voglio pianificare per tempo."),
    ("procedura bonifici",
     "Come si inserisce un bonifico nel portale amministrativo?",
     "È la prima volta che inserisco un bonifico nel portale. C'è una procedura guidata o un tutorial?"),
]


# ---------------------------------------------------------------------------
# LESSICO COMMERCIALE - ALTA (ordini e flussi bloccati)
# ---------------------------------------------------------------------------
_COMM_ALTA_PAIRS = [
    ("Le conferme d'ordine non vengono inviate ai clienti",
     "Da {tempo} le email di conferma d'ordine non partono. I clienti ci stanno chiamando. {impatto}"),
    ("Sistema commerciale bloccato: impossibile inserire ordini",
     "Il sistema commerciale non accetta nuovi ordini {tempo}. {tentativo} {impatto}"),
    ("Impossibile creare nuovi ordini nel gestionale",
     "La procedura di creazione ordine si blocca all'ultimo step {tempo}. {tentativo} {impatto}"),
    ("Conferme automatiche d'ordine non partono, clienti senza risposta",
     "Le conferme automatiche non vengono inviate {tempo}. I clienti non ricevono nessuna notifica."),
    ("Ordini in attesa non processati, clienti insoddisfatti",
     "Abbiamo ordini fermi da {tempo} che non vengono processati. I clienti iniziano a lamentarsi."),
    ("Flusso ordini completamente interrotto, perdita di vendite",
     "Il flusso degli ordini è bloccato {tempo}. Stiamo perdendo opportunità commerciali. {impatto}"),
    ("CRM non raggiungibile, team commerciale fermo",
     "Il CRM è irraggiungibile {tempo}. {tentativo} Il team commerciale non può operare."),
    ("Integrazione e-commerce → gestionale rotta, ordini non recepiti",
     "Gli ordini del sito web non arrivano più nel gestionale {tempo}. {tentativo} {impatto}"),
]


# ---------------------------------------------------------------------------
# LESSICO COMMERCIALE - MEDIA (anomalie dati, permessi, report)
# ---------------------------------------------------------------------------
_COMM_MEDIA_PAIRS = [
    # Report/dashboard
    ("Report vendite non aggiornato nel portale",
     "Il report vendite mostra dati di {tempo} che non corrispondono al reale. Va verificato."),
    ("Dashboard commerciale con KPI incoerenti",
     "La dashboard ha dati incoerenti {tempo}: i KPI non sono allineati con il CRM."),
    ("Statistiche di vendita errate nel portale commerciale",
     "Le statistiche del mese scorso sembrano errate: c'è una discrepanza significativa {tempo}."),
    ("KPI vendite nel portale non corrispondono al CRM",
     "I KPI nel portale non corrispondono ai dati del CRM {tempo}. Problema di sincronizzazione?"),
    ("Valori del report mensile non tornano prima della riunione",
     "Il report mensile ha valori che non quadrano {tempo}. Va sistemato prima della riunione con il direttore."),
    # Permessi
    ("Permessi mancanti sul portale vendite per modificare offerte",
     "Non ho i permessi per modificare le condizioni economiche delle offerte sul portale. Non è bloccante."),
    ("Impossibile modificare le offerte nel CRM",
     "Vorrei modificare alcune offerte nel CRM ma non ho i permessi necessari {tempo}. A chi li chiedo?"),
    # Consegne in ritardo
    ("Ordine cliente non consegnato nei tempi concordati",
     "L'ordine era previsto per lunedì e non è ancora arrivato {tempo}. Il tracking non si aggiorna. {impatto}"),
    ("Spedizione in ritardo, cliente al secondo sollecito",
     "La spedizione è in ritardo di 3 giorni sulla data prevista {tempo}. Come gestiamo il cliente?"),
    ("Tracking fermo da giorni, merce introvabile",
     "Il tracking non si aggiorna {tempo}. Non sappiamo dove si trova la merce. Il cliente aspetta."),
    ("Ordine risulta consegnato ma il cliente non ha ricevuto nulla",
     "Nel sistema risulta consegnato ma il cliente dice di non aver ricevuto nulla {tempo}."),
]


# ---------------------------------------------------------------------------
# LESSICO COMMERCIALE - BASSA (info, listini, preventivi)
# ---------------------------------------------------------------------------
_COMM_BASSA_PAIRS = [
    # Listini / preventivi
    ("Richiesta listino prezzi aggiornato",
     "Avrei bisogno del listino prezzi aggiornato per preparare un'offerta. Dove lo trovo? Prima o poi va sistemato."),
    ("Template preventivo: dove si trova?",
     "Devo preparare un preventivo per un cliente. Dove sono i template aggiornati? Non blocca il lavoro."),
    ("Come si crea un preventivo nel CRM?",
     "Non ho mai creato un preventivo nel CRM. C'è una guida passo-passo o qualcuno che può aiutarmi?"),
    ("Richiesta catalogo prodotti aggiornato per riunione",
     "Ho una riunione con un cliente e mi serve il catalogo prodotti aggiornato. Dove lo trovo? Fatemi sapere."),
    # Documentazione / procedure
    ("Dove si trova la documentazione commerciale standard?",
     "Cerco i materiali standard per preparare un'offerta ma non riesco a trovarli nel portale. Non è prioritario."),
    ("Procedura di approvazione offerte: come funziona?",
     "Non sono chiare le tempistiche di approvazione delle offerte. Chi si occupa della validazione?"),
    ("Informazioni sulle promozioni attive per i clienti",
     "Ho bisogno di sapere quali promozioni sono attive al momento per un cliente interessato. Quando potete, grazie."),
    ("Condizioni commerciali per cliente estero: a chi chiedo?",
     "Sto preparando un'offerta per un cliente straniero. Chi gestisce le condizioni per l'export?"),
    ("Template per proposta commerciale standard",
     "Esiste un template standard per le proposte commerciali? Non voglio partire da zero. Non è urgente."),
    ("Tempi di consegna standard: dove li trovo?",
     "Il cliente chiede i tempi di consegna standard. Dove trovo questa informazione? Appena avete tempo."),
    ("Come inserire un nuovo cliente nel CRM?",
     "Devo inserire un nuovo cliente nel CRM ma non ho mai fatto questa operazione. Come si fa?"),
    ("Accesso al portale vendite: come si richiede per un agente nuovo?",
     "Abbiamo un nuovo agente commerciale che ha bisogno di accesso al portale vendite. Come si procede?"),
]

# ---------------------------------------------------------------------------
# GREY ZONE — ticket media e bassa con vocabolario condiviso
# Regola di etichettatura:
#   BASSA = problema isolato, non impatta altri, workaround disponibile
#   MEDIA = problema ricorrente O impatta il flusso di lavoro O su più utenti
# La differenza è sottile e non emerge dal vocabolario 
# ---------------------------------------------------------------------------
_GREY_ZONE_BASSA = [
    # Titoli e vocabolario IDENTICI a _GREY_ZONE_MEDIA — differenza solo nel contesto
    ("Problema ricorrente nel sistema",
     "Il sistema ogni tanto non funziona su una schermata. Riesco a completare il lavoro usando un percorso alternativo. Segnalo per conoscenza."),
    ("File che non si apre sempre",
     "Ogni tanto il file non si apre al primo tentativo. Riesco ad aprirlo al secondo. Non è bloccante ma succede spesso."),
    ("Report con dati che sembrano non aggiornati",
     "Il report mostra dati che sembrano non aggiornati. Riesco a lavorare comunque. Non è urgente, solo una segnalazione."),
    ("Funzione del gestionale che non risponde sempre",
     "Una funzione del gestionale ogni tanto non risponde subito. Riesco ad usarla aspettando qualche secondo. Non blocca."),
    ("Connessione che si interrompe ogni tanto",
     "La connessione si interrompe ogni tanto ma si riconnette subito da sola. Riesco a lavorare normalmente."),
    ("File che non si salva al primo tentativo",
     "Il file ogni tanto non salva correttamente. Riesco sempre a salvarlo riprovando. Non è urgente."),
    ("Sistema che sembra più lento del solito",
     "Il sistema sembra più lento in certi momenti della giornata. Riesco a lavorare ma ci vuole più tempo. Non è urgente."),
    ("Dati nel portale che sembrano non allineati",
     "I dati nel portale sembrano non allineati rispetto a quello che mi aspettavo. Forse è normale. Non è urgente."),
    ("Funzione di esportazione che ogni tanto fallisce",
     "La funzione di esportazione ogni tanto fallisce. Riesco ad esportare riprovando dopo qualche minuto. Non è bloccante."),
    ("Il sistema non risponde su una schermata",
     "Su una schermata il sistema ogni tanto non risponde. Riesco a bypassarla da un altro menu. Segnalazione per conoscenza."),
    ("Report che a volte non carica",
     "Il report a volte non carica correttamente. Riesco a vederlo ricaricando la pagina. Non è urgente."),
    ("Connessione instabile sul mio PC",
     "La connessione sul mio PC sembra instabile ogni tanto. Devo riconnettermi a volte. Riesco a lavorare."),
    ("Problema sul sistema di report",
     "Ho un problema sul sistema di report che si manifesta ogni tanto. Riesco a lavorare ma il problema persiste."),
    ("File che sembrano persi dopo il salvataggio",
     "Ogni tanto un file sembra non essere stato salvato. Poi lo ritrovo. Forse è un problema di sincronizzazione. Non blocca."),
    ("Sistema che non aggiorna i dati subito",
     "Il sistema non aggiorna i dati immediatamente. Devo ricaricare per vederli aggiornati. Non è urgente."),
]

_GREY_ZONE_MEDIA = [
    # Stessi titoli di _GREY_ZONE_BASSA — il modello non può distinguerli dal titolo
    ("Problema ricorrente nel sistema",
     "Il sistema non funziona correttamente da alcuni giorni su più schermate. Non riesco sempre a completare il lavoro. Va sistemato."),
    ("File che non si apre sempre",
     "I file spesso non si aprono. Ogni tanto riesco a bypassare il problema ma è ricorrente e rallenta tutto il reparto."),
    ("Report con dati che sembrano non aggiornati",
     "Il report mostra dati sbagliati. Non riesco a usarlo per le analisi settimanali. Impatta le decisioni. Va verificato."),
    ("Funzione del gestionale che non risponde sempre",
     "Una funzione del gestionale non funziona per diversi colleghi. Ogni tanto riesco ad usarla, ma spesso no. È sistematico."),
    ("Connessione che si interrompe ogni tanto",
     "La connessione è instabile e si interrompe per minuti. Non riesco a lavorare in modo continuativo. Si ripresenta ogni giorno."),
    ("File che non si salva al primo tentativo",
     "I file spesso non si salvano. A volte sembra che i dati vengano persi. Non riesco a capire quando il salvataggio funziona."),
    ("Sistema lento in modo sistematico, scadenze a rischio",
     "Il sistema è lento in modo sistematico per tutto il reparto. Ogni operazione richiede il doppio del tempo. Ho scadenze oggi che non riesco a rispettare."),
    ("Dati nel portale che sembrano non allineati",
     "I dati nel portale sono sempre sbagliati. Il report non è affidabile. Non riesco ad usarlo per le analisi. Va verificato urgente."),
    ("Funzione di esportazione che ogni tanto fallisce",
     "La funzione di esportazione fallisce spesso. Ogni tanto riesco con un percorso alternativo ma non è sempre possibile. Impatta."),
    ("Il sistema non risponde su una schermata",
     "Il sistema non risponde su diverse schermate. Non riesco a completare le operazioni standard. Il problema è su più postazioni."),
    ("Report che a volte non carica",
     "Il report non carica da ieri. Non riesco a preparare i dati per la riunione di domani. Va risolto prima possibile."),
    ("Connessione instabile sul mio PC",
     "La connessione sul mio PC è instabile. Mi disconnetto ogni tanto e devo riconnettermi. Rallenta il lavoro ma non blocca tutto il reparto."),
    ("Problema sul sistema di report",
     "Il problema sul sistema di report si manifesta ogni giorno. Non riesco più a generare i report settimanali affidabili."),
    ("File che sembrano persi dopo il salvataggio",
     "I file sembrano perdersi dopo il salvataggio. Ho già perso del lavoro. Non riesco a capire quando succede. Va risolto."),
    ("Sistema che non aggiorna i dati subito",
     "Il sistema non aggiorna i dati e i colleghi lavorano su dati vecchi. Non riesco a garantire la coerenza delle informazioni."),
]
# ---------------------------------------------------------------------------
# TICKET VAGHI — nessuna keyword diagnostica, solo contesto generico
# Causa confusione diffusa
# ---------------------------------------------------------------------------
_VAGUE_PAIRS = [
    ("Non funziona come dovrebbe",
     "Da qualche giorno qualcosa non va. Ho provato diverse soluzioni ma il problema persiste. Potete aiutarmi?",
     "Tecnico", "media"),
    ("Problema che non riesco a risolvere",
     "C'è un problema che non riesco a risolvere da solo. Non so bene come descriverlo. Serve supporto.",
     "Tecnico", "media"),
    ("Qualcosa non va, segnalazione",
     "Ho notato qualcosa che non va come al solito. Non è sempre presente ma è abbastanza frequente da segnalarlo.",
     "Tecnico", "bassa"),
    ("Ho bisogno di aiuto su una cosa",
     "Avrei bisogno di aiuto su qualcosa che non riesco a fare. Non è urgente ma mi farebbe comodo risolvere.",
     "Amministrazione", "bassa"),
    ("Situazione da verificare",
     "C'è una situazione che mi sembra anomala. Preferirei parlarne con qualcuno di persona piuttosto che descriverla.",
     "Amministrazione", "media"),
    ("Non riesco ad andare avanti con il lavoro",
     "Sono bloccato su un passaggio e non riesco a procedere. Ho già provato ma non ce la faccio da solo.",
     "Tecnico", "media"),
    ("Richiesta supporto generico",
     "Ho bisogno di supporto. Non so esattamente a chi rivolgermi o come classificare il problema.",
     "Tecnico", "bassa"),
    ("Domanda su come fare una cosa",
     "Avrei una domanda su come si fa una cosa nel sistema. Non è urgente ma mi sarebbe utile saperlo.",
     "Amministrazione", "bassa"),
    ("Comportamento strano che appare ogni tanto",
     "Ogni tanto mi appare qualcosa di strano. Non so quando succede esattamente né come riprodurlo.",
     "Tecnico", "media"),
    ("Da sistemare quando possibile",
     "C'è qualcosa da sistemare. Non è una crisi ma prima lo si risolve meglio è, per evitare problemi futuri.",
     "Tecnico", "media"),
    ("Segnalazione per conoscenza",
     "Volevo solo segnalare qualcosa che ho notato. Non so se è normale. Fatemi sapere se va approfondito.",
     "Tecnico", "bassa"),
    ("Una cosa che non capisco",
     "Ho notato qualcosa nel sistema che non capisco. Forse è normale, forse no. Vorrei un chiarimento.",
     "Tecnico", "bassa"),
    ("Problema ricorrente da capire",
     "Ho un problema che si ripresenta. Non so se dipende da me o dal sistema. Vorrei capire la causa.",
     "Tecnico", "media"),
    ("Anomalia sul mio account",
     "Ci sono alcune cose sul mio account che non mi sembrano corrette. Non è urgente ma vorrei farle verificare.",
     "Amministrazione", "bassa"),
    ("Qualcosa cambiato dopo aggiornamento",
     "Dopo un aggiornamento qualcosa sembra diverso. Le cose funzionano ma non come mi aspettavo. Va bene così?",
     "Tecnico", "media"),
]

# ---------------------------------------------------------------------------
# TICKET MISTI — tecnico + admin nello stesso ticket
# Causa confusione di categoria (e anche di priorità)
# ---------------------------------------------------------------------------
_MIXED_PAIRS = [
    ("Stampante rotta e fatture da ristampare",
     "La stampante non funziona {ctx} e ho delle fatture urgenti da ristampare. Potete sistemare la stampante e dirmi come richiedere le copie?",
     "Tecnico", "alta"),
    ("Errore nel gestionale durante chiusura contabile",
     "Il gestionale dà errore quando accedo al modulo contabile {ctx}. Approfitto per chiedere anche quando scade la chiusura mensile.",
     "Tecnico", "media"),
    ("VPN giù, non riesco ad accedere ai documenti fiscali",
     "La VPN è down {ctx} e non riesco a connettermi. Ho bisogno dei documenti fiscali urgenti. Problema tecnico con impatto amministrativo.",
     "Tecnico", "alta"),
    ("Stampante non risponde, contratto da inviare firmato",
     "La stampante non risponde {ctx}. Devo stampare un contratto per un cliente. Se non riesco, come posso inviarlo firmato?",
     "Tecnico", "media"),
    ("CRM irraggiungibile, offerta urgente in scadenza",
     "Non accedo al CRM dalla mia postazione {ctx}. Avevo bisogno dei dati di un cliente per un'offerta con scadenza oggi.",
     "Tecnico", "alta"),
    ("Dati fatturazione da aggiornare, gestionale in errore",
     "Volevo aggiornare i dati di fatturazione di un fornitore ma il gestionale dà errore su quella schermata {ctx}. Problema tecnico o permessi?",
     "Amministrazione", "media"),
    ("Report vendite con valori errati, accesso portale da configurare",
     "Il report vendite ha valori strani {ctx}. Ho anche bisogno di accesso aggiuntivo al portale per esportare i dati. Due problemi separati.",
     "Commerciale", "media"),
    ("Gestionale offline, devo chiudere l'ordine entro oggi",
     "Il gestionale è offline {ctx} e devo completare la chiusura di un ordine importante entro fine giornata. Serve accesso urgente.",
     "Tecnico", "alta"),
    ("Stampante di reparto offline, scadenza stampa oggi",
     "La stampante condivisa è offline {ctx}. Tutto il reparto non riesce a stampare. Abbiamo documenti urgenti da produrre.",
     "Tecnico", "alta"),
    ("VPN instabile, accesso ai dati amministrativi bloccato",
     "La VPN si connette ma poi cade {ctx}. Non riesco ad accedere ai dati amministrativi per la riunione di domani. Ho anche una domanda sulla procedura.",
     "Tecnico", "media"),
    ("PC non si avvia, documento urgente da consegnare",
     "Il mio PC non si avvia {ctx}. Devo consegnare un documento urgente entro stamattina. Posso usare il PC di un collega?",
     "Tecnico", "alta"),
    ("Accesso negato al portale, offerta da inviare oggi",
     "Non riesco ad accedere al portale vendite {ctx}. Ho un'offerta da inviare a un cliente entro oggi. È un problema di permessi?",
     "Commerciale", "alta"),
]

@st.cache_data
def generate_ticket_dataset(n_tickets=1000, random_state=42) -> pd.DataFrame:
    np.random.seed(random_state)
    rows = []

    cats  = ["Tecnico", "Amministrazione", "Commerciale"]
    probs = [0.43, 0.32, 0.25]
    P_GREY    = 0.35   # 35% ticket con ambiguità tra priorità media e bassa
    P_VAGUE   = 0.43   # +8% ticket con descrizioni vaghe
    P_MIXED   = 0.48   # +5% ticket con linguaggio misto tecnico/amministrativo
    P_LABEL   = 0.54   # +6% label noise (priorità spostata di ±1 livello)

    for i in range(n_tickets):
        cat = np.random.choice(cats, p=probs)
        noise_roll = np.random.rand()

        def pick(pairs, **kwargs):
            tpl_t, tpl_b = pairs[np.random.randint(len(pairs))]
            full_kw = dict(
                tempo    = np.random.choice(VAR_TEMPI),
                tentativo= np.random.choice(VAR_TENTATIVI),
                impatto  = np.random.choice(VAR_IMPATTO),
            )
            full_kw.update(kwargs)
            return _make_ticket(tpl_t, tpl_b, **full_kw)

        # ==================================================================
        # GREY ZONE media↔bassa (25%)
        # Ticket con vocabolario identico: stesse parole, etichette diverse.
        # È il meccanismo che genera la confusione media↔bassa.
        # ==================================================================
        if noise_roll < P_GREY:
            gz_ctx = np.random.choice(VAR_GZ_CONTEXT)
            if np.random.rand() < 0.50:
                rec = _GREY_ZONE_BASSA[np.random.randint(len(_GREY_ZONE_BASSA))]
                title, body = rec; pri = "bassa"
            else:
                rec = _GREY_ZONE_MEDIA[np.random.randint(len(_GREY_ZONE_MEDIA))]
                title, body = rec; pri = "media"
            rows.append([i + 1, title, _join_fragment(body, gz_ctx), "Tecnico", pri])
            continue

        # ==================================================================
        # TICKET VAGHI (8%) — nessuna keyword diagnostica
        # ==================================================================
        if noise_roll < P_VAGUE:
            rec = _VAGUE_PAIRS[np.random.randint(len(_VAGUE_PAIRS))]
            title, body, cat, pri = rec
            suffix = np.random.choice(VAR_BASSA_SUFFIX)
            ctx = np.random.choice(VAR_GZ_CONTEXT)
            body = _join_fragment(_join_fragment(body, suffix), ctx)
            rows.append([i + 1, title, body, cat, pri])
            continue

        # ==================================================================
        # TICKET MISTI (5%) — tecnico + admin → categoria e priorità ambigua
        # ==================================================================
        if noise_roll < P_MIXED:
            rec = _MIXED_PAIRS[np.random.randint(len(_MIXED_PAIRS))]
            title, body, cat, pri = rec
            ctx = np.random.choice(VAR_MISTI_CTX)
            body = _fix_italian(body.replace("{ctx}", ctx))
            rows.append([i + 1, title, body, cat, pri])
            continue

        # ==================================================================
        # LABEL NOISE (6%)
        # Simula errori umani di valutazione della gravità.
        # ==================================================================
        apply_label_noise = noise_roll < P_LABEL

        # ==================================================================
        # GENERAZIONE NORMALE (56%)
        # ==================================================================

        # TECNICO
        if cat == "Tecnico":
            area = np.random.choice(
                ["power", "stampa_alta", "stampa_media",
                 "rete_alta", "rete_media",
                 "app_alta", "app_media", "richiesta_bassa"],
                p=[0.08, 0.07, 0.10,
                   0.05, 0.10,
                   0.05, 0.15, 0.40]
            )
            if area == "power":
                title, body = pick(_POWER_PAIRS); pri = "alta"
            elif area == "stampa_alta":
                title, body = pick(_STAMPA_ALTA_PAIRS); pri = "alta"
            elif area == "stampa_media":
                title, body = pick(_STAMPA_MEDIA_PAIRS); pri = "media"
            elif area == "rete_alta":
                title, body = pick(_RETE_ALTA_PAIRS); pri = "alta"
            elif area == "rete_media":
                title, body = pick(_RETE_MEDIA_PAIRS); pri = "media"
            elif area == "app_alta":
                title, body = pick(_APP_ALTA_PAIRS); pri = "alta"
            elif area == "app_media":
                title, body = pick(_APP_MEDIA_PAIRS); pri = "media"
            else:
                sw = np.random.choice(_SW_NAMES)
                system = np.random.choice(_SYSTEM_NAMES)
                title, body = pick(_RICHIESTE_PAIRS, sw=sw, system=system)
                suffix = np.random.choice(VAR_BASSA_SUFFIX)
                body = _fix_italian(_join_fragment(body, suffix))
                pri = "bassa"

        # AMMINISTRAZIONE
        elif cat == "Amministrazione":
            pri_choice = np.random.choice(
                ["alta", "media", "bassa"], p=[0.10, 0.30, 0.60]
            )
            if pri_choice == "alta":
                title, body = pick(_AMM_ALTA_PAIRS); pri = "alta"
            elif pri_choice == "media":
                title, body = pick(_AMM_MEDIA_PAIRS); pri = "media"
            else:
                _, tpl_t, tpl_b = _AMM_BASSA_TOPICS[
                    np.random.randint(len(_AMM_BASSA_TOPICS))
                ]
                suffix = np.random.choice(VAR_BASSA_SUFFIX)
                title = _fix_italian(tpl_t)
                body  = _fix_italian(_join_fragment(tpl_b, suffix))
                pri = "bassa"

        # COMMERCIALE
        else:
            pri_choice = np.random.choice(
                ["alta", "media", "bassa"], p=[0.10, 0.32, 0.58]
            )
            if pri_choice == "alta":
                title, body = pick(_COMM_ALTA_PAIRS); pri = "alta"
            elif pri_choice == "media":
                title, body = pick(_COMM_MEDIA_PAIRS); pri = "media"
            else:
                tpl_t, tpl_b = _COMM_BASSA_PAIRS[
                    np.random.randint(len(_COMM_BASSA_PAIRS))
                ]
                suffix = np.random.choice(VAR_BASSA_SUFFIX)
                title = _fix_italian(tpl_t)
                body  = _fix_italian(_join_fragment(tpl_b, suffix))
                pri = "bassa"

        # ==================================================================
        # LABEL NOISE MIRATO media↔bassa (15% dei ticket media e bassa)
        # + label noise generico ±1 livello (6% su tutti)
        # ==================================================================
        if apply_label_noise:
            # Noise generico ±1 livello (sempre, 6%)
            pri_levels = ["bassa", "media", "alta"]
            idx_pri = pri_levels.index(pri)
            delta = np.random.choice([-1, 1])
            pri = pri_levels[max(0, min(2, idx_pri + delta))]
        elif pri in ("media", "bassa") and np.random.rand() < 0.37:
            # Noise mirato: flip media↔bassa (34% dei ticket media/bassa)
            pri = "bassa" if pri == "media" else "media"

        # ------------------------------------------------------------------
        # ANTI-DUPLICATI: inietta un suffisso contestuale casuale nei ticket
        # che non hanno già variabili {tempo}/{tentativo}/{impatto}/{ctx}.
        # ------------------------------------------------------------------
        _HAS_VAR = any(v in body for v in ['{tempo}','{tentativo}','{impatto}','{ctx}'])
        if not _HAS_VAR and np.random.rand() < 0.80:
            body = _join_fragment(body, np.random.choice(_ANTI_DUP_SUFFIXES))

        rows.append([i + 1, title, body, cat, pri])

    df = pd.DataFrame(rows, columns=["id", "title", "body", "category", "priority"])
    df["text"] = (df["title"].fillna("") + " " + df["body"].fillna("")).apply(clean_text)
    return df


# =============================================================================
# CACHE DATASET
# =============================================================================

@st.cache_data
def get_dataset(n=1000, seed=42) -> pd.DataFrame:
 
    df = generate_ticket_dataset(n_tickets=n, random_state=seed)
    return df


# =============================================================================
# PERSISTENZA MODELLI ML
# =============================================================================

def _model_paths(base_dir="models"):

    os.makedirs(base_dir, exist_ok=True)
    return {
        "vec_cat": os.path.join(base_dir, "vectorizer_cat.joblib"),
        "vec_pri": os.path.join(base_dir, "vectorizer_pri.joblib"),
        "model_cat": os.path.join(base_dir, "model_cat.joblib"),
        "model_pri": os.path.join(base_dir, "model_pri.joblib"),
        "meta": os.path.join(base_dir, "metadata.json"),
    }


def _save_models(vectorizer_cat, model_cat, vectorizer_pri, model_pri, 
                 metadata, base_dir="models"):
   
    paths = _model_paths(base_dir)
    
    # Salva modelli con compressione
    joblib.dump(vectorizer_cat, paths["vec_cat"], compress=3)
    joblib.dump(model_cat, paths["model_cat"], compress=3)
    joblib.dump(vectorizer_pri, paths["vec_pri"], compress=3)
    joblib.dump(model_pri, paths["model_pri"], compress=3)
    
    # Salva metadata come JSON
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def _load_models_if_compatible(expected_meta, base_dir="models"):
   
    paths = _model_paths(base_dir)
    
    # Verifica esistenza tutti i file necessari
    required = [
        paths["vec_cat"], paths["model_cat"], 
        paths["vec_pri"], paths["model_pri"], 
        paths["meta"]
    ]
    if not all(os.path.exists(p) for p in required):
        return None
    
    # Carica metadata salvati
    try:
        with open(paths["meta"], "r", encoding="utf-8") as f:
            saved_meta = json.load(f)
    except Exception:
        return None  # Metadata corrotti
    
    # Verifica compatibilità configurazione
    keys_to_check = ["vec_kwargs", "sklearn_version_major", "python_version_major"]
    for k in keys_to_check:
        if saved_meta.get(k) != expected_meta.get(k):
            return None  # Configurazione incompatibile
    
    # Carica modelli serializzati
    try:
        vectorizer_cat = joblib.load(paths["vec_cat"])  
        model_cat = joblib.load(paths["model_cat"])    
        vectorizer_pri = joblib.load(paths["vec_pri"])  
        model_pri = joblib.load(paths["model_pri"])    
        return vectorizer_cat, model_cat, vectorizer_pri, model_pri
    except Exception:
        return None  # Errore caricamento (file corrotti, versioni incompatibili)


def _safe_train_test_split(X, y, test_size=0.2, random_state=42, try_stratify=True):

    if try_stratify:
        vc = y.value_counts()
        # Stratify
        if (vc.min() >= 2) and (len(vc) >= 2):
            return train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state, 
                stratify=y
            )
    
    # Fallback
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# =============================================================================
# CARICAMENTO/TRAINING MODELLI ML
# =============================================================================

@st.cache_resource
def load_or_train_models(df, base_dir="models"):

    import sklearn, sys
    
    # Prepara metadata configurazione corrente
    expected_meta = {
        "created_at": datetime.now(UTC).isoformat(),
        "vec_kwargs": VEC_KWARGS,
        "model_cat": "LinearSVC",
        "model_pri": "LogisticRegression(class_weight=balanced, max_iter=500)",
        "sklearn_version": sklearn.__version__,
        "sklearn_version_major": sklearn.__version__.split(".")[0],
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_version_major": str(sys.version_info.major),
    }

    # Prepara dati
    X = df["text"]
    y_cat = df["category"]
    y_pri = df["priority"]
    
    # Tenta caricamento modelli salvati
    loaded = _load_models_if_compatible(expected_meta, base_dir=base_dir)
    
    if loaded is not None:
        # Modelli compatibili trovati → usa quelli
        vectorizer_cat, model_cat, vectorizer_pri, model_pri = loaded
    else:
        # Nessun modello o incompatibili → riallena da zero
        
        # Split train/test con stratificazione sicura
        Xc_tr, Xc_te, yc_tr, yc_te = _safe_train_test_split(
            X, y_cat, test_size=0.2, random_state=42, try_stratify=True
        )
        Xp_tr, Xp_te, yp_tr, yp_te = _safe_train_test_split(
            X, y_pri, test_size=0.2, random_state=42, try_stratify=True
        )
        
        # TRAINING CATEGORIA
        vectorizer_cat = TfidfVectorizer(**VEC_KWARGS)
        Xc_tr_vec = vectorizer_cat.fit_transform(Xc_tr)  # Fit su train, non test
        
        model_cat = LinearSVC()  
        model_cat.fit(Xc_tr_vec, yc_tr)
        
        # TRAINING PRIORITÀ
        vectorizer_pri = TfidfVectorizer(**VEC_KWARGS)
        Xp_tr_vec = vectorizer_pri.fit_transform(Xp_tr) # Fit su train, non test
        
        model_pri = LogisticRegression(
            max_iter=500,           # Aumentato per convergenza
            class_weight="balanced"  # Gestisce classi sbilanciate
        )
        model_pri.fit(Xp_tr_vec, yp_tr)
        
        # Salva modelli trainati per riuso futuro
        _save_models(vectorizer_cat, model_cat, vectorizer_pri, model_pri, 
                     expected_meta, base_dir=base_dir)

    # === VALUTAZIONE PERFORMANCE SU TEST SET ===
    # Ricrea split per valutazione (stesso random_state = stesso split)
    Xc_tr, Xc_te, yc_tr, yc_te = _safe_train_test_split(
        X, y_cat, test_size=0.2, random_state=42, try_stratify=True
    )
    Xp_tr, Xp_te, yp_tr, yp_te = _safe_train_test_split(
        X, y_pri, test_size=0.2, random_state=42, try_stratify=True
    )
    
    # Transform test set (solo transform, non fit_transform!)
    Xc_te_vec = vectorizer_cat.transform(Xc_te)
    Xp_te_vec = vectorizer_pri.transform(Xp_te)
    
    # Predizioni su test set
    ycp = model_cat.predict(Xc_te_vec)
    ypp = model_pri.predict(Xp_te_vec)

    # Prepara labels per confusion matrix (ordine consistente)
    labels_cat = model_cat.classes_  # Ordine automatico da modello
    
    # Priorità: ordine semantico bassa→media→alta
    order_map = {"bassa": 0, "media": 1, "alta": 2}
    labels_pri = np.array(sorted(
        list(set(yp_te)), 
        key=lambda x: order_map.get(x, 99)
    ))

    # Calcola metriche complete
    metrics = {
        "acc_cat": accuracy_score(yc_te, ycp),
        "f1_cat_macro": f1_score(yc_te, ycp, average="macro"),
        "acc_pri": accuracy_score(yp_te, ypp),
        "f1_pri_macro": f1_score(yp_te, ypp, average="macro"),
        "cm_cat": confusion_matrix(yc_te, ycp, labels=labels_cat),
        "cm_pri": confusion_matrix(yp_te, ypp, labels=labels_pri),
        "report_cat": classification_report(yc_te, ycp, output_dict=True),
        "report_pri": classification_report(yp_te, ypp, output_dict=True),
        "labels_cat": labels_cat,
        "labels_pri": labels_pri,
    }
    
    # =========================================================================
    # STAMPA METRICHE CONSOLE (per report e debugging)
    # =========================================================================
    print("\n" + "="*80)
    print("📊 PERFORMANCE MODELLI ML - METRICHE TEST SET")
    print("="*80)
    
    print("\n🏷️  CATEGORIA:")
    print(f"   • Accuracy:     {metrics['acc_cat']:.2%}  ({metrics['acc_cat']:.4f})")
    print(f"   • F1-Macro:     {metrics['f1_cat_macro']:.4f}")
    print(f"   • N° test set:  {len(yc_te)} ticket")
    
    print("\n   Dettaglio per classe:")
    for label in labels_cat:
        f1 = metrics['report_cat'][label]['f1-score']
        precision = metrics['report_cat'][label]['precision']
        recall = metrics['report_cat'][label]['recall']
        support = metrics['report_cat'][label]['support']
        print(f"      {label:15s} - F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Support: {support}")
    
    print("\n🎚️  PRIORITÀ:")
    print(f"   • Accuracy:     {metrics['acc_pri']:.2%}  ({metrics['acc_pri']:.4f})")
    print(f"   • F1-Macro:     {metrics['f1_pri_macro']:.4f}")
    print(f"   • N° test set:  {len(yp_te)} ticket")
    
    print("\n   Dettaglio per classe:")
    for label in labels_pri:
        f1 = metrics['report_pri'][label]['f1-score']
        precision = metrics['report_pri'][label]['precision']
        recall = metrics['report_pri'][label]['recall']
        support = metrics['report_pri'][label]['support']
        print(f"      {label:15s} - F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Support: {support}")
    
    print("\n" + "="*80)
    print("✅ Metriche calcolate con successo!")
    print("="*80 + "\n")
    
    return vectorizer_cat, model_cat, vectorizer_pri, model_pri, metrics


# =============================================================================
# SPIEGABILITÀ MODELLI (EXPLAINABILITY)
# =============================================================================

def get_top_words_for_category_hybrid(
    text: str, 
    vectorizer_cat: TfidfVectorizer,
    model_cat: LinearSVC, 
    predicted_category: str,
    top_n: int = 5
) -> list[str]:
    
    # Dizionari keywords per categoria
    category_rules_map = {
    "Tecnico": (
        CATEGORY_RULES_TECNICO_STRONG |
        CATEGORY_RULES_TECNICO |
        CATEGORY_RULES_INSTALL_SOFTWARE
    ),
    "Amministrazione": CATEGORY_RULES_AMMINISTRAZIONE,
    "Commerciale": CATEGORY_RULES_COMMERCIALE
    }
    
    high_priority_keywords = set(priorities_keywords.get("alta", []))
    
    # Estrazione features ML
    cleaned_text = clean_text(text)
    vec = vectorizer_cat.transform([cleaned_text]).tocsr()
    
    feature_names = np.array(vectorizer_cat.get_feature_names_out())
    
    pred_class = predicted_category
    try:
        class_index = list(model_cat.classes_).index(pred_class)
    except ValueError:
        return []
    
    nonzero_indices = vec.indices
    if nonzero_indices.size == 0:
        return []
    
    tfidf_weights = vec.data
    coef_for_class = model_cat.coef_[class_index]
    ml_contributions = tfidf_weights * coef_for_class[nonzero_indices]
    
    # Match con regole
    relevant_rules = category_rules_map.get(pred_class, set())
    
    matched_rules = []
    for keyword in relevant_rules:
        if keyword in cleaned_text:
            matched_rules.append(keyword)
    
    matched_high_priority = []
    for keyword in high_priority_keywords:
        if keyword in cleaned_text:
            matched_high_priority.append(keyword)
    
    # Scoring ibrido
    feature_scores = {}
    
    if ml_contributions.max() > 0:
        ml_scores_normalized = ml_contributions / ml_contributions.max()
    else:
        ml_scores_normalized = ml_contributions
    
    for idx, feat_idx in enumerate(nonzero_indices):
        feature = feature_names[feat_idx]
        ml_score = ml_scores_normalized[idx]
        
        total_score = float(ml_score)
        
        # Boost regole categoria
        if feature in relevant_rules:
            total_score += 5.0
        
        # Boost priorità alta
        if feature in high_priority_keywords:
            total_score += 3.0
        
        # Boost match parziali
        for rule_kw in relevant_rules:
            if feature in rule_kw or rule_kw in feature:
                total_score += 1.0
                break
        
        for hp_kw in high_priority_keywords:
            if feature in hp_kw or hp_kw in feature:
                total_score += 0.5
                break
        
        feature_scores[feature] = total_score
    
    # Aggiungi keywords regole matchate
    for matched_kw in matched_rules:
        if matched_kw not in feature_scores:
            feature_scores[matched_kw] = 10.0
    
    for matched_hp in matched_high_priority:
        if matched_hp not in feature_scores:
            feature_scores[matched_hp] = 8.0
    
    # Ordina e restituisci top N
    sorted_features = sorted(
        feature_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    top_features = [feat for feat, score in sorted_features[:top_n]]
    
    return top_features


def get_top_words_for_category(
    text: str, 
    vectorizer_cat: TfidfVectorizer,
    model_cat: LinearSVC, 
    top_n: int = 5
) -> list[str]:
  
    cleaned_text = clean_text(text)
    vec = vectorizer_cat.transform([cleaned_text])
    pred_category = model_cat.predict(vec)[0]
    
    return get_top_words_for_category_hybrid(
        text, vectorizer_cat, model_cat, pred_category, top_n
    )


# =============================================================================
# INIZIALIZZAZIONE APPLICAZIONE
# =============================================================================

# Carica/genera dataset sintetico (1000 ticket)
# @st.cache_data evita rigenerazioni ad ogni reload
df = get_dataset()

# Carica o allena modelli ML
# @st.cache_resource cachea i modelli trainati
vectorizer_cat, model_cat, vectorizer_pri, model_pri, M = load_or_train_models(df)

# Prepara distribuzioni dataset per visualizzazioni
# Conversione esplicita a tipi Python puri (evita problemi JSON/serializzazione)
dist_cat = {str(k): int(v) for k, v in df['category'].value_counts().items()}
dist_pri = {str(k): int(v) for k, v in df['priority'].value_counts().items()}


# =============================================================================
# INTERFACCIA UTENTE - TABS
# =============================================================================

# Crea 4 tabs principali dell'applicazione
tab_pred, tab_saved, tab_batch, tab_metrics = st.tabs([
    "🎫 Predizione Ticket",      # Tab 1: Predizione singola
    "📋 Ticket Salvati",         # Tab 2: Storico ticket
    "📊 Batch & Dataset",        # Tab 3: Upload CSV e dataset sintetico
    "📈 Analisi modello"         # Tab 4: Performance ML
])


# =============================================================================
# TAB 1: PREDIZIONE TICKET SINGOLA
# =============================================================================
with tab_pred:
    st.title(MSG_PRED_TITLE)
    st.caption(MSG_PRED_CAPTION)

    # Inizializza session_state per persistenza dati form
    # Evita perdita dati durante reload/submit form
    for key in ["title_input", "body_input", "result", "top_words", 
                "ticket_saved", "form_key"]:
        if key not in st.session_state:
            # Default values per tipo
            if key in ["title_input", "body_input"]:
                st.session_state[key] = ""
            elif key == "ticket_saved":
                st.session_state[key] = False
            elif key == "form_key":
                st.session_state[key] = 0  # Counter per reset form
            else:
                st.session_state[key] = None

    # Form predizione (clear_on_submit=False mantiene dati dopo submit)
    with st.form(f"ticket_form_{st.session_state.form_key}", clear_on_submit=False):
        # Input titolo e descrizione ticket
        title_input = st.text_input(
            "Titolo ticket", 
            value=st.session_state.title_input, 
            placeholder=MSG_PLACEHOLDER_TITLE
        )
        body_input = st.text_area(
            "Descrizione ticket", 
            value=st.session_state.body_input, 
            placeholder=MSG_PLACEHOLDER_BODY, 
            height=160
        )
        
        # Pulsante submit centrato in colonna centrale
        col_spacer1, col_submit, col_spacer2 = st.columns([1, 2, 1])
        with col_submit:
            submitted = st.form_submit_button(
                "🔍 Analizza Ticket", 
                type="primary", 
                use_container_width=True
            )

    # GESTIONE SUBMIT FORM
    if submitted:
        # Salva input in session_state per persistenza
        st.session_state.title_input = title_input
        st.session_state.body_input = body_input
        
        # Pulisce e combina testo
        full_text = clean_text((title_input or "") + " " + (body_input or ""))
        
        # Validazione input
        if not full_text.strip():
            st.warning(MSG_INSERT_FIELDS)
        else:
            # Mostra spinner durante predizione
            with st.spinner(MSG_ANALYSIS_SPINNER):
                # PREDIZIONE CATEGORIA
                X_input_cat = vectorizer_cat.transform([full_text])
                pred_cat = model_cat.predict(X_input_cat)[0]
                
                # PREDIZIONE PRIORITÀ
                X_input_pri = vectorizer_pri.transform([full_text])
                pred_pri = model_pri.predict(X_input_pri)[0]
                
                # Traccia regole applicate (per debugging/trasparenza)
                applied_rules = []

                # APPLICAZIONE REGOLE PRIORITÀ
                # Regole possono sovrascrivere predizione ML
                rule_pri = rule_based_priority(full_text)
                if rule_pri is not None and rule_pri != pred_pri:
                    pred_pri = rule_pri
                    applied_rules.append(f"Priorità forzata da regola: {rule_pri}")

                # APPLICAZIONE REGOLE CATEGORIA
                rule_cat = rule_based_category(full_text)
                if rule_cat is not None and rule_cat != pred_cat:
                    pred_cat = rule_cat
                    applied_rules.append(f"Categoria forzata da regola: {rule_cat}")

                # ESTRAZIONE PAROLE CHIAVE (EXPLAINABILITY)
                top_words = get_top_words_for_category_hybrid(
                    full_text, 
                    vectorizer_cat, 
                    model_cat, 
                    pred_cat,  # USA categoria FINALE (dopo regole applicate)
                    top_n=5
                )
                
                # Salva risultati in session_state
                st.session_state.result = (pred_cat, pred_pri)
                st.session_state.top_words = top_words
                st.session_state.ticket_saved = True

                # PERSISTENZA PREDIZIONE IN CSV
                try:
                    df_pred = pd.read_csv(PRED_FILE)
                except Exception:
                    # File corrotto o mancante - ricrea
                    df_pred = pd.DataFrame(columns=[
                        "id", "date", "title", "body", "category", "priority"
                    ])
                
                # Genera nuovo ID (max + 1)
                new_id = 1 if df_pred.empty else int(df_pred["id"].max()) + 1
                now_str = datetime.now().strftime("%Y-%m-%d")
                
                # Crea nuova riga
                new_row = pd.DataFrame([{
                    "id": new_id, 
                    "date": now_str, 
                    "title": title_input, 
                    "body": body_input,
                    "category": pred_cat, 
                    "priority": pred_pri
                }])
                
                # Append e salva (encoding UTF-8 con BOM per Excel)
                df_pred = pd.concat([df_pred, new_row], ignore_index=True)
                df_pred.to_csv(PRED_FILE, index=False, encoding="utf-8-sig")

                # Mostra regole applicate (se presenti)
                if applied_rules:
                    with st.expander("ℹ️ Regole applicate"):
                        for r in applied_rules:
                            st.markdown(f"- {r}")

    # VISUALIZZAZIONE RISULTATI
    if st.session_state.result:
        # Spacing section risultati
        st.markdown('<div class="results-section"></div>', unsafe_allow_html=True)
        
        categoria, priorita = st.session_state.result
        
        # Header sezione
        st.markdown("### 📊 Risultati dell'Analisi")
        
        # Risultati in 2 colonne (categoria | priorità)
        col_res1, col_res2 = st.columns(2)
        
        # CARD CATEGORIA
        with col_res1:
            st.markdown("#### Categoria")
            st.markdown(
                f'<div style="padding: 1rem; background: #f8f9fa; '
                f'border-radius: 8px; border-left: 4px solid #4a90e2;">'
                f'<span class="badge badge-cat" style="font-size: 1.1rem;">{categoria}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        # CARD PRIORITÀ
        with col_res2:
            st.markdown("#### Priorità")
            # Colori semantici per priorità
            pri_class = (
                "badge-pri-alta" if priorita == "alta" else 
                ("badge-pri-media" if priorita == "media" else "badge-pri-bassa")
            )
            border_color = (
                "#dc3545" if priorita == "alta" else 
                ("#ffc107" if priorita == "media" else "#28a745")
            )
            st.markdown(
                f'<div style="padding: 1rem; background: #f8f9fa; '
                f'border-radius: 8px; border-left: 4px solid {border_color};">'
                f'<span class="badge {pri_class}" style="font-size: 1.1rem;">'
                f'{priorita.upper() if priorita=="alta" else priorita.capitalize()}'
                f'</span></div>',
                unsafe_allow_html=True
            )
        
        # PAROLE CHIAVE (EXPLAINABILITY)
        if st.session_state.top_words:
            st.markdown('<div class="keywords-section"></div>', unsafe_allow_html=True)
            st.markdown("#### 🔑 Parole Chiave Identificate")
            
            # Visualizza le parole chiave come etichette per facilitarne la lettura
            chips = " ".join([
                f"<span class='codechip' style='margin: 0 4px;'>{w}</span>" 
                for w in st.session_state.top_words
            ])
            st.markdown(f"<div style='line-height: 2.5;'>{chips}</div>", 
                       unsafe_allow_html=True)
        
        # Conferma salvataggio
        if st.session_state.ticket_saved:
            st.success("✅ Ticket salvato correttamente nel sistema")
        
        # PULSANTE "+ NUOVO TICKET"
        st.markdown('<div class="new-ticket-button"></div>', unsafe_allow_html=True)
        
        # Centrato in colonna centrale
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            # CSS custom per pulsante verde gradient
            new_ticket_html = """
            <style>
            .stButton > button {
                width: 100%;
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
                border: none !important;
                color: white !important;
                font-weight: 600 !important;
                padding: 0.75rem 1.5rem !important;
                border-radius: 8px !important;
                box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3) !important;
                transition: all 0.3s ease !important;
                font-size: 1rem !important;
            }
            .stButton > button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 16px rgba(40, 167, 69, 0.4) !important;
                background: linear-gradient(135deg, #20c997 0%, #17a2b8 100%) !important;
            }
            .stButton > button:before {
                content: "+ ";
                font-size: 1.2rem;
                font-weight: 700;
            }
            </style>
            """
            st.markdown(new_ticket_html, unsafe_allow_html=True)
            
            # Handler click: reset form e session_state
            if st.button("Nuovo Ticket", key="new_ticket", use_container_width=True):
                st.session_state.title_input = ""
                st.session_state.body_input = ""
                st.session_state.result = None
                st.session_state.top_words = []
                st.session_state.ticket_saved = False
                st.session_state.form_key += 1  # Incrementa per nuovo form ID
                st.rerun()  # Reload pagina per applicare reset


# =============================================================================
# TAB 2: TICKET SALVATI (STORICO)
# =============================================================================
with tab_saved:
    st.title(MSG_SAVED_TITLE)
    
    # Verifica esistenza file predizioni
    if os.path.exists(PRED_FILE):
        # Carica CSV con gestione errori
        try:
            df_pred = pd.read_csv(PRED_FILE)
        except Exception:
            # File corrotto → DataFrame vuoto
            df_pred = pd.DataFrame(columns=[
                "id", "date", "title", "body", "category", "priority"
            ])
        
        # Aggiunge colonna 'date' se mancante (backward compatibility)
        if 'date' not in df_pred.columns:
            df_pred.insert(1, 'date', '')
        
        # Formatta date in formato YYYY-MM-DD
        dt = pd.to_datetime(df_pred['date'], errors='coerce')
        df_pred['date'] = dt.dt.strftime('%Y-%m-%d').fillna('')

        # Rinomina colonne per visualizzazione italiana
        df_view = df_pred.rename(columns={
            'id': 'ID', 
            'date': 'Data', 
            'title': 'Titolo Ticket', 
            'body': 'Descrizione', 
            'category': 'Categoria', 
            'priority': 'Priorità'
        })
        
        # Ordina per data decrescente (ticket più recenti prima)
        dt2 = pd.to_datetime(df_view['Data'], errors='coerce')
        df_view = (
            df_view.assign(_DataSort=dt2)
            .sort_values('_DataSort', ascending=False)
            .drop(columns=['_DataSort'])
        )

        # FILTRI RAPIDI
        st.subheader(MSG_FILTERS_SUBTITLE)
        col_f0, col_f1, col_f2, col_f3 = st.columns([2, 1, 1, 1])
        
        # Filtro ricerca testuale
        with col_f0:
            search = st.text_input(
                "Cerca", 
                placeholder="Cerca in titolo/descrizione…"
            )
        
        # Filtro data
        with col_f1:
            dates = sorted([
                d for d in df_view['Data'].unique().tolist() 
                if isinstance(d, str) and d
            ], reverse=True)
            sel_date = st.selectbox(
                MSG_FILTER_DATE, 
                options=([MSG_ALL_DATES] + dates), 
                index=0
            )
        
        # Filtro categorie
        with col_f2:
            cats = sorted([
                c for c in df_view['Categoria'].dropna().unique().tolist()
            ])
            sel_cats = st.multiselect(MSG_FILTER_CATS, options=cats, default=[])
        
        # Filtro priorità
        with col_f3:
            sel_pri = st.multiselect(MSG_FILTER_PRI, options=PRI_ORDER, default=[])

        # APPLICAZIONE FILTRI
        mask = pd.Series(True, index=df_view.index)  # Start: tutti True
        
        # Filtro data
        if sel_date != MSG_ALL_DATES:
            mask &= (df_view['Data'] == sel_date)
        
        # Filtro categorie
        if sel_cats:
            mask &= df_view['Categoria'].isin(sel_cats)
        
        # Filtro priorità
        if sel_pri:
            mask &= df_view['Priorità'].isin(sel_pri)
        
        # Filtro ricerca testuale (case-insensitive, titolo / descrizione)
        if search:
            mask &= (
                df_view['Titolo Ticket'].fillna('').str.contains(search, case=False) | 
                df_view['Descrizione'].fillna('').str.contains(search, case=False)
            )

        # Applica filtri
        df_filtered = df_view[mask]
        
        # Seleziona colonne per visualizzazione
        view_cols = ["ID", "Data", "Titolo Ticket", "Descrizione", "Categoria", "Priorità"]
        df_tbl = df_filtered[view_cols]

        # STYLING DATAFRAME
        def _style_priorita(col: pd.Series) -> list[str]:    
            palette = {
                "bassa": "#1b5e20",   # Verde scuro
                "media": "#E08E0B",   # Arancione
                "alta": "#a61e1e"     # Rosso scuro
            }
            styles = []
            for v in col.astype(str).str.lower():
                color = palette.get(v, "#333333")  # Default grigio
                styles.append(f"color: {color}; font-weight: 600;")
            return styles
        
        def _row_style(row):
            bg = (
                "background-color: #FFF0F0" 
                if str(row['Priorità']).lower() == 'alta' 
                else ""
            )
            return [bg] * len(row)

        # Applica styling
        styled = (
            df_tbl.style
            .apply(_style_priorita, subset=["Priorità"])  # Colori priorità
            .apply(_row_style, axis=1)                    # Sfondo righe ALTA
        )
        
        # Mostra tabella
        st.dataframe(styled, hide_index=True, use_container_width=True)

        # DOWNLOAD CSV
        with open(PRED_FILE, "rb") as f:
            st.download_button(
                label=MSG_TICKET_SAVED_CSV_BTN, 
                data=f, 
                file_name="predizione.csv", 
                mime="text/csv"
            )
    else:
        # Nessun ticket salvato
        st.info(MSG_NO_TICKETS)


# =============================================================================
# TAB 3: BATCH PREDICTION & DATASET
# =============================================================================
with tab_batch:
    st.title("📊 Batch Prediction & Dataset")
    
    # =========================================================================
    # SEZIONE 1: DOWNLOAD DATASET SINTETICO
    # =========================================================================
    st.markdown("### 📥 Dataset Sintetico per Training")
    
    # Card informativa con icon e descrizione
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
                padding: 1.5rem; border-radius: 12px; border-left: 4px solid #667eea; 
                margin-bottom: 1.5rem;'>
        <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
            <span style='font-size: 1.5rem; margin-right: 0.5rem;'>🎲</span>
            <span style='font-weight: 600; font-size: 1.1rem;'>Dataset di 1000 ticket sintetici</span>
        </div>
        <p style='margin: 0.5rem 0 0 2.5rem; color: #555;'>
            Ticket realistici generati automaticamente per testare il sistema o addestrare modelli personalizzati.
            Include colonne: <code>id</code>, <code>title</code>, <code>body</code>, <code>category</code>, <code>priority</code>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Prepara CSV dataset sintetico
    csv_data = df[["id", "title", "body", "category", "priority"]].to_csv(
        index=False
    ).encode("utf-8-sig")
    
    # Pulsante download 
    col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
    with col_dl2:
        st.download_button(
            label="📥 Scarica Dataset Sintetico (CSV)",
            data=csv_data,
            file_name="ticket_sintetici.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download di 1000 ticket di esempio con categoria e priorità già classificate"
        )

    # Spacing tra sezioni
    st.markdown("<div style='margin: 2.5rem 0;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # SEZIONE 2: DOWNLOAD PREDIZIONI SALVATE
    # =========================================================================
    st.markdown("### 💾 Esporta Predizioni Salvate")
    
    if os.path.exists(PRED_FILE):
        # Leggi file per mostrare statistiche
        try:
            df_pred_stats = pd.read_csv(PRED_FILE)
            n_saved = len(df_pred_stats)
            
            # Card con statistiche
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Ticket Salvati", f"{n_saved:,}")
            with col_stat2:
                if 'category' in df_pred_stats.columns:
                    most_common_cat = df_pred_stats['category'].mode()[0] if not df_pred_stats['category'].mode().empty else "N/A"
                    st.metric("Categoria Prevalente", most_common_cat)
            with col_stat3:
                if 'priority' in df_pred_stats.columns:
                    n_alta = (df_pred_stats['priority'] == 'alta').sum()
                    st.metric("Priorità Alta", f"{n_alta:,}", 
                             delta=f"{(n_alta/n_saved*100):.1f}%" if n_saved > 0 else None)
        except Exception:
            pass
        
        st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
        
        # Pulsante download
        with open(PRED_FILE, "rb") as f:
            col_dl4, col_dl5, col_dl6 = st.columns([1, 2, 1])
            with col_dl5:
                st.download_button(
                    label="💾 Scarica Storico Predizioni (CSV)",
                    data=f,
                    file_name="predizioni_storico.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="Esporta tutti i ticket analizzati con le relative predizioni"
                )
    else:
        # Nessun ticket salvato
        st.info("📭 Nessuna predizione salvata. Inizia ad analizzare ticket nella sezione **Predizione Ticket**!")

    # Spacing tra sezioni
    st.markdown("<div style='margin: 2.5rem 0;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # SEZIONE 3: PREDIZIONE BATCH DA CSV
    # =========================================================================
    st.markdown("### 🚀 Predizione Batch da File CSV")
    
    # Card informativa upload
    st.markdown("""
    <div style='background: linear-gradient(135deg, #20c99715 0%, #17a2b815 100%); 
                padding: 1.5rem; border-radius: 12px; border-left: 4px solid #20c997; 
                margin-bottom: 1.5rem;'>
        <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
            <span style='font-size: 1.5rem; margin-right: 0.5rem;'>📤</span>
            <span style='font-weight: 600; font-size: 1.1rem;'>Carica un file CSV per l'analisi rapida dei ticket</span>
        </div>
        <div style='margin: 0.5rem 0 0 2.5rem;'>
            <p style='margin: 0.3rem 0; color: #555;'><strong>Formato richiesto:</strong></p>
            <ul style='margin: 0.3rem 0; color: #555;'>
                <li>Colonne obbligatorie: <code>title</code>/<code>titolo</code> e <code>body</code>/<code>descrizione</code></li>
                <li>Encoding consigliato: <strong>UTF-8</strong> o <strong>UTF-8 con BOM</strong></li>
                <li>Separatore: virgola, punto e virgola o tab (rilevamento automatico)</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader con styling
    uploaded_file = st.file_uploader(
        "Seleziona file CSV",
        type=["csv"],
        help="Trascina qui il file oppure clicca per selezionarlo",
        label_visibility="collapsed"
    )

    # Mappatura nomi colonne (flessibilità input utente)
    COLUMN_ALIASES = {
        "title": ["title", "titolo", "subject"],
        "body": ["body", "descrizione", "testo"]
    }

    def _resolve_columns(cols):
        """
        Risolve nomi colonne usando aliases (case-insensitive).
        
        Args:
            cols: Lista nomi colonne dal CSV
            
        Returns:
            Dict {standard_name: actual_name}
            
        Esempio:
            >>> _resolve_columns(["Titolo", "Descrizione"])
            {'title': 'Titolo', 'body': 'Descrizione'}
        """
        cols_lower = {c.lower(): c for c in cols}
        resolved = {}
        for std, aliases in COLUMN_ALIASES.items():
            for a in aliases:
                if a in cols_lower:
                    resolved[std] = cols_lower[a]
                    break
        return resolved

    # GESTIONE FILE UPLOAD
    if uploaded_file is not None:
        def _try_read_csv(raw_bytes, enc):
      
            bio = io.BytesIO(raw_bytes)
            return pd.read_csv(
                bio, 
                sep=None,              # Auto-detect separatore
                engine='python',       # Engine flessibile
                quotechar='"',         
                skip_blank_lines=True, 
                on_bad_lines='warn',   # Logga righe malformate
                encoding=enc
            )

        # Leggi contenuto file
        raw = uploaded_file.read()
        df_in = None
        used_encoding = None
        errors = []
        
        # Prova encodings comuni in ordine di probabilità
        with st.spinner("🔄 Lettura file in corso..."):
            for enc in ('utf-8-sig', 'utf-8', 'cp1252', 'latin-1', 'iso-8859-1'):
                try:
                    df_in = _try_read_csv(raw, enc)
                    used_encoding = enc
                    break  # Successo → stop
                except UnicodeDecodeError as e:
                    errors.append(f" {enc}: Encoding non compatibile")
                except Exception as e:
                    errors.append(f" {enc}: {str(e)[:100]}")

        # GESTIONE ERRORI LETTURA
        if df_in is None:
            # Nessun encoding funzionante → mostra errori dettagliati
            st.error("❌ **Impossibile leggere il file CSV**")
            
            with st.expander("🔍 Dettagli errore"):
                st.markdown("**Tentativi effettuati:**")
                for err in errors:
                    st.markdown(f"- {err}")
                
                st.markdown("---")
                st.markdown("**💡 Suggerimenti per risolvere:**")
                st.markdown("""
                1. **Se usi Excel**:
                   - Vai su `File` → `Salva con nome`
                   - Seleziona formato `CSV UTF-8 (delimitato da virgole) (*.csv)`
                   
                2. **Se usi Google Sheets**:
                   - Vai su `File` → `Scarica` → `Valori separati da virgola (.csv)`
                   
                3. **Controlla il contenuto**:
                   - Apri il CSV con un editor di testo (es: Notepad++)
                   - Verifica che non ci siano caratteri strani o formattazione insolita
                   
                4. **Serve aiuto?**
                   - Contatta il supporto e allega il file problematico
                """)
        else:
            # LETTURA CSV RIUSCITA
            st.success(f"✅ **File caricato correttamente!** (Encoding: `{used_encoding}`)")
            
            # Mostra info colonne rilevate
            st.info(f"📋 **Colonne rilevate:** {', '.join([f'`{c}`' for c in df_in.columns])}")
            
            # Risolvi nomi colonne
            mapping = _resolve_columns(df_in.columns)
            
            # Validazione colonne obbligatorie
            if not {"title", "body"}.issubset(mapping.keys()):
                st.error("❌ **Colonne obbligatorie mancanti**")
                st.markdown("""
                Il CSV deve contenere **almeno** le seguenti colonne:
                - `title` / `titolo` / `subject`
                - `body` / `descrizione` / `testo`
                
                **Colonne trovate nel tuo file:**
                """)
                for col in df_in.columns:
                    st.markdown(f"- `{col}`")
            else:
                # PREDIZIONE BATCH 
                with st.spinner("Analisi in corso... Questo potrebbe richiedere qualche secondo."):
                    # Prepara colonna 'text' combinata
                    df_in["text"] = (
                        df_in[mapping["title"]].fillna("") + " " + 
                        df_in[mapping["body"]].fillna("")
                    ).apply(clean_text)
                    
                    # Vectorizza batch
                    X_cat_batch = vectorizer_cat.transform(df_in["text"])                
                    X_pri_batch = vectorizer_pri.transform(df_in["text"])
                    
                    # Predizioni ML
                    df_in["predicted_category"] = model_cat.predict(X_cat_batch)
                    df_in["predicted_priority"] = model_pri.predict(X_pri_batch)
                    
                    # APPLICAZIONE REGOLE POST-PREDIZIONE
                    # Applica regole priorità (sovrascrive ML se necessario)
                    df_in["predicted_priority"] = [
                        rule_based_priority(t) or p 
                        for t, p in zip(df_in["text"], df_in["predicted_priority"])
                    ]
                    
                    # Applica regole categoria (sovrascrive ML se necessario)
                    df_in["predicted_category"] = [
                        rule_based_category(t) or c 
                        for t, c in zip(df_in["text"], df_in["predicted_category"])
                    ]

                    # PREPARAZIONE OUTPUT
                    df_in.insert(0, "id", range(1, len(df_in) + 1))
                
                    output_cols = [
                        "id", 
                        mapping["title"], 
                        mapping["body"], 
                        "predicted_category", 
                        "predicted_priority"
                    ]
                    df_out = df_in[output_cols].rename(columns={
                        mapping["title"]: "title", 
                        mapping["body"]: "body",
                        "predicted_category": "category",
                        "predicted_priority": "priority"
                    })

                # Success message con statistiche
                st.success(f"✅ **Analisi completata!** {len(df_out):,} ticket classificati")
                
                # Statistiche predizioni
                col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                with col_stats1:
                    st.metric("Totale Ticket", f"{len(df_out):,}")
                with col_stats2:
                    n_tec = (df_out['category'] == 'Tecnico').sum()
                    st.metric("Tecnico", f"{n_tec:,}", delta=f"{n_tec/len(df_out)*100:.0f}%")
                with col_stats3:
                    n_amm = (df_out['category'] == 'Amministrazione').sum()
                    st.metric("Amministrazione", f"{n_amm:,}", delta=f"{n_amm/len(df_out)*100:.0f}%")
                with col_stats4:
                    n_comm = (df_out['category'] == 'Commerciale').sum()
                    st.metric("Commerciale", f"{n_comm:,}", delta=f"{n_comm/len(df_out)*100:.0f}%")
                
                st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
                
                # Priorità stats
                col_pri_stats1, col_pri_stats2, col_pri_stats3 = st.columns(3)
                with col_pri_stats1:
                    n_bassa = (df_out['priority'] == 'bassa').sum()
                    st.metric("🟢 Priorità Bassa", f"{n_bassa:,}", delta=f"{n_bassa/len(df_out)*100:.0f}%")
                with col_pri_stats2:
                    n_media = (df_out['priority'] == 'media').sum()
                    st.metric("🟡 Priorità Media", f"{n_media:,}", delta=f"{n_media/len(df_out)*100:.0f}%")
                with col_pri_stats3:
                    n_alta = (df_out['priority'] == 'alta').sum()
                    st.metric("🔴 Priorità Alta", f"{n_alta:,}", delta=f"{n_alta/len(df_out)*100:.0f}%")

                st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

                # VISUALIZZAZIONE RISULTATI
                st.markdown("#### 📋 Anteprima Risultati (Prime 50 righe)")
                
                # Mostra dataframe con styling
                st.dataframe(
                    df_out.head(50), 
                    hide_index=True, 
                    use_container_width=True,
                    height=400
                )
                
                # DOWNLOAD CSV PREDETTO
                st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
                
                csv_pred = df_out.to_csv(index=False).encode("utf-8-sig")
                
                col_dl7, col_dl8, col_dl9 = st.columns([1, 2, 1])
                with col_dl8:
                    st.download_button(
                        label=f"📥 Scarica Risultati Completi ({len(df_out):,} ticket)",
                        data=csv_pred,
                        file_name=f"ticket_classificati_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        help="Download CSV con tutte le predizioni (categoria e priorità)"
                    )


# =============================================================================
# TAB 4: ANALISI PERFORMANCE MODELLI
# =============================================================================
with tab_metrics:
    st.title("📊 Analisi delle Prestazioni dei Modelli")
    
    # =========================================================================
    # SEZIONE 1: DISTRIBUZIONE DATASET
    # =========================================================================
    st.markdown("---")
    st.subheader("📦 Distribuzione dei Dati")
    
    col_dist1, col_dist2, col_dist3 = st.columns(3)
    
    # Metrica 1: Totale ticket
    with col_dist1:
        st.metric(
            label="Dataset Totale",
            value=f"{len(df):,} ticket",
            delta=None
        )
    
    # Metrica 2: Distribuzione categorie
    with col_dist2:
        st.markdown("**Categorie**")
        for cat, count in sorted(dist_cat.items(), key=lambda x: -x[1]):
            pct = (count / len(df)) * 100
            st.markdown(f"- **{cat}**: {count} ({pct:.1f}%)")
    
    # Metrica 3: Distribuzione priorità
    with col_dist3:
        st.markdown("**Priorità**")
        pri_order_display = {
            'bassa': '🟢 Bassa', 
            'media': '🟡 Media', 
            'alta': '🔴 Alta'
        }
        for pri in PRI_ORDER:
            if pri in dist_pri:
                count = dist_pri[pri]
                pct = (count / len(df)) * 100
                st.markdown(f"- {pri_order_display[pri]}: {count} ({pct:.1f}%)")
    
    # =========================================================================
    # SEZIONE 2: METRICHE PERFORMANCE GLOBALI
    # =========================================================================
    st.markdown("---")
    st.subheader("🎯 Metriche di Performance")
    
    # Metrics con confronto baseline
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    # Metrica 1: Accuracy Categoria
    with col_m1:
        acc_cat_val = M.get('acc_cat', 0)
        st.metric(
            label="Accuracy Categoria",
            value=f"{acc_cat_val:.1%}",
            delta=f"{(acc_cat_val - 0.80):.1%} vs baseline" if acc_cat_val > 0 else None,
            delta_color="normal"
        )
    
    # Metrica 2: F1-score Macro Categoria
    with col_m2:
        f1_cat_val = M.get('f1_cat_macro', 0)
        st.metric(
            label="F1 Macro Categoria",
            value=f"{f1_cat_val:.2f}",
            help="Media F1-score per classe, utile con classi sbilanciate"
        )
    
    # Metrica 3: Accuracy Priorità
    with col_m3:
        acc_pri_val = M.get('acc_pri', 0)
        st.metric(
            label="Accuracy Priorità",
            value=f"{acc_pri_val:.1%}",
            delta=f"{(acc_pri_val - 0.75):.1%} vs baseline" if acc_pri_val > 0 else None,
            delta_color="normal"
        )
    
    # Metrica 4: F1-score Macro Priorità
    with col_m4:
        f1_pri_val = M.get('f1_pri_macro', 0)
        st.metric(
            label="F1 Macro Priorità",
            value=f"{f1_pri_val:.2f}",
            help="Capacità di distinguere correttamente le priorità"
        )
    
    # =========================================================================
    # SEZIONE 3: ANALISI DETTAGLIATA CATEGORIA
    # =========================================================================
    st.markdown("---")
    st.subheader("🏷️ Analisi Dettagliata: Categoria")
    
    col_cat1, col_cat2 = st.columns(2)
    
    # GRAFICO 1: F1-SCORE PER CLASSE (CATEGORIA)
    with col_cat1:
        st.markdown("**F1-score per Classe**")
        
        # Estrae F1-score per ogni classe dal classification report
        f1_per_class_cat = {
            label: metrics["f1-score"] 
            for label, metrics in M.get("report_cat", {}).items() 
            if label in ["Amministrazione", "Tecnico", "Commerciale"]
        }
        
        # Crea grafico a barre
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        if f1_per_class_cat:
            colors = ['#4a90e2', '#2ecc71', '#f39c12']  
            bars1 = ax1.bar(
                list(f1_per_class_cat.keys()), 
                list(f1_per_class_cat.values()), 
                color=colors,
                edgecolor='white',
                linewidth=1.5
            )
            
            # Etichette valori sopra barre
            ax1.bar_label(bars1, fmt="%.2f", padding=3, fontweight='bold')
            
            # Configurazione assi
            ax1.set_ylim(0, 1.1)
            ax1.set_ylabel("F1-score", fontweight='bold')
            ax1.set_title("Performance per Categoria", fontweight='bold', pad=15)
            
            # Linea target 0.80
            ax1.axhline(
                y=0.8, color='red', linestyle='--', 
                linewidth=1, alpha=0.5, label='Target 0.80'
            )
            ax1.legend(loc='lower right')
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        st.pyplot(fig1)
        plt.close(fig1)
    
    # GRAFICO 2: CONFUSION MATRIX (CATEGORIA)
    with col_cat2:
        st.markdown("**Matrice di Confusione**")
        
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        cm_cat = M.get("cm_cat")
        labels_cat = M.get("labels_cat")
        
        if cm_cat is not None and labels_cat is not None:
            sns.heatmap(
                cm_cat, 
                annot=True,           # Mostra valori numerici
                fmt="d",              # Formato intero
                cmap="Blues",         
                ax=ax2,
                xticklabels=labels_cat, 
                yticklabels=labels_cat,
                cbar_kws={'label': 'Numero predizioni'},
                linewidths=0.5,       
                linecolor='gray'
            )
            
            # Etichette assi
            ax2.set_xlabel("Predetto", fontweight='bold')
            ax2.set_ylabel("Reale", fontweight='bold')
            ax2.set_title("Confusion Matrix - Categoria", fontweight='bold', pad=15)
        
        st.pyplot(fig2)
        plt.close(fig2)
    
    # INSIGHTS AUTOMATICI CATEGORIA
    with st.expander("💡 Insights Categoria"):
        if f1_per_class_cat:
            # Identifica classe migliore/peggiore
            worst_class = min(f1_per_class_cat, key=f1_per_class_cat.get)
            best_class = max(f1_per_class_cat, key=f1_per_class_cat.get)
            
            # Calcola variabilità (range F1-scores)
            variability = max(f1_per_class_cat.values()) - min(f1_per_class_cat.values())
            
            st.markdown(f"""
            - **Classe migliore**: {best_class} (F1: {f1_per_class_cat[best_class]:.2f})
            - **Classe da migliorare**: {worst_class} (F1: {f1_per_class_cat[worst_class]:.2f})
            - **Variabilità**: {variability:.2f} (ideale <0.10)
            
            **Interpretazione**:
            - F1 ≥ 0.80: Performance eccellente
            - F1 0.70-0.80: Performance buona
            - F1 < 0.70: Necessita miglioramenti
            """)
    
    # =========================================================================
    # SEZIONE 4: ANALISI DETTAGLIATA PRIORITÀ
    # =========================================================================
    st.markdown("---")
    st.subheader("🎚️ Analisi Dettagliata: Priorità")
    
    col_pri1, col_pri2 = st.columns(2)
    
    # GRAFICO 3: F1-SCORE PER CLASSE (PRIORITÀ)
    with col_pri1:
        st.markdown("**F1-score per Classe**")
        
        # Estrae F1-score per ogni priorità
        f1_per_class_pri = {
            label: metrics["f1-score"] 
            for label, metrics in M.get("report_pri", {}).items() 
            if label in ['bassa','media','alta']
        }
        
        # Crea grafico a barre
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        if f1_per_class_pri:
            colors_pri = {
                'bassa': '#2ecc71',   # Verde
                'media': '#f39c12',   # Arancione
                'alta': '#e74c3c'     # Rosso
            }
            
            # Ordine semantico bassa→media→alta
            ordered_pri = ['bassa', 'media', 'alta']
            values = [f1_per_class_pri.get(p, 0) for p in ordered_pri]
            colors = [colors_pri[p] for p in ordered_pri]
            
            bars3 = ax3.bar(
                ordered_pri, values, 
                color=colors, 
                edgecolor='white', 
                linewidth=1.5
            )
            
            # Etichette valori
            ax3.bar_label(bars3, fmt="%.2f", padding=3, fontweight='bold')
            
            # Configurazione assi
            ax3.set_ylim(0, 1.1)
            ax3.set_ylabel("F1-score", fontweight='bold')
            ax3.set_title("Performance per Priorità", fontweight='bold', pad=15)
            
            # Linea target 0.75 (più bassa per priorità)
            ax3.axhline(
                y=0.75, color='red', linestyle='--', 
                linewidth=1, alpha=0.5, label='Target 0.75'
            )
            ax3.legend(loc='lower right')
            ax3.grid(axis='y', alpha=0.3, linestyle='--')
        
        st.pyplot(fig3)
        plt.close(fig3)
    
    # GRAFICO 4: CONFUSION MATRIX (PRIORITÀ)
    with col_pri2:
        st.markdown("**Matrice di Confusione**")
        
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        cm_pri = M.get("cm_pri")
        labels_pri = M.get("labels_pri")
        
        if cm_pri is not None and labels_pri is not None:
            sns.heatmap(
                cm_pri, 
                annot=True, 
                fmt="d", 
                cmap="Greens",        # Colormap verde (diversa da categoria)
                ax=ax4,
                xticklabels=labels_pri, 
                yticklabels=labels_pri,
                cbar_kws={'label': 'Numero predizioni'},
                linewidths=0.5,
                linecolor='gray'
            )
            
            # Etichette assi
            ax4.set_xlabel("Predetto", fontweight='bold')
            ax4.set_ylabel("Reale", fontweight='bold')
            ax4.set_title("Confusion Matrix - Priorità", fontweight='bold', pad=15)
        
        st.pyplot(fig4)
        plt.close(fig4)
    
    # INSIGHTS AUTOMATICI PRIORITÀ
    with st.expander("💡 Insights Priorità"):
        if f1_per_class_pri:
            st.markdown(f"""
            - **Alta priorità F1**: {f1_per_class_pri.get('alta', 0):.2f} (critico per business)
            - **Media priorità F1**: {f1_per_class_pri.get('media', 0):.2f}
            - **Bassa priorità F1**: {f1_per_class_pri.get('bassa', 0):.2f}
            
            **Nota Importante**: 
            - F1 "alta" deve essere ≥0.75 per minimizzare **falsi negativi** su casi critici
            - Un falso negativo su priorità alta = ticket urgente classificato erroneamente come bassa/media
            - Questo può causare ritardi gravi nella gestione ticket business-critical
            
            **Confusion Matrix - Come Leggerla**:
            - **Diagonale principale** (alto): predizioni corrette
            - **Off-diagonale**: confusioni tra classi (es: alta predetta come media)
            - Valori alti off-diagonale indicano dove il modello confonde
            """)
    # =========================================================================
    # 🆕 SEZIONE: METRICHE NUMERICHE DETTAGLIATE
    # =========================================================================
    st.markdown("---")
    st.subheader("📋 Metriche Numeriche Complete")
    
    with st.expander("🔍 Visualizza Metriche Raw (per Report)", expanded=False):
        # CATEGORIA
        st.markdown("### 🏷️ CATEGORIA")
        
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        with col_metric1:
            st.metric("Accuracy", f"{M.get('acc_cat', 0):.4f}")
        with col_metric2:
            st.metric("F1-Macro", f"{M.get('f1_cat_macro', 0):.4f}")
        with col_metric3:
            test_size_cat = sum(M.get('report_cat', {}).get(label, {}).get('support', 0) 
                                for label in M.get('labels_cat', []))
            st.metric("Test Set Size", f"{test_size_cat}")
        
        st.markdown("**📊 Dettaglio per Classe:**")
        
        # Tabella performance categoria
        cat_data = []
        for label in M.get('labels_cat', []):
            report = M.get('report_cat', {}).get(label, {})
            cat_data.append({
                "Classe": label,
                "Precision": f"{report.get('precision', 0):.4f}",
                "Recall": f"{report.get('recall', 0):.4f}",
                "F1-Score": f"{report.get('f1-score', 0):.4f}",
                "Support": report.get('support', 0)
            })
        
        if cat_data:
            df_cat_metrics = pd.DataFrame(cat_data)
            st.dataframe(df_cat_metrics, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        # PRIORITÀ
        st.markdown("### 🎚️ PRIORITÀ")
        
        col_metric4, col_metric5, col_metric6 = st.columns(3)
        with col_metric4:
            st.metric("Accuracy", f"{M.get('acc_pri', 0):.4f}")
        with col_metric5:
            st.metric("F1-Macro", f"{M.get('f1_pri_macro', 0):.4f}")
        with col_metric6:
            test_size_pri = sum(M.get('report_pri', {}).get(label, {}).get('support', 0) 
                                for label in M.get('labels_pri', []))
            st.metric("Test Set Size", f"{test_size_pri}")
        
        st.markdown("**📊 Dettaglio per Classe:**")
        
        # Tabella performance priorità
        pri_data = []
        for label in M.get('labels_pri', []):
            report = M.get('report_pri', {}).get(label, {})
            pri_data.append({
                "Classe": label,
                "Precision": f"{report.get('precision', 0):.4f}",
                "Recall": f"{report.get('recall', 0):.4f}",
                "F1-Score": f"{report.get('f1-score', 0):.4f}",
                "Support": report.get('support', 0)
            })
        
        if pri_data:
            df_pri_metrics = pd.DataFrame(pri_data)
            st.dataframe(df_pri_metrics, hide_index=True, use_container_width=True)
        
        # INTERPRETAZIONE
        st.markdown("---")
        st.markdown("### 💡 Come Leggere Queste Metriche")
        st.markdown("""
        **Accuracy**: Percentuale predizioni corrette (es. 0.8523 = 85.23%)
        
        **Precision**: Delle predizioni positive, quante sono corrette
        - Alta Precision = pochi falsi allarmi
        - Es: Precision Tecnico 0.90 = 90% ticket predetti come "Tecnico" lo sono davvero
        
        **Recall**: Dei casi reali positivi, quanti sono stati identificati
        - Alto Recall = pochi casi mancati
        - Es: Recall Tecnico 0.85 = 85% ticket realmente "Tecnico" sono stati identificati
        
        **F1-Score**: Media armonica Precision-Recall (bilancia falsi positivi/negativi)
        - Ideale ≥0.80 per categoria, ≥0.75 per priorità
        
        **Support**: Numero esempi test per quella classe
        - Indica affidabilità statistica della metrica
        - Support basso (<20) = metrica meno affidabile
        
        **F1-Macro**: Media semplice F1 per classe (peso uguale a tutte le classi)
        - Utile con dataset sbilanciato
        - Non si lascia "ingannare" da classi maggioritarie
        """)
        
        # COPY-PASTE per REPORT
        st.markdown("---")
        st.markdown("### 📋 Sezione Risultati per Report")
        
        # Genera testo formattato per copia-incolla
        report_text = f"""
RISULTATI SPERIMENTALI

1. PERFORMANCE CATEGORIA
   - Accuracy: {M.get('acc_cat', 0):.2%} ({M.get('acc_cat', 0):.4f})
   - F1-Macro: {M.get('f1_cat_macro', 0):.4f}
   - Test Set: {test_size_cat} ticket
   
   Dettaglio per classe:
"""
        for label in M.get('labels_cat', []):
            report = M.get('report_cat', {}).get(label, {})
            report_text += f"   • {label}: F1={report.get('f1-score', 0):.4f}, Precision={report.get('precision', 0):.4f}, Recall={report.get('recall', 0):.4f}\n"
        
        report_text += f"""
2. PERFORMANCE PRIORITÀ
   - Accuracy: {M.get('acc_pri', 0):.2%} ({M.get('acc_pri', 0):.4f})
   - F1-Macro: {M.get('f1_pri_macro', 0):.4f}
   - Test Set: {test_size_pri} ticket
   
   Dettaglio per classe:
"""
        for label in M.get('labels_pri', []):
            report = M.get('report_pri', {}).get(label, {})
            report_text += f"   • {label}: F1={report.get('f1-score', 0):.4f}, Precision={report.get('precision', 0):.4f}, Recall={report.get('recall', 0):.4f}\n"
        
        st.code(report_text, language=None)


    # =========================================================================
    # SEZIONE 5: NOTE TECNICHE
    # =========================================================================
    st.markdown("---")
    with st.expander("ℹ️ Note Tecniche"):
        st.markdown("""
        ### Metriche Utilizzate
        
        **Accuracy (Accuratezza)**:
        - Percentuale predizioni corrette sul totale
        - Formula: `corrette / totale`
        - Range: 0-100% (più alto = meglio)
        - Limite: non affidabile con classi molto sbilanciate
        
        **F1-score**:
        - Media armonica di Precision e Recall
        - Formula: `2 * (precision * recall) / (precision + recall)`
        - Range: 0-1 (ideale ≥0.80)
        - Bilancia falsi positivi e falsi negativi
        
        **F1 Macro**:
        - Media semplice F1-score per classe (senza pesatura)
        - Utile con classi sbilanciate (dà peso uguale a ogni classe)
        - Non si lascia "ingannare" da classi maggioritarie
        
        **Precision (Precisione)**:
        - Quante predizioni positive sono realmente positive
        - Formula: `veri positivi / (veri positivi + falsi positivi)`
        - Alto = pochi falsi allarmi
        
        **Recall (Richiamo)**:
        - Quanti casi positivi reali sono stati identificati
        - Formula: `veri positivi / (veri positivi + falsi negativi)`
        - Alto = pochi casi mancati
        
        ---
        
        ### Interpretazione Confusion Matrix
        
        **Struttura**:
        - **Righe**: Classi reali (ground truth)
        - **Colonne**: Classi predette dal modello
        - **Diagonale**: Predizioni corrette (più alti = meglio)
        - **Off-diagonale**: Errori/confusioni (più bassi = meglio)
        
        **Esempio Lettura**:
        ```
                    Predetto
                 Tec  Amm  Comm
        Reale Tec  90   5    5      ← 90 corretti, 5 confusi con Amm, 5 con Comm
              Amm  10  80   10      ← 10 confusi con Tec, 80 corretti, 10 con Comm
              Comm  5   5   90      ← 5 confusi con Tec, 5 con Amm, 90 corretti
        ```
        
        ---
        
        ### Configurazione Modelli
        
        **Modello Categoria**:
        - Algoritmo: `LinearSVC` (Support Vector Machine con kernel lineare)
        - Motivo scelta: veloce e performante su dati testuali alta dimensionalità
        - Parametri: `C=1.0` (default), kernel lineare
        
        **Modello Priorità**:
        - Algoritmo: `LogisticRegression`
        - Parametri: `class_weight='balanced'`, `max_iter=500`
        - `class_weight='balanced'`: compensa sbilanciamento classi automaticamente
        - `max_iter=500`: aumentato per garantire convergenza
        
        **Feature Extraction**:
        - Tecnica: `TF-IDF` (Term Frequency - Inverse Document Frequency)
        - Ngram range: (1, 2) - unigrammi + bigrammi
        - Stopwords: lista custom italiana (54 parole)
        - Strip accents: unicode (normalizzazione à→a, è→e)
        - Min DF: 1 (mantiene termini rari, utili per casi specifici)
        - Max DF: 0.95 (scarta termini troppo comuni, presente in >95% documenti)
        
        **Train/Test Split**:
        - Proporzione: 80% train, 20% test
        - Stratificazione: SI (mantiene proporzioni classi in train/test)
        - Random state: 42 (riproducibilità)
        
        ---
        
        ### Sistema Ibrido: ML + Regole
        
        Questo sistema combina:
        1. **Modelli ML** (predizione base)
        2. **Regole basate su keywords** (override predizione ML)
        
        **Vantaggi approccio ibrido**:
        - ML cattura pattern complessi non codificabili
        - Regole garantiscono correttezza su casi business-critical
        - Es: "conferme ordine non inviate" → sempre Commerciale + Alta (regola)
        - Es: "stampante bloccata" → sempre Tecnico (regola)
        
        **Priorità applicazione**:
        1. ML predice categoria e priorità
        2. Regole verificano e sovrascrivono se necessario
        3. Risultato finale = predizione ML o override regola
        
        Questo garantisce:
        - Flessibilità ML per casi nuovi
        - Affidabilità regole per casi critici noti
        """)
