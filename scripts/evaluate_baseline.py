import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def pick_col(df, candidates):
    """Ritorna la prima colonna presente nel df tra quelle candidate."""
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Nessuna colonna trovata tra: {candidates}\n"
        f"Colonne disponibili: {list(df.columns)}"
    )


def build_text_series(df, col_title, col_body):
    return (df[col_title].fillna("") + " " + df[col_body].fillna("")).str.strip()


def eval_one(model, Xtr, Xte, ytr, yte):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    acc = accuracy_score(yte, pred)
    f1m = f1_score(yte, pred, average="macro")
    return acc, f1m


def run_task(task_name, X_text, y, seed, test_size, vec_params, models_dict):
    """
    Esegue:
      - split stratificato su y
      - TF-IDF fit su train
      - confronto baseline Dummy + modello(i) scelto(i)
    """
    idx_train, idx_test = train_test_split(
        y.index,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    X_train = X_text.loc[idx_train]
    X_test  = X_text.loc[idx_test]
    y_train = y.loc[idx_train]
    y_test  = y.loc[idx_test]

    vec = TfidfVectorizer(**vec_params)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    results = {}
    for name, model in models_dict.items():
        acc, f1m = eval_one(model, Xtr, Xte, y_train, y_test)
        results[name] = {"accuracy": acc, "f1_macro": f1m}

    return results, len(idx_test)


def main(path, seed=42, test_size=0.2):
    df = pd.read_csv(path)

    # ---- mapping colonne IT/EN (robusto) ----
    col_title = pick_col(df, ["titolo", "title", "oggetto", "subject"])
    col_body  = pick_col(df, ["descrizione", "body", "description", "testo"])
    col_cat   = pick_col(df, ["categoria", "category"])
    col_pri   = pick_col(df, ["priorità", "priorita", "priority"])

    X_text = build_text_series(df, col_title, col_body)
    y_cat = df[col_cat].astype(str)
    y_pri = df[col_pri].astype(str)

    # ---- TF-IDF params (semplice, coerente con progetto) ----
    vec_params = dict(
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents="unicode"
        # se vuoi allineare ancora di più al tuo progetto:
        # stop_words=..., min_df=..., max_df=...
    )

    # ---- Modelli (solo ML, senza regole) ----
    models_cat = {
        "Dummy_most_frequent": DummyClassifier(strategy="most_frequent", random_state=seed),
        "Dummy_stratified": DummyClassifier(strategy="stratified", random_state=seed),
        "LinearSVC_scelto": LinearSVC()
    }

    models_pri = {
        "Dummy_most_frequent": DummyClassifier(strategy="most_frequent", random_state=seed),
        "Dummy_stratified": DummyClassifier(strategy="stratified", random_state=seed),
        "LogReg_balanced_scelto": LogisticRegression(max_iter=2000, class_weight="balanced")
    }

    # ---- DOPPIO SPLIT: uno per task ----
    res_cat, ntest_cat = run_task(
        task_name="categoria",
        X_text=X_text,
        y=y_cat,
        seed=seed,
        test_size=test_size,
        vec_params=vec_params,
        models_dict=models_cat
    )

    res_pri, ntest_pri = run_task(
        task_name="priorità",
        X_text=X_text,
        y=y_pri,
        seed=seed,
        test_size=test_size,
        vec_params=vec_params,
        models_dict=models_pri
    )

    # ---- stampa risultati ----
    print("\n=== CONFRONTO BASELINE (Dummy) vs MODELLO SCELTO — SOLO ML (senza regole) ===")
    print(f"Dataset: {path}")
    print(f"seed={seed} | test_size={test_size}")

    print("\n[CATEGORIA] (split stratificato su categoria)")
    print(f"Test set: {ntest_cat}")
    for k, v in res_cat.items():
        print(f"{k:22s}  accuracy={v['accuracy']:.4f}  f1_macro={v['f1_macro']:.4f}")

    print("\n[PRIORITÀ] (split stratificato su priorità)")
    print(f"Test set: {ntest_pri}")
    for k, v in res_pri.items():
        print(f"{k:22s}  accuracy={v['accuracy']:.4f}  f1_macro={v['f1_macro']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path del CSV (es. ticket_sintetici.csv)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()
    main(args.data, seed=args.seed, test_size=args.test_size)