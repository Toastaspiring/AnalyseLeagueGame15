#!/usr/bin/env python3
"""Entraînement, comparaison et sélection du modèle de probabilité de victoire à 15 minutes.

Objectif (projet 1) : estimer à 15:00 la probabilité que l'équipe bleue gagne, pour l'afficher
en direct (overlay de stream) ou sur une VOD. Le modèle n'utilise que des variables disponibles
en direct : l'or d'équipe en est exclu (un modèle « avec or » sert seulement de référence).

Chaîne : dataset nettoyé -> variables d'écart -> découpage temporel -> recherche
d'hyperparamètres en validation croisée -> évaluation sur le test -> sélection -> analyses
(intervalles de confiance, sous-groupes, résidus, importance, convergence) -> export.

Sorties :
    models/modele_retenu.joblib (+ .json)   modèle retenu et ses métadonnées
    models/candidats/<modele>.joblib        tous les modèles comparés
    reports/metrics.json                    métriques et analyses (lues par make_figures.py
                                            et generate_model_card.py)
    reports/predictions_test.csv            probabilités de chaque modèle sur le test
    reports/comparaison_modeles.csv         tableau comparatif

Usage :
    python train_models.py
"""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (CLEAN_CSV, FEATURES, FULL, LIVE, METRICS_JSON, MODELS_DIR, PREDICTIONS_CSV, REPORTS_DIR,
                    SEED, load_dataset, relative)

CV_FOLDS = 5          # validation croisée sur l'apprentissage, pour choisir les hyperparamètres
N_BOOTSTRAP = 500     # rééchantillonnages du test pour les intervalles de confiance
REFERENCE_WITH_GOLD = "logistique_avec_or"


# ---------------------------------------------------------------------------
# Modèles candidats et grilles d'hyperparamètres (recherche volontairement légère)
# ---------------------------------------------------------------------------
def _logistic():
    return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=5000))])


# label : nom affiché ; simplicite : rang utilisé pour départager des modèles équivalents (1 = le plus simple) ;
# variables : "direct" (déployable en direct) ou "complet" (avec l'or, référence non déployable).
CANDIDATES = {
    "reference": dict(
        label="Référence (taux de victoire bleue)", simplicite=0, variables="direct",
        estimator=DummyClassifier(strategy="prior"), grid={}),
    "logistique": dict(
        label="Régression logistique", simplicite=1, variables="direct",
        estimator=_logistic(), grid={"model__C": [0.001, 0.01, 0.1, 1.0, 10.0]}),
    "gradient_boosting": dict(
        label="Gradient boosting", simplicite=2, variables="direct",
        estimator=HistGradientBoostingClassifier(max_iter=500, early_stopping=True, validation_fraction=0.1,
                                                 n_iter_no_change=20, random_state=SEED),
        grid={"learning_rate": [0.03, 0.1], "max_leaf_nodes": [7, 15, 31], "l2_regularization": [0.0, 1.0]}),
    "foret_aleatoire": dict(
        label="Forêt aléatoire", simplicite=3, variables="direct",
        estimator=RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
        grid={"max_depth": [6, 10, 14], "min_samples_leaf": [10, 50]}),
    "reseau_neurones": dict(
        label="Réseau de neurones (MLP)", simplicite=4, variables="direct",
        estimator=Pipeline([("scale", StandardScaler()),
                            ("model", MLPClassifier(max_iter=300, early_stopping=True, n_iter_no_change=10,
                                                    random_state=SEED))]),
        grid={"model__hidden_layer_sizes": [(16,), (32, 16)], "model__alpha": [1e-4, 1e-2],
              "model__learning_rate_init": [1e-3, 1e-2]}),
    REFERENCE_WITH_GOLD: dict(
        label="Logistique avec or (non déployable)", simplicite=1, variables="complet",
        estimator=_logistic(), grid={"model__C": [0.001, 0.01, 0.1, 1.0, 10.0]}),
}
ELIGIBLE = [k for k, c in CANDIDATES.items() if k != "reference" and c["variables"] == "direct"]


def columns_for(key: str) -> list[str]:
    return LIVE if CANDIDATES[key]["variables"] == "direct" else FULL


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------
def ece_score(y, p, n_bins: int = 10) -> float:
    """Erreur de calibration attendue : écart moyen |fréquence observée - probabilité prédite|."""
    y, p = np.asarray(y), np.asarray(p)
    idx = np.clip(np.digitize(p, np.linspace(0, 1, n_bins + 1)[1:-1]), 0, n_bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(n_bins) if (idx == b).any()))


METRICS = {
    "log_loss": lambda y, p: log_loss(y, p, labels=[0, 1]),
    "brier": brier_score_loss,
    "roc_auc": roc_auc_score,
    "average_precision": average_precision_score,
    "accuracy": lambda y, p: accuracy_score(y, p >= 0.5),
    "ece": ece_score,
}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


# ---------------------------------------------------------------------------
# Recherche d'hyperparamètres et évaluation
# ---------------------------------------------------------------------------
def search_model(key, X_train, y_train, cv):
    spec, cols = CANDIDATES[key], columns_for(key)
    gs = GridSearchCV(clone(spec["estimator"]), spec["grid"], scoring="neg_log_loss", cv=cv,
                      n_jobs=1 if key == "foret_aleatoire" else 2, refit=True, return_train_score=True)
    start = time.perf_counter()
    gs.fit(X_train[cols], y_train)
    duration = time.perf_counter() - start
    res = gs.cv_results_
    best = gs.best_index_
    folds = -np.array([res[f"split{k}_test_score"][best] for k in range(CV_FOLDS)])
    configs = [{
        "params": _jsonable(res["params"][i]),
        "cv_log_loss": float(-res["mean_test_score"][i]),
        "cv_log_loss_std": float(res["std_test_score"][i]),
        "train_log_loss": float(-res["mean_train_score"][i]),
        "fit_time_s": float(res["mean_fit_time"][i]),
    } for i in range(len(res["params"]))]
    return gs.best_estimator_, {
        "label": spec["label"], "simplicite": spec["simplicite"], "variables": spec["variables"],
        "colonnes": cols, "grid": _jsonable(spec["grid"]), "best_params": _jsonable(gs.best_params_),
        "n_configs": len(configs), "configs": configs,
        "cv_log_loss": float(folds.mean()), "cv_log_loss_std": float(folds.std(ddof=1)),
        "cv_log_loss_se": float(folds.std(ddof=1) / np.sqrt(CV_FOLDS)), "cv_folds": folds.tolist(),
        "search_time_s": duration, "refit_time_s": float(gs.refit_time_),
    }


def evaluate(estimator, cols, X_test, y_test):
    proba = estimator.predict_proba(X_test[cols])[:, 1]
    timings = []
    for _ in range(5):
        start = time.perf_counter()
        estimator.predict_proba(X_test[cols])
        timings.append(time.perf_counter() - start)
    out = {name: float(func(y_test, proba)) for name, func in METRICS.items()}
    out["predict_ms_per_1000"] = min(timings) / len(X_test) * 1e6
    out["size_kb"] = len(pickle.dumps(estimator)) / 1024
    return proba, out


def paired_bootstrap(y, probas, selected, n_boot=N_BOOTSTRAP):
    """IC à 95 % de (métrique du modèle m) - (métrique du modèle retenu), sur les mêmes tirages du test."""
    rng = np.random.default_rng(SEED)
    y = np.asarray(y)
    n = len(y)
    samples = {k: {m: [] for m in METRICS} for k in probas}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        for k, p in probas.items():
            for m, func in METRICS.items():
                samples[k][m].append(func(y[i], p[i]))
    ci = {}
    for k in probas:
        if k == selected:
            continue
        ci[k] = {}
        for m in METRICS:
            diff = np.array(samples[k][m]) - np.array(samples[selected][m])
            ci[k][m] = {"diff": float(METRICS[m](y, probas[k]) - METRICS[m](y, probas[selected])),
                        "low": float(np.percentile(diff, 2.5)), "high": float(np.percentile(diff, 97.5))}
    return ci


# ---------------------------------------------------------------------------
# Analyses (données des graphes et de la model card)
# ---------------------------------------------------------------------------
def convergence_curves(fitted, X_train, y_train):
    """Perte d'apprentissage et de validation au fil des itérations, pour les modèles itératifs."""
    curves = {}
    X_tr, X_val, y_tr, y_val = train_test_split(X_train[LIVE], y_train, test_size=0.2, stratify=y_train,
                                                random_state=SEED)
    scaler = StandardScaler().fit(X_tr)
    S_tr, S_val = scaler.transform(X_tr), scaler.transform(X_val)

    if "logistique" in fitted:  # itérations du solveur L-BFGS
        C = fitted["logistique"].named_steps["model"].C
        tr, va = [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for it in range(1, 26):
                m = LogisticRegression(C=C, max_iter=it).fit(S_tr, y_tr)
                tr.append(log_loss(y_tr, m.predict_proba(S_tr)[:, 1]))
                va.append(log_loss(y_val, m.predict_proba(S_val)[:, 1]))
        final_iter = int(np.ravel(fitted["logistique"].named_steps["model"].n_iter_)[0])
        curves["logistique"] = {"x": list(range(1, 26)), "train": tr, "val": va, "choisi": final_iter,
                                "unite": "Itérations du solveur L-BFGS"}

    if "gradient_boosting" in fitted:  # itérations de boosting, sans arrêt précoce pour voir la suite
        gb = fitted["gradient_boosting"]
        chosen = int(gb.n_iter_)
        long_run = clone(gb).set_params(early_stopping=False, max_iter=max(3 * chosen, 150)).fit(X_tr, y_tr)
        tr = [log_loss(y_tr, p[:, 1]) for p in long_run.staged_predict_proba(X_tr)]
        va = [log_loss(y_val, p[:, 1]) for p in long_run.staged_predict_proba(X_val)]
        curves["gradient_boosting"] = {"x": list(range(1, len(tr) + 1)), "train": tr, "val": va, "choisi": chosen,
                                       "unite": "Arbres ajoutés (itérations de boosting)"}

    if "reseau_neurones" in fitted:  # époques (un passage complet sur les données)
        mlp_best = fitted["reseau_neurones"].named_steps["model"]
        mlp = MLPClassifier(hidden_layer_sizes=mlp_best.hidden_layer_sizes, alpha=mlp_best.alpha,
                            learning_rate_init=mlp_best.learning_rate_init, random_state=SEED)
        tr, va = [], []
        epochs = max(3 * int(mlp_best.n_iter_), 40)
        for _ in range(epochs):
            mlp.partial_fit(S_tr, y_tr, classes=[0, 1])
            tr.append(log_loss(y_tr, mlp.predict_proba(S_tr)[:, 1]))
            va.append(log_loss(y_val, mlp.predict_proba(S_val)[:, 1]))
        curves["reseau_neurones"] = {"x": list(range(1, epochs + 1)), "train": tr, "val": va,
                                     "choisi": int(mlp_best.n_iter_), "unite": "Époques"}
    return curves


def learning_curves(fitted, X_train, y_train):
    """Log loss de validation selon la taille de l'apprentissage."""
    sizes = np.array([0.02, 0.05, 0.15, 0.4, 1.0])
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    results = {}
    for key in ELIGIBLE:
        n_abs, _, val = learning_curve(clone(fitted[key]), X_train[LIVE], y_train, train_sizes=sizes, cv=cv,
                                       scoring="neg_log_loss", shuffle=True, random_state=SEED,
                                       n_jobs=1 if key == "foret_aleatoire" else 2)
        results[key] = {"n": n_abs.tolist(), "val": (-val.mean(axis=1)).tolist()}
    return results


SUBGROUPS = [("Partie serrée", 0, 1000), ("Avantage net", 1000, 3000), ("Large avantage", 3000, np.inf)]


def subgroup_labels(gold_gap_abs: pd.Series) -> pd.Series:
    labels = pd.Series(index=gold_gap_abs.index, dtype="object")
    for name, lo, hi in SUBGROUPS:
        labels[(gold_gap_abs >= lo) & (gold_gap_abs < hi)] = name
    return labels


def subgroup_metrics(probas, y_test, groups):
    out = {}
    for key, p in probas.items():
        out[key] = {}
        for name, _, _ in SUBGROUPS:
            m = (groups == name).to_numpy()
            yy, pp = y_test.to_numpy()[m], p[m]
            out[key][name] = {"n": int(m.sum()), "accuracy": float(accuracy_score(yy, pp >= 0.5)),
                              "roc_auc": float(roc_auc_score(yy, pp)) if len(set(yy)) == 2 else None,
                              "log_loss": float(log_loss(yy, pp, labels=[0, 1])),
                              "brier": float(brier_score_loss(yy, pp)),
                              "proba_moyenne": float(pp.mean()), "victoires_bleues": float(yy.mean())}
    return out


def binned_residuals(p, y, edges):
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= 50:
            pm = p[m].mean()
            rows.append({"x": float(pm), "residu": float((y[m] - p[m]).mean()),
                         "se2": float(2 * np.sqrt(pm * (1 - pm) / m.sum())), "n": int(m.sum())})
    return rows


def residuals_by_feature(p, y, feature, edges):
    idx = np.clip(np.digitize(feature, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= 50:
            rows.append({"x": float(feature[m].mean()), "residu": float((y[m] - p[m]).mean()), "n": int(m.sum())})
    return rows


def residual_analysis(probas, y_test, gold_gap, selected):
    y = y_test.to_numpy()
    by_proba = binned_residuals(probas[selected], y, np.quantile(probas[selected], np.linspace(0, 1, 21)))
    edges = np.arange(-8000, 8001, 1000)
    by_gold = {k: residuals_by_feature(probas[k], y, gold_gap, edges) for k in (selected, REFERENCE_WITH_GOLD)}
    return by_proba, by_gold


def importance(estimator, cols, X_test, y_test):
    imp = permutation_importance(estimator, X_test[cols], y_test, scoring="neg_log_loss", n_repeats=10,
                                 random_state=SEED, n_jobs=1)
    order = np.argsort(imp.importances_mean)[::-1]
    return [{"variable": cols[i], "hausse_log_loss": float(imp.importances_mean[i]),
             "ecart_type": float(imp.importances_std[i])} for i in order]


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------
def main():
    for folder in (MODELS_DIR, MODELS_DIR / "candidats", REPORTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    df, X, y, train, test = load_dataset()
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    print(f"Données : {len(df)} parties ; apprentissage {train.sum()} (plus anciennes), test {test.sum()} (plus récentes)")

    # 1. Recherche d'hyperparamètres et évaluation de chaque candidat
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    fitted, results, probas = {}, {}, {}
    for key in CANDIDATES:
        fitted[key], results[key] = search_model(key, X_train, y_train, cv)
        probas[key], results[key]["test"] = evaluate(fitted[key], columns_for(key), X_test, y_test)
        p_train = fitted[key].predict_proba(X_train[columns_for(key)])[:, 1]
        results[key]["train"] = {"log_loss": float(log_loss(y_train, p_train, labels=[0, 1])),
                                 "roc_auc": float(roc_auc_score(y_train, p_train)),
                                 "accuracy": float(accuracy_score(y_train, p_train >= 0.5))}
        r = results[key]
        print(f"{r['label']:<38} {r['n_configs']:>2} config. | log loss CV {r['cv_log_loss']:.4f} "
              f"(± {r['cv_log_loss_se']:.4f}) | test : log loss {r['test']['log_loss']:.4f}, "
              f"AUC {r['test']['roc_auc']:.4f}, exactitude {r['test']['accuracy']:.4f} | {r['search_time_s']:.0f} s")

    # 2. Sélection : règle d'un écart-type (erreur standard) sur la log loss de validation croisée
    best = min(ELIGIBLE, key=lambda k: results[k]["cv_log_loss"])
    threshold = results[best]["cv_log_loss"] + results[best]["cv_log_loss_se"]
    within = [k for k in ELIGIBLE if results[k]["cv_log_loss"] <= threshold]
    selected = min(within, key=lambda k: CANDIDATES[k]["simplicite"])
    selection = {
        "regle": "Parmi les modèles déployables en direct, on garde ceux dont la log loss de validation croisée "
                 "dépasse celle du meilleur de moins d'une erreur standard, puis le plus simple d'entre eux.",
        "meilleur_cv": best, "seuil": threshold, "dans_la_tolerance": within, "retenu": selected,
        "ordre_de_simplicite": {k: CANDIDATES[k]["simplicite"] for k in ELIGIBLE},
    }
    print(f"\nMeilleure log loss CV : {results[best]['label']} ; tolérance {threshold:.4f} ; "
          f"dans la tolérance : {', '.join(results[k]['label'] for k in within)}")
    print(f"Modèle retenu : {results[selected]['label']}")

    # 3. Contrôles de surapprentissage et de fuite d'information
    rng = np.random.default_rng(SEED)
    cols = columns_for(selected)
    shuffled = [float(log_loss(y_test, clone(fitted[selected]).fit(X_train[cols], rng.permutation(y_train.to_numpy()))
                               .predict_proba(X_test[cols])[:, 1], labels=[0, 1])) for _ in range(5)]
    single_auc = {c: float(roc_auc_score(y_test, X_test[c])) for c in FULL}
    print(f"Cibles mélangées : log loss test {np.mean(shuffled):.4f} (hasard : 0.6931)")

    # 4. Analyses
    print("Intervalles de confiance (bootstrap apparié)...")
    ci = paired_bootstrap(y_test, {k: probas[k] for k in CANDIDATES}, selected)
    print("Analyses (convergence, courbes d'apprentissage, sous-groupes, résidus, importance)...")
    gold_gap = X_test["ecart_or"].to_numpy()
    groups = subgroup_labels(pd.Series(np.abs(gold_gap), index=y_test.index))
    curves = convergence_curves(fitted, X_train, y_train)
    learning = learning_curves(fitted, X_train, y_train)
    sub = subgroup_metrics(probas, y_test, groups)
    res_proba, res_gold = residual_analysis(probas, y_test, gold_gap, selected)
    explanation = {"importance_permutation": importance(fitted[selected], cols, X_test, y_test)}
    model = fitted[selected]
    if isinstance(model, Pipeline) and isinstance(model.named_steps.get("model"), LogisticRegression):
        scaler, logreg = model.named_steps["scale"], model.named_steps["model"]
        explanation["intercept"] = float(logreg.intercept_[0] - np.sum(logreg.coef_[0] * scaler.mean_ / scaler.scale_))
        explanation["coefficients"] = [{
            "variable": col, "coef_standardise": float(c), "ecart_type_variable": float(s),
            "coef_par_unite": float(c / s), "multiplicateur_cote": float(np.exp(c / s)),
        } for col, c, s in zip(cols, logreg.coef_[0], scaler.scale_)]

    # 5. Exports
    data_sha = hashlib.sha256(CLEAN_CSV.read_bytes()).hexdigest()
    joblib.dump(model, MODELS_DIR / "modele_retenu.joblib")
    for key, estimator in fitted.items():
        joblib.dump(estimator, MODELS_DIR / "candidats" / f"{key}.joblib")
    metadata = {
        "modele": selected, "label": CANDIDATES[selected]["label"], "variables": cols,
        "hyperparametres": results[selected]["best_params"], "seuil_decision": 0.5, "sortie": "P(victoire bleue)",
        "donnees": {"fichier": relative(CLEAN_CSV), "sha256": data_sha},
        "entraine_le": datetime.now().isoformat(timespec="seconds"), "scikit_learn": sklearn.__version__,
    }
    (MODELS_DIR / "modele_retenu.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    predictions = pd.DataFrame({"matchId": df.loc[test, "matchId"].to_numpy(), "blue_win": y_test.to_numpy(),
                                "groupe": groups.to_numpy()})
    for col in FULL:
        predictions[col] = X_test[col].to_numpy()
    for key, p in probas.items():
        predictions[f"proba_{key}"] = p
    predictions.to_csv(PREDICTIONS_CSV, index=False)

    metrics = {
        "genere_le": datetime.now().isoformat(timespec="seconds"),
        "objectif": "Probabilité de victoire de l'équipe bleue à 15 minutes, calibrée, avec des variables disponibles en direct",
        "donnees": {
            "fichier": relative(CLEAN_CSV), "sha256": data_sha, "lignes": int(len(df)),
            "apprentissage": int(train.sum()), "test": int(test.sum()),
            "decoupage": "apprentissage sur les 80 % de parties aux matchId les plus anciens, test sur les 20 % les plus récents",
            "matchid_apprentissage": [int(df.loc[train, "matchId"].min()), int(df.loc[train, "matchId"].max())],
            "matchid_test": [int(df.loc[test, "matchId"].min()), int(df.loc[test, "matchId"].max())],
            "victoires_bleues_apprentissage": float(y_train.mean()), "victoires_bleues_test": float(y_test.mean()),
        },
        "variables": {name: {"description": desc, "direct": live} for name, (desc, live, _) in FEATURES.items()},
        "validation_croisee": {"plis": CV_FOLDS, "critere": "log loss", "melange": True, "graine": SEED},
        "modeles": results,
        "selection": selection,
        "ic_bootstrap": {"n": N_BOOTSTRAP, "differences": ci},
        "sous_groupes": {"definition": [[n, lo, None if not np.isfinite(hi) else hi] for n, lo, hi in SUBGROUPS],
                         "metriques": sub},
        "residus": {"par_probabilite": res_proba, "par_ecart_or": res_gold},
        "convergence": curves,
        "controle_surapprentissage": {"cibles_melangees_log_loss_test": shuffled, "auc_variable_seule_test": single_auc},
        "courbes_apprentissage": learning,
        "explication": explanation,
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    METRICS_JSON.write_text(json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False), encoding="utf-8")
    table = pd.DataFrame([{
        "modele": k, "label": r["label"], "variables": r["variables"], "cv_log_loss": r["cv_log_loss"],
        "cv_log_loss_se": r["cv_log_loss_se"], **{f"test_{m}": r["test"][m] for m in METRICS},
        "predict_ms_per_1000": r["test"]["predict_ms_per_1000"], "size_kb": r["test"]["size_kb"],
        "refit_time_s": r["refit_time_s"], "best_params": json.dumps(r["best_params"], ensure_ascii=False),
        "retenu": k == selected,
    } for k, r in results.items()])
    table.to_csv(REPORTS_DIR / "comparaison_modeles.csv", index=False)
    print("\nExports : models/modele_retenu.joblib, models/candidats/, reports/metrics.json, "
          "reports/predictions_test.csv, reports/comparaison_modeles.csv")


if __name__ == "__main__":
    main()
