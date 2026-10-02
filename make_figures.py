#!/usr/bin/env python3
"""Génération des graphes à partir des résultats d'entraînement (sans réentraîner).

Entrées
    reports/metrics.json           métriques et analyses (train_models.py)
    reports/predictions_test.csv   probabilités de chaque modèle sur le test (train_models.py)
    data/clean/...csv              dataset nettoyé, pour la matrice de corrélation (clean_data.py)
Sorties
    figures/*.png                  un fichier par graphe
    figures/manifest.json          titre, légende et section de chaque graphe : generate_model_card.py
                                   s'en sert pour insérer automatiquement les graphes dans la model card

Pour ajouter un graphe : écrire une fonction qui trace la figure puis appelle save(), et l'appeler
dans main(). Il apparaîtra tout seul dans la model card (section « Autres graphes » si sa section
n'est pas prévue).

Usage :
    python make_figures.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from common import FIGURES_DIR, FULL, MANIFEST_JSON, METRICS_JSON, PREDICTIONS_CSV, SHORT, TARGET, load_dataset

REFERENCE_WITH_GOLD = "logistique_avec_or"

# Couleurs : une par modèle, toujours la même d'un graphe à l'autre (palette validée pour le daltonisme)
MODEL_COLORS = {"reference": "#898781", "logistique": "#2a78d6", "gradient_boosting": "#eb6834",
                "foret_aleatoire": "#1baf7a", "reseau_neurones": "#eda100", REFERENCE_WITH_GOLD: "#e87ba4"}
SPARE_COLORS = ["#008300", "#4a3aa7", "#52514e"]  # pour les modèles ajoutés plus tard
BLUE, RED = "#2a78d6", "#e34948"
INK, INK_SOFT, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

DECIMAL = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
THOUSANDS = FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " "))
MANIFEST: list[dict] = []


# ---------------------------------------------------------------------------
# Outils
# ---------------------------------------------------------------------------
def setup_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "font.size": 9.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "axes.labelcolor": INK_SOFT, "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK_SOFT, "ytick.labelcolor": INK_SOFT, "legend.frameon": False,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def color(key: str) -> str:
    """Couleur d'un modèle ; un modèle ajouté reçoit la prochaine couleur de réserve."""
    if key not in MODEL_COLORS:
        n_spare = sum(1 for c in MODEL_COLORS.values() if c in SPARE_COLORS)
        MODEL_COLORS[key] = SPARE_COLORS[min(n_spare, len(SPARE_COLORS) - 1)]
    return MODEL_COLORS[key]


def fr(x, decimals: int = 3) -> str:
    return f"{x:.{decimals}f}".replace(".", ",")


def save(fig, name: str, title: str, caption: str, section: str) -> None:
    """Enregistre la figure et la déclare dans le manifeste (lu par generate_model_card.py)."""
    fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    MANIFEST.append({"fichier": f"figures/{name}", "titre": title, "legende": caption, "section": section})


def suptitle(fig, text: str) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=12, fontweight="semibold", color=INK)


class Results:
    """Accès simple aux résultats d'entraînement."""

    def __init__(self, metrics: dict, predictions: pd.DataFrame):
        self.metrics = metrics
        self.pred = predictions
        self.models = metrics["modeles"]
        self.selected = metrics["selection"]["retenu"]
        self.eligible = list(metrics["selection"]["ordre_de_simplicite"])
        self.y = predictions[TARGET].to_numpy()

    def label(self, key: str) -> str:
        return self.models[key]["label"]

    def proba(self, key: str) -> np.ndarray:
        return self.pred[f"proba_{key}"].to_numpy()

    @property
    def compared(self) -> list[str]:
        """Modèles déployables, puis la référence avec or si elle existe."""
        return self.eligible + ([REFERENCE_WITH_GOLD] if REFERENCE_WITH_GOLD in self.models else [])


# ---------------------------------------------------------------------------
# Graphes
# ---------------------------------------------------------------------------
def fig_correlations():
    _, X, y, train, _ = load_dataset()
    data = X.loc[train, FULL].assign(**{TARGET: y[train].values})
    corr = data.corr()
    names = [SHORT.get(c, c) for c in corr.columns]
    cmap = LinearSegmentedColormap.from_list("rouge_gris_bleu", [RED, "#f0efec", BLUE])
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    image = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xticks(np.arange(-0.5, len(names)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names)), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(names)):
        for j in range(len(names)):
            v = corr.values[i, j]
            ax.text(j, i, fr(v, 2), ha="center", va="center", fontsize=8.5, color="white" if abs(v) > 0.6 else INK)
    bar = fig.colorbar(image, ax=ax, shrink=0.8)
    bar.outline.set_visible(False)
    bar.ax.yaxis.set_major_formatter(DECIMAL)
    ax.set_title("Corrélations entre les variables d'écart et la victoire bleue (apprentissage)")
    save(fig, "01_correlations.png", "Matrice de corrélation",
         "Corrélations de Pearson sur le jeu d'apprentissage. Les variables d'écart sont fortement liées entre "
         "elles (une élimination rapporte de l'or et de l'expérience) : prudence dans la lecture des coefficients.",
         "donnees")


def fig_convergence(r: Results):
    curves = r.metrics.get("convergence", {})
    keys = [k for k in ("logistique", "gradient_boosting", "reseau_neurones") if k in curves]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(4.5 * len(keys), 4.1), squeeze=False)
    for ax, key in zip(axes[0], keys):
        c = curves[key]
        ax.plot(c["x"], c["train"], color=MUTED, linewidth=1.6, label="Apprentissage")
        ax.plot(c["x"], c["val"], color=color(key), label="Validation")
        ax.axvline(c["choisi"], color=AXIS, linewidth=1)
        ax.text(c["choisi"], 0.97, f" arrêt : {c['choisi']}", transform=ax.get_xaxis_transform(),
                va="top", fontsize=8.5, color=INK_SOFT)
        lo = min(min(c["train"]), min(c["val"]))
        ax.set_ylim(lo - 0.01, lo + 0.12)
        ax.set_title(r.label(key))
        ax.set_xlabel(c["unite"])
        ax.yaxis.set_major_formatter(DECIMAL)
    axes[0][0].set_ylabel("Log loss")
    axes[0][0].legend(loc="upper right")
    suptitle(fig, "Graphe de convergence : perte d'apprentissage et de validation au fil de l'entraînement")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "02_convergence.png", "Graphe de convergence",
         "Log loss sur 80 % de l'apprentissage (gris) et sur les 20 % restants (couleur), au fil des itérations. "
         "La ligne verticale indique l'itération retenue par le modèle final (convergence du solveur ou arrêt précoce). "
         "Axe vertical limité au voisinage du minimum.", "entrainement")


def fig_learning_curves(r: Results):
    learning = r.metrics.get("courbes_apprentissage", {})
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for key, c in learning.items():
        ax.plot(c["n"], c["val"], color=color(key), marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=r.label(key))
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(THOUSANDS)
    ax.yaxis.set_major_formatter(DECIMAL)
    ax.set_xlabel("Nombre de parties d'apprentissage (échelle logarithmique)")
    ax.set_ylabel("Log loss en validation")
    ax.legend()
    ax.set_title("Courbes d'apprentissage : la performance converge-t-elle avec plus de données ?")
    save(fig, "03_courbes_apprentissage.png", "Courbes d'apprentissage",
         "Log loss en validation croisée (3 plis) selon le nombre de parties d'apprentissage. La référence "
         "(log loss de 0,693) est hors du cadre.", "entrainement")


def fig_roc_pr(r: Results):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    for key in r.compared:
        fpr, tpr, _ = roc_curve(r.y, r.proba(key))
        ax.plot(fpr, tpr, color=color(key), linewidth=1.8,
                label=f"{r.label(key)} (AUC {fr(r.models[key]['test']['roc_auc'])})")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.xaxis.set_major_formatter(DECIMAL)
    ax.yaxis.set_major_formatter(DECIMAL)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("Courbes ROC sur le jeu de test")
    save(fig, "04_roc.png", "Courbes ROC",
         "Positif = victoire bleue. Les courbes se superposent presque : tous les modèles classent les parties "
         "de façon équivalente. La diagonale correspond au hasard.", "classement")

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    prevalence = float(np.mean(r.y))
    ax.axhline(prevalence, color=AXIS, linewidth=1)
    for key in r.compared:
        precision, recall, _ = precision_recall_curve(r.y, r.proba(key))
        ax.plot(recall, precision, color=color(key), linewidth=1.8,
                label=f"{r.label(key)} (AP {fr(r.models[key]['test']['average_precision'])})")
    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.xaxis.set_major_formatter(DECIMAL)
    ax.yaxis.set_major_formatter(DECIMAL)
    ax.set_ylim(0.4, 1.01)
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_title("Courbes précision-rappel sur le jeu de test")
    save(fig, "05_precision_rappel.png", "Courbes précision-rappel",
         f"Positif = victoire bleue. La ligne horizontale ({fr(prevalence * 100, 1)} %) est la précision d'un "
         "modèle au hasard. AP = précision moyenne (aire sous la courbe).", "classement")


def fig_calibration(r: Results):
    fig, (ax, ax_hist) = plt.subplots(2, 1, figsize=(6.6, 7.2), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    for key in r.compared:
        observed, predicted = calibration_curve(r.y, r.proba(key), n_bins=10)
        ax.plot(predicted, observed, color=color(key), marker="o", markersize=4.5, markeredgecolor=SURFACE,
                markeredgewidth=1.2, linewidth=1.6, label=r.label(key))
    ax.set_ylabel("Fréquence observée de victoire bleue")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Prédiction vs réalité : probabilités prédites et victoires observées")
    ax_hist.hist(r.proba(r.selected), bins=np.linspace(0, 1, 21), color=color(r.selected), edgecolor=SURFACE,
                 linewidth=1.5)
    ax_hist.set_xlabel("Probabilité prédite de victoire bleue")
    ax_hist.set_ylabel("Parties")
    ax_hist.xaxis.set_major_formatter(DECIMAL)
    ax.yaxis.set_major_formatter(DECIMAL)
    ax_hist.yaxis.set_major_formatter(THOUSANDS)
    ax_hist.set_title(f"Répartition des probabilités du modèle retenu ({r.label(r.selected)})", fontsize=9.5)
    fig.tight_layout()
    save(fig, "06_prediction_vs_realite.png", "Prédiction vs réalité (calibration)",
         "Chaque point regroupe les parties de test par tranche de 10 % de probabilité prédite. Un modèle bien "
         "calibré suit la diagonale : quand il annonce 70 %, l'équipe bleue gagne 7 fois sur 10.", "calibration")


def fig_confusion(r: Results):
    cm = confusion_matrix(r.y, r.proba(r.selected) >= 0.5)
    share = cm / cm.sum(axis=1, keepdims=True)
    labels = ["Victoire rouge", "Victoire bleue"]
    cmap = LinearSegmentedColormap.from_list("bleu", ["#f0efec", BLUE])
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}".replace(",", " ") + f"\n{fr(share[i, j] * 100, 1)} %", ha="center",
                    va="center", fontsize=10.5, color="white" if share[i, j] > 0.55 else INK)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Prédiction (seuil 0,5)")
    ax.set_ylabel("Réalité")
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Matrice de confusion : {r.label(r.selected)}")
    save(fig, "07_matrice_confusion.png", "Matrice de confusion du modèle retenu",
         "Effectifs et pourcentage de chaque ligne (rappel par classe) sur le jeu de test, au seuil de 0,5.",
         "calibration")


def fig_subgroups(r: Results):
    definition = r.metrics["sous_groupes"]["definition"]
    sub = r.metrics["sous_groupes"]["metriques"]
    groups = r.pred["groupe"].to_numpy()
    fig, axes = plt.subplots(1, len(definition), figsize=(13, 3.9), sharey=True)
    bins = np.linspace(0, 1, 21)
    for ax, (name, lo, hi) in zip(axes, definition):
        m = groups == name
        p, yy = r.proba(r.selected)[m], r.y[m]
        ax.hist(p[yy == 1], bins=bins, histtype="step", linewidth=1.8, color=BLUE, label="Victoire bleue réelle")
        ax.hist(p[yy == 0], bins=bins, histtype="step", linewidth=1.8, color=RED, label="Victoire rouge réelle")
        rng = f"{lo:,.0f} à {hi:,.0f}".replace(",", " ") if hi is not None else f"plus de {lo:,.0f}".replace(",", " ")
        stats = sub[r.selected][name]
        n_games = f"{stats['n']:,}".replace(",", " ")
        ax.set_title(f"{name} (écart d'or {rng})\n{n_games} parties, exactitude {fr(stats['accuracy'] * 100, 1)} %")
        ax.set_xlabel("Probabilité prédite de victoire bleue")
        ax.xaxis.set_major_formatter(DECIMAL)
        ax.yaxis.set_major_formatter(THOUSANDS)
    axes[0].set_ylabel("Parties")
    axes[0].legend(loc="upper center", fontsize=8.5)
    suptitle(fig, f"Analyse par sous-groupe : probabilités prédites selon l'issue réelle ({r.label(r.selected)})")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "08_histogrammes_sous_groupes.png", "Histogrammes par sous-groupe",
         "Parties de test réparties selon l'écart d'or à 15 minutes (une variable que le modèle ne voit pas). "
         "Plus les deux histogrammes sont séparés, plus le modèle distingue bien les issues.", "sous_groupes")

    keys = ["reference"] + r.compared
    names = [name for name, _, _ in definition]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    width = 0.8 / len(keys)
    x = np.arange(len(names))
    for ax, metric, ylabel in ((axes[0], "accuracy", "Exactitude (%)"), (axes[1], "log_loss", "Log loss")):
        for i, key in enumerate(keys):
            vals = [sub[key][name][metric] * (100 if metric == "accuracy" else 1) for name in names]
            ax.bar(x - 0.4 + width * (i + 0.5), vals, width=width * 0.92, color=color(key), label=r.label(key))
        ax.set_xticks(x, names)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(DECIMAL)
    axes[0].set_title("Exactitude par sous-groupe (plus haut = mieux)")
    axes[1].set_title("Log loss par sous-groupe (plus bas = mieux)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "09_metriques_sous_groupes.png", "Performance par sous-groupe",
         "Mêmes sous-groupes que ci-dessus, pour tous les modèles. Les parties serrées sont, logiquement, les plus "
         "difficiles à prédire.", "sous_groupes")


def fig_residuals(r: Results):
    res = r.metrics["residus"]
    by_proba, by_gold = res["par_probabilite"], res["par_ecart_or"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    ax = axes[0]
    xs = [row["x"] for row in by_proba]
    ax.fill_between(xs, [-row["se2"] for row in by_proba], [row["se2"] for row in by_proba], color=GRID,
                    linewidth=0, label="± 2 erreurs types")
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.plot(xs, [row["residu"] for row in by_proba], color=color(r.selected), marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label="Résidu moyen")
    ax.set_xlabel("Probabilité prédite (20 classes de même effectif)")
    ax.xaxis.set_major_formatter(DECIMAL)
    ax.set_ylabel("Résidu moyen (réalité - probabilité)")
    ax.yaxis.set_major_formatter(DECIMAL)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title(f"Résidus selon la prédiction : {r.label(r.selected)}")
    ax = axes[1]
    ax.axhline(0, color=AXIS, linewidth=1)
    for key, rows in by_gold.items():
        ax.plot([row["x"] for row in rows], [row["residu"] for row in rows], color=color(key), marker="o",
                markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=r.label(key))
    ax.set_xlabel("Écart d'or à 15 minutes (classes de 1 000)")
    ax.xaxis.set_major_formatter(THOUSANDS)
    ax.yaxis.set_major_formatter(DECIMAL)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Résidus selon l'écart d'or (variable absente du modèle en direct)")
    fig.tight_layout()
    save(fig, "10_residus.png", "Analyse des résidus",
         "Résidu = issue réelle (0 ou 1) moins probabilité prédite, moyenné par classe. À gauche, des résidus dans "
         "la bande grise signalent un modèle sans biais systématique. À droite, une tendance révèle l'information "
         "que l'or apporterait au modèle en direct.", "residus")


def fig_importance(r: Results):
    rows = r.metrics["explication"]["importance_permutation"][::-1]  # du moins au plus important
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.barh(np.arange(len(rows)), [row["hausse_log_loss"] for row in rows],
            xerr=[row["ecart_type"] for row in rows], height=0.55, color=color(r.selected),
            error_kw={"ecolor": INK_SOFT, "elinewidth": 1})
    ax.set_yticks(np.arange(len(rows)), [SHORT.get(row["variable"], row["variable"]) for row in rows])
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(DECIMAL)
    ax.set_xlabel("Hausse de la log loss quand la variable est mélangée")
    ax.set_title(f"Importance des variables par permutation : {r.label(r.selected)}")
    save(fig, "11_importance_variables.png", "Importance des variables",
         "Chaque variable est mélangée au hasard sur le jeu de test (10 fois) : plus la log loss augmente, plus le "
         "modèle dépend de cette variable. Les barres d'erreur donnent l'écart-type.", "explication")


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------
def main():
    for path in (METRICS_JSON, PREDICTIONS_CSV):
        if not path.exists():
            raise SystemExit(f"{path.name} introuvable : lancer d'abord « python train_models.py ».")
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for old in FIGURES_DIR.glob("*.png"):
        old.unlink()

    r = Results(json.loads(METRICS_JSON.read_text(encoding="utf-8")), pd.read_csv(PREDICTIONS_CSV))
    fig_correlations()
    fig_convergence(r)
    fig_learning_curves(r)
    fig_roc_pr(r)
    fig_calibration(r)
    fig_confusion(r)
    fig_subgroups(r)
    fig_residuals(r)
    fig_importance(r)

    MANIFEST_JSON.write_text(json.dumps(MANIFEST, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(MANIFEST)} graphes enregistrés dans figures/ (liste : figures/manifest.json)")


if __name__ == "__main__":
    main()
