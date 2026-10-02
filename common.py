"""Définitions partagées par les scripts du projet.

Un seul endroit pour les chemins, les données d'entrée, les variables du modèle (écarts
entre les deux équipes) et le découpage apprentissage / test. Modifier une définition
ici la modifie pour tous les scripts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Données d'entrée (versionnées dans le dépôt)
# ---------------------------------------------------------------------------
OPENML_ID = 43635
OPENML_MD5 = "e717b2ddbc4160b17253a1e8a8296bc7"  # somme publiée par OpenML pour l'ARFF (version 1)
RAW_ARFF = ROOT / "data" / "raw" / "dataset_43635.arff"
RAW_METADATA = ROOT / "data" / "raw" / "openml_43635_metadata.json"

# ---------------------------------------------------------------------------
# Fichiers produits par la chaîne (jamais versionnés, régénérés en local)
# ---------------------------------------------------------------------------
CLEAN_CSV = ROOT / "data" / "clean" / "lol_diamond_15min_clean.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = ROOT / "figures"
METRICS_JSON = REPORTS_DIR / "metrics.json"
PREDICTIONS_CSV = REPORTS_DIR / "predictions_test.csv"
MANIFEST_JSON = FIGURES_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# Paramètres de modélisation
# ---------------------------------------------------------------------------
SEED = 42
TARGET = "blue_win"
TEST_SHARE = 0.2  # 20 % des parties les plus récentes pour le test final

# Variables : écarts bleu moins rouge (modèle du domaine), et disponibilité en direct.
# Chaque entrée : nom -> (description, disponible en direct, calcul à partir du dataset nettoyé)
FEATURES = {
    "ecart_eliminations": ("Éliminations de l'équipe bleue moins celles de l'équipe rouge", True,
                           lambda d: d["blueChampKills"] - d["redChampKills"]),
    "ecart_cs": ("Score de sbires (sbires de voie + monstres de jungle), bleu moins rouge", True,
                 lambda d: (d["blueMinionsKilled"] + d["blueJungleMinionsKilled"])
                 - (d["redMinionsKilled"] + d["redJungleMinionsKilled"])),
    "ecart_niveau": ("Niveau moyen de l'équipe bleue moins celui de l'équipe rouge", True,
                     lambda d: d["blueAvgLevel"] - d["redAvgLevel"]),
    "ecart_monstres_epiques": ("Monstres épiques (dragons et Hérauts) tués, bleu moins rouge", True,
                               lambda d: d["blueEpicMonsters"] - d["redEpicMonsters"]),
    "ecart_tours": ("Tours détruites par l'équipe bleue moins tours détruites par l'équipe rouge", True,
                    lambda d: d["redTowersLost"] - d["blueTowersLost"]),
    "ecart_or": ("Or total bleu moins rouge (non disponible en direct)", False,
                 lambda d: d["blueGold"] - d["redGold"]),
}
LIVE = [name for name, (_, live, _) in FEATURES.items() if live]  # variables utilisables en direct
FULL = list(FEATURES)                                              # toutes les variables
SHORT = {"ecart_eliminations": "Éliminations", "ecart_cs": "Score de sbires", "ecart_niveau": "Niveau",
         "ecart_monstres_epiques": "Monstres épiques", "ecart_tours": "Tours", "ecart_or": "Or (hors direct)",
         TARGET: "Victoire bleue"}


def load_dataset(path: Path = CLEAN_CSV):
    """Charge le dataset nettoyé et renvoie (df, X, y, masque apprentissage, masque test).

    Découpage temporel : les identifiants Riot croissent avec le temps, donc les 20 % de parties
    aux matchId les plus grands (les plus récentes) forment le test.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"{Path(path).name} introuvable : lancer d'abord « python clean_data.py ».")
    df = pd.read_csv(path).sort_values("matchId").reset_index(drop=True)
    X = pd.DataFrame({name: func(df) for name, (_, _, func) in FEATURES.items()})
    y = df[TARGET].astype(int)
    n_test = int(round(len(df) * TEST_SHARE))
    train = np.arange(len(df)) < len(df) - n_test
    return df, X, y, train, ~train


def relative(path: Path) -> str:
    """Chemin lisible relatif à la racine du dépôt (pour les messages et les documents)."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).name
