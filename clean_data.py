#!/usr/bin/env python3
"""Nettoyage sommaire du jeu OpenML 43635 : import -> traitement -> export.

Chaque étape applique une décision prise pendant l'exploration (notebook, section 5).
Le script mesure l'effet de chaque étape et écrit le lineage :

    data/clean/lol_diamond_15min_clean.csv   dataset nettoyé
    data/clean/lineage.json                  trace machine des étapes
    LINEAGE.md                               document récapitulatif

Usage :
    python clean_data.py
    python clean_data.py --input data/raw/autre_batch.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from common import CLEAN_CSV, OPENML_MD5, RAW_ARFF, ROOT
from generate_datacard import content_fingerprint  # même empreinte que dans la datacard

HERE = ROOT
DEFAULT_INPUT = RAW_ARFF
DEFAULT_OUTPUT = CLEAN_CSV
DEFAULT_LINEAGE_MD = ROOT / "LINEAGE.md"

RENAME = {
    "blueHeraldKills": "blueEpicMonsters",
    "redHeraldKills": "redEpicMonsters",
    "blueTowersDestroyed": "blueTowersLost",
    "redTowersDestroyed": "redTowersLost",
}
DECIMAL_COLUMNS = ["blueAvgLevel", "redAvgLevel"]
MIN_AVG_LEVEL = 7.0  # en dessous, au moins un joueur est absent ou déconnecté


# ---------------------------------------------------------------------------
# 1. Import
# ---------------------------------------------------------------------------

def _hash_file(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _looks_like_arff(path: Path) -> bool:
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                return line.startswith("%") or line.lower().startswith("@relation")
    return False


def read_raw(path: Path) -> pd.DataFrame:
    """Lit un batch brut : ARFF (format OpenML), CSV ou Parquet."""
    suffix = path.suffix.lower()
    if suffix == ".arff" or (suffix not in (".csv", ".parquet", ".pq") and _looks_like_arff(path)):
        from scipy.io import arff

        data, _meta = arff.loadarff(path)
        return pd.DataFrame(data)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Format non reconnu : {path.name}")


# ---------------------------------------------------------------------------
# 2. Traitement : une fonction par décision de nettoyage
# ---------------------------------------------------------------------------

@dataclass
class Step:
    name: str
    why: str
    func: Callable[[pd.DataFrame], tuple[pd.DataFrame, str]]


def drop_technical_index(df):
    cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    return df.drop(columns=cols), f"Colonnes supprimées : {', '.join(cols) or 'aucune'}"


def drop_duplicates(df):
    n_exact = int(df.duplicated().sum())
    df = df.drop_duplicates()
    n_conflicts = int(df["matchId"].duplicated().sum())  # même partie, valeurs différentes
    if n_conflicts:
        df = df.drop_duplicates(subset="matchId", keep="first")
    conflicts = "aucune partie en conflit" if n_conflicts == 0 else f"{n_conflicts} partie(s) en conflit"
    return df, f"{n_exact} doublons exacts retirés ; {conflicts} (même matchId, valeurs différentes)"


def drop_dragon_columns(df):
    cols = [c for c in ("blueDragonKills", "redDragonKills") if c in df.columns]
    constant = [c for c in cols if df[c].nunique(dropna=False) <= 1]
    kept = [c for c in cols if c not in constant]
    detail = f"Colonnes constantes supprimées : {', '.join(constant) or 'aucune'}"
    if kept:
        detail += f" ; non constantes dans ce batch, donc conservées : {', '.join(kept)}"
    return df.drop(columns=constant), detail


def rename_misleading_columns(df):
    mapping = {old: new for old, new in RENAME.items() if old in df.columns}
    return df.rename(columns=mapping), "; ".join(f"{old} -> {new}" for old, new in mapping.items())


def cast_types(df):
    df = df.copy()
    integer_cols = [c for c in df.columns if c not in DECIMAL_COLUMNS]
    df[integer_cols] = df[integer_cols].astype("int64")
    df[DECIMAL_COLUMNS] = df[DECIMAL_COLUMNS].astype("float64")
    return df, f"{len(integer_cols)} colonnes en int64 (signé), {len(DECIMAL_COLUMNS)} en float64"


def drop_abnormal_games(df):
    abnormal = (df["blueAvgLevel"] < MIN_AVG_LEVEL) | (df["redAvgLevel"] < MIN_AVG_LEVEL)
    return df[~abnormal], f"{int(abnormal.sum())} parties retirées (niveau moyen < {MIN_AVG_LEVEL:g} pour une équipe)"


def check_constraints(df):
    """Contrôles bloquants : le script s'arrête si une valeur impossible apparaît."""
    checks = {
        "aucune valeur manquante": df.notna().all().all(),
        "matchId unique": df["matchId"].is_unique,
        "cible égale à 0 ou 1": df["blue_win"].isin([0, 1]).all(),
        "or d'équipe >= 2 500": ((df["blueGold"] >= 2500) & (df["redGold"] >= 2500)).all(),
        "niveau moyen entre 1 et 18": df[DECIMAL_COLUMNS].stack().between(1, 18).all(),
        "tours perdues entre 0 et 11": df[["blueTowersLost", "redTowersLost"]].stack().between(0, 11).all(),
        "au plus 4 monstres épiques par partie": ((df["blueEpicMonsters"] + df["redEpicMonsters"]) <= 4).all(),
        "comptages positifs ou nuls": (df.drop(columns=DECIMAL_COLUMNS) >= 0).all().all(),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"Contrôles échoués : {', '.join(failed)}")
    return df, f"{len(checks)} contrôles passés : {', '.join(checks)}"


def sort_by_match(df):
    return df.sort_values("matchId").reset_index(drop=True), "Tri par matchId croissant"


STEPS = [
    Step("Supprimer l'index technique",
         "`Unnamed:_0` est un numéro de ligne exporté par pandas : il ne porte aucune information "
         "et empêche de repérer les doublons.", drop_technical_index),
    Step("Supprimer les doublons",
         "La même partie a été collectée deux fois (même matchId, mêmes valeurs) : la garder deux fois "
         "lui donnerait un poids double à l'entraînement.", drop_duplicates),
    Step("Supprimer les colonnes dragons",
         "Toujours égales à 0, ce qui est impossible en Diamant : colonnes cassées à la collecte.",
         drop_dragon_columns),
    Step("Renommer les colonnes trompeuses",
         "`HeraldKills` compte les monstres épiques (dragons et Hérauts) et `TowersDestroyed` les tours "
         "perdues par l'équipe : les noms doivent dire ce que mesurent les colonnes.",
         rename_misleading_columns),
    Step("Typer les colonnes",
         "L'ARFF stocke tout en flottant. Entiers signés pour pouvoir calculer des écarts entre équipes "
         "(avec un type non signé, 6 - 12 donne 250).", cast_types),
    Step("Retirer les parties anormales",
         "Un niveau moyen inférieur à 7 à 15 minutes signale un joueur absent : ces parties ne "
         "représentent pas le jeu normal que le modèle doit apprendre.", drop_abnormal_games),
    Step("Contrôler les contraintes du domaine",
         "Vérifie les règles du jeu établies pendant l'exploration ; le script s'arrête si un contrôle "
         "échoue, pour ne jamais exporter un dataset incohérent.", check_constraints),
    Step("Trier par matchId",
         "Les identifiants Riot croissent avec le temps : ce tri rend l'ordre chronologique explicite "
         "pour le découpage temporel de l'entraînement.", sort_by_match),
]


def run_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    log = []
    for i, step in enumerate(STEPS, start=1):
        rows_before, cols_before = len(df), list(df.columns)
        df, detail = step.func(df)
        log.append({
            "numero": i,
            "etape": step.name,
            "pourquoi": step.why,
            "fonction": step.func.__name__,
            "lignes_avant": rows_before,
            "lignes_apres": len(df),
            "colonnes_avant": len(cols_before),
            "colonnes_apres": df.shape[1],
            "colonnes_supprimees": [c for c in cols_before if c not in df.columns],
            "colonnes_ajoutees": [c for c in df.columns if c not in cols_before],
            "detail": detail,
        })
    return df, log


# ---------------------------------------------------------------------------
# 3. Export et lineage
# ---------------------------------------------------------------------------

def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(HERE).as_posix()
    except ValueError:
        return path.name


def _fmt(n) -> str:
    return f"{n:,}".replace(",", " ")


def _pct(share: float) -> str:
    """Part (0 à 1) au format français : 0,10 %."""
    return f"{share * 100:.2f} %".replace(".", ",")


def render_lineage(lineage: dict) -> str:
    src, out, steps = lineage["entree"], lineage["sortie"], lineage["etapes"]
    md5_txt = {True: "identique à la somme publiée par OpenML : fichier intègre",
               False: "différente de la somme OpenML : fichier modifié ou autre batch",
               None: "pas de somme de référence pour ce format"}[src["md5_conforme_openml"]]
    removed = src["lignes"] - out["lignes"]
    L = [
        "# Lineage : du jeu OpenML 43635 au dataset d'entraînement",
        "",
        f"> Document généré par `clean_data.py` le {datetime.fromisoformat(lineage['genere_le']):%d/%m/%Y à %H:%M}. "
        "Ne pas le modifier à la main : relancer le script.",
        "",
        "## 1. Vue d'ensemble",
        "",
        "```",
        "API Riot Games (NA1, Diamant I à IV, « Season 13 »)",
        "   |  extraction automatisée par Ben Fattori, collecte du 31/07/2020",
        "   v",
        "Kaggle : League of Legends Diamond Games (First 15 Minutes)",
        "   |  import sur OpenML le 24/03/2022 (identifiant 43635, version 1, ARFF)",
        "   v",
        f"{src['fichier']}   (brut, documenté par DATACARD.md)",
        "   |  clean_data.py : nettoyage en " + str(len(steps)) + " étapes (section 3)",
        "   v",
        f"{out['fichier']}   (nettoyé)",
        "   |  train_models.py : variables, entraînement, sélection, analyses",
        "   v",
        "models/, reports/   (modèles, métriques, prédictions sur le test)",
        "   |  make_figures.py",
        "   v",
        "figures/   (graphes)",
        "   |  generate_model_card.py",
        "   v",
        "MODEL_CARD.md",
        "```",
        "",
        "## 2. Entrée",
        "",
        "| Élément | Valeur |",
        "|---|---|",
        f"| Fichier | `{src['fichier']}` |",
        f"| Taille | {_fmt(src['octets'])} octets |",
        f"| MD5 | `{src['md5']}` ({md5_txt}) |",
        f"| SHA-256 (16 premiers caractères) | `{src['sha256'][:16]}` |",
        f"| Empreinte du contenu (même calcul que DATACARD.md) | `{src['empreinte_contenu'][:16]}` |",
        f"| Lignes x colonnes | {_fmt(src['lignes'])} x {src['colonnes']} |",
        "",
        "## 3. Étapes de nettoyage",
        "",
        "| # | Étape | Pourquoi | Lignes | Colonnes | Effet mesuré |",
        "|---|---|---|---|---|---|",
    ]
    for s in steps:
        rows = (f"{_fmt(s['lignes_avant'])} → {_fmt(s['lignes_apres'])}"
                if s["lignes_avant"] != s["lignes_apres"] else f"{_fmt(s['lignes_apres'])} (inchangé)")
        cols = (f"{s['colonnes_avant']} → {s['colonnes_apres']}"
                if s["colonnes_avant"] != s["colonnes_apres"] else f"{s['colonnes_apres']} (inchangé)")
        L.append(f"| {s['numero']} | {s['etape']} | {s['pourquoi']} | {rows} | {cols} | {s['detail']} |")
    L += [
        "",
        f"**Bilan** : {_fmt(src['lignes'])} → {_fmt(out['lignes'])} lignes ({_fmt(removed)} retirées, "
        f"soit {_pct(removed / src['lignes'])}), {src['colonnes']} → {out['colonnes']} colonnes. "
        "Les lignes retirées sont des doublons et des parties anormales ; aucune valeur n'a été imputée ni "
        "modifiée, seuls les noms et les types des colonnes ont changé.",
        "",
        "## 4. Sortie",
        "",
        "| Élément | Valeur |",
        "|---|---|",
        f"| Fichier | `{out['fichier']}` |",
        f"| SHA-256 (16 premiers caractères) | `{out['sha256'][:16]}` |",
        f"| Empreinte du contenu | `{out['empreinte_contenu'][:16]}` |",
        f"| Lignes x colonnes | {_fmt(out['lignes'])} x {out['colonnes']} |",
        f"| Part de victoires bleues | {_pct(out['part_victoires_bleues'])} |",
        "",
        "**Schéma du dataset nettoyé**",
        "",
        "| Colonne | Type | Origine |",
        "|---|---|---|",
    ]
    inverse = {new: old for old, new in RENAME.items()}
    for col, dtype in out["schema"].items():
        origin = f"renommée depuis `{inverse[col]}`" if col in inverse else "colonne d'origine"
        L.append(f"| `{col}` | `{dtype}` | {origin} |")
    L += [
        "",
        "## 5. Reproduire",
        "",
        "```bash",
        f"python clean_data.py --input {src['fichier']}",
        "```",
        "",
        "Le script est déterministe : relancé sur le même fichier, il produit le même dataset (même SHA-256).",
        "",
        "## 6. Versions",
        "",
        f"- Script : `clean_data.py` (SHA-256 `{lineage['script_sha256'][:16]}`)",
        f"- Python {lineage['versions']['python']}, pandas {lineage['versions']['pandas']}, "
        f"numpy {lineage['versions']['numpy']}",
        "",
    ]
    return "\n".join(L)


def clean(input_path: Path, output_path: Path, lineage_md: Path) -> dict:
    # 1. Import
    raw = read_raw(input_path)
    md5 = _hash_file(input_path, "md5")
    entree = {
        "fichier": _rel(input_path),
        "octets": input_path.stat().st_size,
        "md5": md5,
        "md5_conforme_openml": (md5 == OPENML_MD5) if input_path.suffix.lower() == ".arff" or _looks_like_arff(input_path) else None,
        "sha256": _hash_file(input_path, "sha256"),
        "empreinte_contenu": content_fingerprint(raw),
        "lignes": len(raw),
        "colonnes": raw.shape[1],
    }

    # 2. Traitement
    clean_df, steps = run_pipeline(raw)

    # 3. Export
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_path, index=False)
    lineage = {
        "genere_le": datetime.now().isoformat(timespec="seconds"),
        "script": "clean_data.py",
        "script_sha256": _hash_file(Path(__file__), "sha256"),
        "entree": entree,
        "etapes": steps,
        "sortie": {
            "fichier": _rel(output_path),
            "sha256": _hash_file(output_path, "sha256"),
            "empreinte_contenu": content_fingerprint(clean_df),
            "lignes": len(clean_df),
            "colonnes": clean_df.shape[1],
            "part_victoires_bleues": float(clean_df["blue_win"].mean()),
            "schema": {c: str(t) for c, t in clean_df.dtypes.items()},
        },
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    (output_path.parent / "lineage.json").write_text(json.dumps(lineage, indent=2, ensure_ascii=False), encoding="utf-8")
    lineage_md.write_text(render_lineage(lineage), encoding="utf-8")
    return lineage


def main(argv=None):
    parser = argparse.ArgumentParser(description="Nettoyage du dataset LoL Diamond 15 min (import -> traitement -> export).")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Batch brut (.arff, .csv ou .parquet)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Dataset nettoyé (.csv)")
    parser.add_argument("--lineage", default=str(DEFAULT_LINEAGE_MD), help="Document de lineage (.md)")
    args = parser.parse_args(argv)

    lineage = clean(Path(args.input), Path(args.output), Path(args.lineage))
    for s in lineage["etapes"]:
        print(f"{s['numero']}. {s['etape']:<38} {s['lignes_avant']:>6} -> {s['lignes_apres']:<6} lignes | {s['detail']}")
    out = lineage["sortie"]
    print(f"\nExport : {out['fichier']} ({out['lignes']} lignes x {out['colonnes']} colonnes)")
    print(f"Lineage : {Path(args.lineage).name} et {Path(out['fichier']).parent.as_posix()}/lineage.json")


if __name__ == "__main__":
    main()
