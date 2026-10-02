#!/usr/bin/env python3
"""Exécute toute la chaîne du projet, de la collecte des données à la documentation.

Étapes, dans l'ordre :
    collect     collect_data.py         vérifie (ou télécharge) les données d'entrée
    clean       clean_data.py           nettoie les données, écrit LINEAGE.md
    train       train_models.py         entraîne, compare et exporte les modèles
    figures     make_figures.py         trace les graphes
    datacard    generate_datacard.py    génère DATACARD.md
    modelcard   generate_model_card.py  génère MODEL_CARD.md

Usage :
    python run_pipeline.py                      # toute la chaîne
    python run_pipeline.py --from train         # à partir d'une étape
    python run_pipeline.py --only figures modelcard
    python run_pipeline.py --list               # affiche les étapes
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("collect", "collect_data.py", "Collecte : vérification des données d'entrée"),
    ("clean", "clean_data.py", "Traitement : nettoyage des données et lineage"),
    ("train", "train_models.py", "Entraînement, sélection et export des modèles"),
    ("figures", "make_figures.py", "Génération des graphes"),
    ("datacard", "generate_datacard.py", "Documentation : datacard"),
    ("modelcard", "generate_model_card.py", "Documentation : model card"),
]
NAMES = [name for name, _, _ in STEPS]


def run(selected: list[str]) -> int:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")  # accents corrects sous Windows
    total = time.perf_counter()
    for i, (name, script, title) in enumerate([s for s in STEPS if s[0] in selected], start=1):
        print(f"\n=== {i}/{len(selected)} {title} ({script}) ===", flush=True)
        start = time.perf_counter()
        result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, env=env)
        if result.returncode != 0:
            print(f"\nÉchec de l'étape « {name} » ({script}, code {result.returncode}). "
                  f"Corriger puis reprendre avec : python run_pipeline.py --from {name}")
            return result.returncode
        print(f"--- étape « {name} » terminée en {time.perf_counter() - start:.0f} s", flush=True)
    print(f"\nChaîne terminée en {time.perf_counter() - total:.0f} s. "
          "Résultats : LINEAGE.md, DATACARD.md, MODEL_CARD.md, models/, reports/, figures/")
    return 0


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Exécute la chaîne complète du projet.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from", dest="start", choices=NAMES, help="Reprendre à partir de cette étape")
    group.add_argument("--only", nargs="+", choices=NAMES, metavar="ETAPE", help="N'exécuter que ces étapes")
    group.add_argument("--list", action="store_true", help="Afficher les étapes et quitter")
    args = parser.parse_args(argv)

    if args.list:
        for name, script, title in STEPS:
            print(f"{name:<10} {script:<24} {title}")
        return
    if args.only:
        selected = [n for n in NAMES if n in args.only]
    elif args.start:
        selected = NAMES[NAMES.index(args.start):]
    else:
        selected = NAMES
    sys.exit(run(selected))


if __name__ == "__main__":
    main()
