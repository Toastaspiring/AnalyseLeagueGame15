#!/usr/bin/env python3
"""Collecte des données d'entrée : jeu OpenML 43635 (parties Diamant de LoL à 15 minutes).

Le fichier brut est versionné dans le dépôt (data/raw/dataset_43635.arff). Ce script vérifie
qu'il est intact (somme MD5 publiée par OpenML) et ne le retélécharge que s'il manque, s'il est
abîmé, ou si on le demande avec --force. Il enregistre aussi les métadonnées OpenML (description,
date de collecte, licence) dans data/raw/openml_43635_metadata.json.

Usage :
    python collect_data.py            # vérifie, télécharge seulement si nécessaire
    python collect_data.py --force    # retélécharge depuis OpenML
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

from common import OPENML_ID, OPENML_MD5, RAW_ARFF, RAW_METADATA, relative

API_URL = f"https://www.openml.org/api/v1/json/data/{OPENML_ID}"
TIMEOUT = 60


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_metadata() -> dict:
    with urllib.request.urlopen(API_URL, timeout=TIMEOUT) as response:
        return json.load(response)["data_set_description"]


def download(url: str, target: Path) -> None:
    """Télécharge dans un fichier temporaire, puis remplace la cible (jamais de fichier à moitié écrit)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
            while chunk := response.read(1 << 20):
                tmp.write(chunk)
    tmp_path.replace(target)


def collect(force: bool = False) -> int:
    if RAW_ARFF.exists() and not force:
        if md5_of(RAW_ARFF) == OPENML_MD5:
            print(f"Données d'entrée présentes et intactes : {relative(RAW_ARFF)} (MD5 conforme à OpenML)")
            if not RAW_METADATA.exists():
                try:
                    RAW_METADATA.write_text(json.dumps(fetch_metadata(), indent=2, ensure_ascii=False), encoding="utf-8")
                    print(f"Métadonnées enregistrées : {relative(RAW_METADATA)}")
                except OSError as error:
                    print(f"Métadonnées non récupérées (OpenML injoignable : {error}). Sans incidence sur la suite.")
            return 0
        print(f"{relative(RAW_ARFF)} est présent mais sa somme MD5 ne correspond pas à celle d'OpenML.\n"
              "Cause fréquente : conversion des fins de ligne par Git sous Windows. Relancer avec --force\n"
              "pour retélécharger le fichier, ou recloner le dépôt (le fichier .gitattributes l'évite).")
        return 1

    print(f"Téléchargement du jeu OpenML {OPENML_ID}...")
    try:
        meta = fetch_metadata()
        download(meta["url"], RAW_ARFF)
    except OSError as error:
        print(f"Téléchargement impossible : {error}\n"
              "Le fichier brut est normalement fourni par le dépôt : vérifier le clone (git status).")
        return 1

    md5 = md5_of(RAW_ARFF)
    published = meta.get("md5_checksum")
    if md5 != published:
        print(f"Somme MD5 inattendue ({md5}, OpenML annonce {published}) : téléchargement corrompu, relancer.")
        return 1
    if published != OPENML_MD5:
        print("Attention : OpenML publie une nouvelle version du fichier. Mettre à jour OPENML_MD5 dans "
              "common.py et régénérer la datacard pour documenter ce nouveau batch.")
    RAW_METADATA.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Fichier enregistré : {relative(RAW_ARFF)} (MD5 {md5})")
    print(f"Métadonnées enregistrées : {relative(RAW_METADATA)}")
    return 0


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Collecte des données d'entrée (jeu OpenML 43635).")
    parser.add_argument("--force", action="store_true", help="Retélécharger le fichier même s'il est présent")
    args = parser.parse_args(argv)
    sys.exit(collect(force=args.force))


if __name__ == "__main__":
    main()
