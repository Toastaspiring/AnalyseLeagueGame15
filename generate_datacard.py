#!/usr/bin/env python3
"""Génération automatique de la partie « données » de la datacard.

La datacard (DATACARD.md) assemble deux parties :

* une entête rédigée à la main (datacard_header.md) : contexte, provenance,
  collecte, modèle du domaine, usages, limites ;
* une partie calculée par ce script à partir du batch de données : empreinte,
  dictionnaire des données, répartition des classes, qualité, règles métier et
  évolution par rapport au batch précédent.

Toute la connaissance métier (descriptions des colonnes, rôles, règles de
cohérence) est déclarée dans datacard_config.json. Le script reste générique :
il suffit de le relancer à chaque nouveau batch pour mettre la datacard à jour.

Exemples
--------
    python generate_datacard.py
    python generate_datacard.py --source data/raw/batch_2026_11.csv
    python generate_datacard.py --source openml:43635
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NBSP = "\u00a0"  # espace insécable : séparateur de milliers à la française

AUTO_BEGIN = (
    "<!-- DÉBUT DE LA PARTIE GÉNÉRÉE AUTOMATIQUEMENT (generate_datacard.py) : "
    "ne pas modifier à la main -->"
)
AUTO_END = "<!-- FIN DE LA PARTIE GÉNÉRÉE AUTOMATIQUEMENT -->"

DEFAULT_SETTINGS = {
    "section_number": 8,       # numéro de la section « Données » dans la datacard
    "z_threshold": 3.0,        # seuil |z| au-delà duquel une valeur est jugée aberrante
    "max_discrete_unique": 12,  # nb max de valeurs distinctes pour détailler une répartition
    "smd_threshold": 0.1,      # décalage de moyenne (en écarts-types) signalé entre deux batchs
}

ROLE_TARGET = "cible"
ROLE_ID = "identifiant"
ROLE_TECH = "technique"
ROLE_FEATURE = "variable"
NO_STATS_ROLES = (ROLE_ID, ROLE_TECH)


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    """Lit la configuration (descriptions, rôles, règles métier)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def settings_from(config: dict) -> dict:
    """Paramètres de calcul : valeurs par défaut surchargées par la configuration."""
    return {**DEFAULT_SETTINGS, **config.get("datacard", {})}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data(source: str | Path, cache_dir: str | Path = HERE / "data" / "openml_cache"):
    """Charge un batch depuis OpenML ('openml:<id>') ou un fichier .csv/.parquet/.arff.

    Retourne (DataFrame, informations sur la source).
    """
    source = str(source)
    info: dict[str, Any] = {"source": source}

    if source.startswith("openml:"):
        from sklearn.datasets import fetch_openml  # import local : utile seulement ici

        data_id = int(source.split(":", 1)[1])
        bunch = fetch_openml(
            data_id=data_id, as_frame=True, parser="auto",
            target_column=None, data_home=str(cache_dir),
        )
        details = bunch.details or {}
        info.update(
            type="openml",
            openml_id=data_id,
            openml_name=details.get("name"),
            openml_version=details.get("version"),
            openml_upload_date=details.get("upload_date"),
            openml_md5=details.get("md5_checksum"),
        )
        return bunch.frame.copy(), info

    path = Path(source)
    if not path.exists() and not path.is_absolute() and (HERE / path).exists():
        path = HERE / path  # chemin relatif à la racine du dépôt
    if not path.exists():
        raise FileNotFoundError(f"Source introuvable : {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif suffix == ".arff":
        from scipy.io import arff

        data, _meta = arff.loadarff(path)
        df = pd.DataFrame(data)
        for col in df.columns:  # les attributs nominaux ARFF arrivent en bytes
            if df[col].map(lambda v: isinstance(v, bytes)).any():
                df[col] = df[col].map(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v)
    else:
        raise ValueError(f"Format non pris en charge : {suffix} (attendu : .csv, .parquet, .arff)")

    info.update(type="fichier", file_name=path.name, file_sha256=_sha256_file(path))
    return df, info


# ---------------------------------------------------------------------------
# Calculs
# ---------------------------------------------------------------------------

def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _is_integer_valued(s: pd.Series) -> bool:
    x = s.dropna()
    return _is_numeric(s) and len(x) > 0 and bool((x.astype("float64") % 1 == 0).all())


def content_fingerprint(df: pd.DataFrame) -> str:
    """Empreinte SHA-256 du contenu, indépendante du format de stockage.

    Les colonnes numériques sont converties en float64 avant hachage : un même
    batch lu depuis un CSV (entiers) ou depuis OpenML (flottants) a la même empreinte.
    """
    norm = pd.DataFrame({
        col: (df[col].astype("float64") if _is_numeric(df[col]) or pd.api.types.is_bool_dtype(df[col])
              else df[col].astype(str))
        for col in df.columns
    })
    h = hashlib.sha256()
    h.update("|".join(map(str, df.columns)).encode("utf-8"))
    h.update(pd.util.hash_pandas_object(norm, index=False).to_numpy().tobytes())
    return h.hexdigest()


def column_roles(df: pd.DataFrame, config: dict) -> dict[str, str]:
    """Rôle de chaque colonne : cible, identifiant, technique ou variable."""
    target = config.get("target")
    ids = set(config.get("id_columns", []))
    tech = set(config.get("technical_columns", []))
    roles = {}
    for col in df.columns:
        if col == target:
            roles[col] = ROLE_TARGET
        elif col in ids:
            roles[col] = ROLE_ID
        elif col in tech:
            roles[col] = ROLE_TECH
        else:
            roles[col] = ROLE_FEATURE
    return roles


def infer_nature(s: pd.Series, role: str) -> str:
    """Nature statistique déduite des valeurs (et du rôle déclaré)."""
    if role in NO_STATS_ROLES:
        return role
    n_unique = s.nunique(dropna=True)
    if n_unique <= 1:
        return "constante"
    if n_unique == 2:
        return "binaire"
    if not _is_numeric(s):
        return "catégorielle" if n_unique <= 50 else "texte"
    return "quantitative discrète" if _is_integer_valued(s) else "quantitative continue"


def _outlier_eligible(s: pd.Series, role: str, nature: str) -> bool:
    return _is_numeric(s) and role not in NO_STATS_ROLES and nature not in ("binaire", "constante")


def zscore_mask(x: pd.Series, threshold: float) -> pd.Series:
    """Booléen : |z| > seuil, avec z = (x - moyenne) / écart-type."""
    x = x.astype("float64")
    std = x.std()
    if not std or math.isnan(std):
        return pd.Series(False, index=x.index)
    return ((x - x.mean()).abs() / std) > threshold


def iqr_mask(x: pd.Series, k: float = 1.5) -> pd.Series:
    """Booléen : valeur hors de [Q1 - k*IQR ; Q3 + k*IQR] (méthode de Tukey)."""
    x = x.astype("float64")
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    return (x < q1 - k * iqr) | (x > q3 + k * iqr)


def build_data_dictionary(df: pd.DataFrame, config: dict, z_threshold: float | None = None) -> pd.DataFrame:
    """Dictionnaire des données : une ligne par colonne, avec description et statistiques."""
    settings = settings_from(config)
    z_threshold = settings["z_threshold"] if z_threshold is None else z_threshold
    docs = config.get("columns", {})
    roles = column_roles(df, config)
    n = len(df)
    rows = []
    for col in df.columns:
        s = df[col]
        role = roles[col]
        nature = infer_nature(s, role)
        doc = docs.get(col, {})
        row: dict[str, Any] = {
            "colonne": col,
            "description": doc.get("description", "(description manquante : compléter datacard_config.json)"),
            "role": role,
            "type_stocke": str(s.dtype),
            "nature": nature,
            "unite": doc.get("unit", ""),
            "n_valeurs_distinctes": int(s.nunique(dropna=True)),
            "n_manquants": int(s.isna().sum()),
            "pct_manquants": float(s.isna().mean() * 100) if n else 0.0,
        }
        stats = dict.fromkeys(["min", "max", "moyenne", "mediane", "q1", "q3", "ecart_type"], np.nan)
        x = s.dropna()
        if _is_numeric(s) and len(x):
            x = x.astype("float64")
            stats["min"], stats["max"] = x.min(), x.max()
            if role not in NO_STATS_ROLES:
                stats.update(
                    moyenne=x.mean(), mediane=x.median(), q1=x.quantile(0.25),
                    q3=x.quantile(0.75), ecart_type=x.std(),
                )
        row.update(stats)
        if _outlier_eligible(s, role, nature) and len(x):
            n_z = int(zscore_mask(x, z_threshold).sum())
            row.update(n_z_eleve=n_z, pct_z_eleve=n_z / n * 100, n_hors_iqr=int(iqr_mask(x).sum()))
        else:
            row.update(n_z_eleve=np.nan, pct_z_eleve=np.nan, n_hors_iqr=np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def class_distribution(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Effectif et proportion de chaque classe de la cible."""
    target = config.get("target")
    if not target or target not in df.columns:
        return pd.DataFrame(columns=["classe", "signification", "effectif", "proportion"])
    labels = config.get("target_labels", {})
    counts = df[target].value_counts(dropna=False).sort_index()
    rows = []
    for value, count in counts.items():
        if pd.isna(value):
            key, label = "manquant", "valeur manquante"
        else:
            key = str(int(value)) if isinstance(value, (int, float, np.number)) and float(value).is_integer() else str(value)
            label = labels.get(key, "")
        rows.append({"classe": key, "signification": label, "effectif": int(count), "proportion": count / len(df)})
    return pd.DataFrame(rows)


def discrete_distributions(df: pd.DataFrame, config: dict, max_unique: int | None = None) -> dict[str, pd.Series]:
    """Répartition des valeurs des variables à faible cardinalité (hors cible)."""
    max_unique = settings_from(config)["max_discrete_unique"] if max_unique is None else max_unique
    roles = column_roles(df, config)
    out = {}
    for col in df.columns:
        s = df[col]
        nature = infer_nature(s, roles[col])
        if roles[col] == ROLE_FEATURE and nature != "constante" and s.nunique(dropna=True) <= max_unique:
            out[col] = s.value_counts(normalize=True, dropna=False).sort_index()
    return out


def _looks_like_index(s: pd.Series) -> bool:
    """Vrai si la colonne vaut 0..n-1 (ou 1..n) : index exporté avec les données."""
    if not _is_integer_valued(s) or s.isna().any() or len(s) < 2:
        return False
    values = s.to_numpy(dtype="float64")
    start = values[0]
    return start in (0.0, 1.0) and np.array_equal(values, np.arange(len(s)) + start)


def quality_report(df: pd.DataFrame, config: dict, z_threshold: float | None = None) -> dict:
    """Indicateurs de qualité : manquants, doublons, colonnes suspectes, schéma, aberrants."""
    settings = settings_from(config)
    z_threshold = settings["z_threshold"] if z_threshold is None else z_threshold
    roles = column_roles(df, config)
    tech = [c for c, r in roles.items() if r == ROLE_TECH]
    ids = [c for c, r in roles.items() if r == ROLE_ID]
    documented = config.get("columns", {})

    eligible = [c for c in df.columns if _outlier_eligible(df[c], roles[c], infer_nature(df[c], roles[c]))]
    if eligible:
        z_rows = pd.concat([zscore_mask(df[c].dropna(), z_threshold) for c in eligible], axis=1)
        rows_with_z = int(z_rows.reindex(df.index).fillna(False).any(axis=1).sum())
    else:
        rows_with_z = 0

    missing = df.isna().sum()
    return {
        "n_lignes": int(len(df)),
        "n_colonnes": int(df.shape[1]),
        "memoire_mo": float(df.memory_usage(deep=True).sum() / 1e6),
        "manquants_total": int(missing.sum()),
        "manquants_par_colonne": {c: int(v) for c, v in missing.items() if v > 0},
        "lignes_vides": int(df.isna().all(axis=1).sum()),
        "colonnes_techniques": tech,
        "doublons_complets": int(df.duplicated().sum()),
        "doublons_hors_techniques": int(df.drop(columns=tech).duplicated().sum()) if tech else int(df.duplicated().sum()),
        "doublons_identifiants": {c: int(df[c].dropna().duplicated().sum()) for c in ids},
        "colonnes_constantes": [c for c in df.columns if df[c].nunique(dropna=False) <= 1],
        "colonnes_type_index": [c for c in df.columns if _looks_like_index(df[c])],
        "entiers_stockes_en_flottant": [
            c for c in df.columns if pd.api.types.is_float_dtype(df[c]) and _is_integer_valued(df[c])
        ],
        "colonnes_non_documentees": [c for c in df.columns if c not in documented],
        "colonnes_documentees_absentes": [c for c in documented if c not in df.columns],
        "colonnes_aberrants_evaluees": eligible,
        "lignes_avec_z_eleve": rows_with_z,
    }


def evaluate_rules(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Évalue les règles métier déclarées (expression pandas décrivant une ligne VALIDE).

    Une valeur manquante dans une colonne utilisée par la règle compte comme une violation.
    Une règle qui cite une colonne absente du batch est marquée « non évaluable ».
    """
    records = []
    for rule in config.get("rules", []):
        rec = {
            "regle": rule["name"],
            "severite": rule.get("severity", "alerte"),
            "expression": rule["expr"],
            "justification": rule.get("why", ""),
        }
        try:
            valid = pd.Series(df.eval(rule["expr"], engine="python"), index=df.index)
            valid = valid.astype("boolean").fillna(False).astype(bool)
            n_viol = int((~valid).sum())
            rec.update(
                violations=n_viol,
                pct_violations=n_viol / len(df) * 100 if len(df) else 0.0,
                statut="conforme" if n_viol == 0 else "violée",
            )
        except Exception as exc:  # colonne absente, expression invalide...
            rec.update(violations=np.nan, pct_violations=np.nan, statut=f"non évaluable ({type(exc).__name__})")
        records.append(rec)
    return pd.DataFrame(records, columns=[
        "regle", "severite", "violations", "pct_violations", "statut", "expression", "justification",
    ])


# ---------------------------------------------------------------------------
# Comparaison entre batchs
# ---------------------------------------------------------------------------

def _clean_json(value):
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) else float(value)
    return value


def build_snapshot(df, config, dictionary, quality, source_info, fingerprint, generated_at) -> dict:
    """Résumé chiffré du batch, enregistré pour la comparaison avec le batch suivant."""
    target = config.get("target")
    snapshot = {
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "source": source_info.get("source"),
        "fingerprint": fingerprint,
        "n_lignes": quality["n_lignes"],
        "n_colonnes": quality["n_colonnes"],
        "colonnes": {
            row.colonne: {
                "dtype": row.type_stocke,
                "moyenne": row.moyenne,
                "ecart_type": row.ecart_type,
                "pct_manquants": row.pct_manquants,
            }
            for row in dictionary.itertuples(index=False)
        },
        "taux_cible_positive": float(df[target].mean()) if target in df.columns and _is_numeric(df[target]) else None,
    }
    return _clean_json(snapshot)


def compare_snapshots(previous: dict | None, current: dict, smd_threshold: float) -> dict:
    """Différences entre le batch précédent et le batch courant."""
    if previous is None:
        return {"premiere_generation": True}
    if previous.get("fingerprint") == current["fingerprint"]:
        return {"identique": True, "precedent": previous}

    prev_cols, cur_cols = previous.get("colonnes", {}), current["colonnes"]
    shifts, missing_changes = [], []
    for col in cur_cols:
        if col not in prev_cols:
            continue
        p, c = prev_cols[col], cur_cols[col]
        if p.get("moyenne") is not None and c.get("moyenne") is not None and p.get("ecart_type"):
            smd = abs(c["moyenne"] - p["moyenne"]) / p["ecart_type"]
            if smd >= smd_threshold:
                shifts.append((col, p["moyenne"], c["moyenne"], smd))
        if p.get("pct_manquants") is not None and c.get("pct_manquants") is not None:
            if abs(c["pct_manquants"] - p["pct_manquants"]) >= 1.0:
                missing_changes.append((col, p["pct_manquants"], c["pct_manquants"]))
    return {
        "precedent": previous,
        "lignes": (previous.get("n_lignes"), current["n_lignes"]),
        "colonnes_ajoutees": [c for c in cur_cols if c not in prev_cols],
        "colonnes_supprimees": [c for c in prev_cols if c not in cur_cols],
        "types_modifies": [
            (c, prev_cols[c].get("dtype"), cur_cols[c].get("dtype"))
            for c in cur_cols if c in prev_cols and prev_cols[c].get("dtype") != cur_cols[c].get("dtype")
        ],
        "decalages_moyenne": sorted(shifts, key=lambda t: -t[3]),
        "manquants_modifies": missing_changes,
        "taux_cible": (previous.get("taux_cible_positive"), current.get("taux_cible_positive")),
    }


# ---------------------------------------------------------------------------
# Rendu Markdown
# ---------------------------------------------------------------------------

def _isnan(x) -> bool:
    return x is None or (isinstance(x, (float, np.floating)) and math.isnan(x))


def fmt_number(x, decimals: int = 2) -> str:
    """Nombre au format français : 26 349,16 (espace insécable pour les milliers)."""
    if _isnan(x):
        return "n/a"
    x = float(x)
    text = f"{int(x):,}" if x.is_integer() and abs(x) < 1e15 else f"{x:,.{decimals}f}"
    return text.replace(",", NBSP).replace(".", ",")


def fmt_pct(x, decimals: int = 1) -> str:
    if _isnan(x):
        return "n/a"
    if 0 < x < 10 ** -decimals:
        return f"< {10 ** -decimals:.{decimals}f}{NBSP}%".replace(".", ",")
    return f"{x:.{decimals}f}{NBSP}%".replace(".", ",")


def fmt_count_pct(count, pct) -> str:
    if _isnan(count):
        return "n/a"
    return fmt_number(count) if count == 0 else f"{fmt_number(count)} ({fmt_pct(pct)})"


def _fmt_param(x) -> str:
    """Paramètre sans zéros inutiles : 3 -> « 3 », 0.1 -> « 0,1 »."""
    return f"{x:g}".replace(".", ",")


def _code_list(cols) -> str:
    return ", ".join(f"`{c}`" for c in cols) if cols else "aucune"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(v) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _fmt_stat(value, role) -> str:
    if role == ROLE_ID and not _isnan(value):
        return str(int(value))  # pas de séparateur de milliers pour un identifiant
    return fmt_number(value)


def render_generated_part(*, config, settings, source_info, fingerprint, generated_at,
                          dictionary, classes, discrete, quality, rules, comparison,
                          snapshot_name) -> str:
    sec = settings["section_number"]
    z = settings["z_threshold"]
    target = config.get("target")
    L: list[str] = [AUTO_BEGIN, "", f"## {sec}. Données (partie générée automatiquement)", ""]
    L += [
        f"> Section produite par `generate_datacard.py` le {generated_at:%d/%m/%Y à %H:%M} "
        f"à partir de `{source_info.get('source')}`.",
        "> Pour la faire évoluer : compléter `datacard_config.json` (descriptions, règles) "
        "ou fournir un nouveau batch, puis relancer le script. Ne pas l'éditer à la main.",
        "",
    ]

    # Empreinte du batch -----------------------------------------------------
    L += [f"### {sec}.1 Empreinte du batch", ""]
    src = f"`{source_info.get('source')}`"
    if source_info.get("type") == "openml":
        src += f" (OpenML, version {source_info.get('openml_version')})"
    rows = [["Source", src]]
    if source_info.get("type") == "fichier":
        rows.append(["Fichier (SHA-256, 16 premiers caractères)",
                     f"`{source_info['file_name']}` (`{source_info['file_sha256'][:16]}`)"])
    rows += [
        ["Lignes", fmt_number(quality["n_lignes"])],
        ["Colonnes", fmt_number(quality["n_colonnes"])],
        ["Mémoire en RAM", f"{fmt_number(quality['memoire_mo'], 1)}{NBSP}Mo"],
        ["Empreinte du contenu (SHA-256, 16 premiers caractères)", f"`{fingerprint[:16]}`"],
    ]
    L += [md_table(["Élément", "Valeur"], rows), ""]

    # Dictionnaire des données ---------------------------------------------
    L += [f"### {sec}.2 Dictionnaire des données", "", "**Description des colonnes**", ""]
    L += [md_table(
        ["Colonne", "Description", "Rôle", "Type stocké", "Nature", "Unité", "Valeurs distinctes"],
        [[f"`{r.colonne}`", r.description, r.role, f"`{r.type_stocke}`", r.nature, r.unite or "",
          fmt_number(r.n_valeurs_distinctes)] for r in dictionary.itertuples(index=False)],
    ), ""]
    L += ["**Statistiques descriptives**", ""]
    stat_rows = []
    for r in dictionary.itertuples(index=False):
        interval = "n/a" if _isnan(r.min) else f"[{_fmt_stat(r.min, r.role)} ; {_fmt_stat(r.max, r.role)}]"
        stat_rows.append([
            f"`{r.colonne}`", interval, fmt_number(r.moyenne), fmt_number(r.mediane),
            fmt_number(r.q1), fmt_number(r.q3), fmt_number(r.ecart_type),
            fmt_count_pct(r.n_manquants, r.pct_manquants),
            fmt_count_pct(r.n_z_eleve, r.pct_z_eleve), fmt_number(r.n_hors_iqr),
        ])
    L += [md_table(
        ["Colonne", "Intervalle [min ; max]", "Moyenne", "Médiane", "Q1", "Q3", "Écart-type",
         "Manquants", f"Aberrantes (z-score > {fmt_number(z, 1)})", "Hors IQR"],
        stat_rows,
    ), ""]
    L += [
        "Lecture : Q1 et Q3 sont les 1er et 3e quartiles (25 % et 75 % des valeurs sont en dessous). "
        f"« Aberrantes (z-score) » compte les valeurs dont le z-score dépasse {fmt_number(z, 1)} en valeur absolue, "
        "avec z = (x - moyenne) / écart-type. « Hors IQR » compte les valeurs en dehors de "
        "[Q1 - 1,5 × IQR ; Q3 + 1,5 × IQR] (méthode de Tukey, plus robuste aux distributions asymétriques). "
        "Les statistiques ne sont pas calculées pour les identifiants et les colonnes techniques, "
        "ni les aberrants pour les variables binaires ou constantes (n/a).",
        "",
    ]

    # Répartition des classes ----------------------------------------------
    L += [f"### {sec}.3 Répartition des classes (cible `{target}`)", ""]
    if classes.empty:
        L += ["Aucune cible déclarée dans la configuration ou cible absente du batch.", ""]
    else:
        L += [md_table(
            ["Classe", "Signification", "Effectif", "Proportion"],
            [[r.classe, r.signification, fmt_number(r.effectif), fmt_pct(r.proportion * 100)]
             for r in classes.itertuples(index=False)],
        ), ""]
        minority = classes["proportion"].min()
        ratio = classes["effectif"].max() / max(classes["effectif"].min(), 1)
        if minority >= 0.4:
            verdict = "Classes équilibrées : l'exactitude (accuracy) reste une métrique lisible."
        elif minority >= 0.2:
            verdict = "Déséquilibre modéré : stratifier les découpages et suivre F1 ou AUC en plus de l'exactitude."
        else:
            verdict = "Fort déséquilibre : privilégier AUC-PR, F1 et un rééquilibrage (pondération, rééchantillonnage)."
        L += [f"Rapport classe majoritaire / classe minoritaire : {fmt_number(ratio)}. {verdict}", ""]

    # Variables discrètes ----------------------------------------------------
    L += [f"### {sec}.4 Répartition des variables discrètes", ""]
    if discrete:
        L += [md_table(
            ["Colonne", "Répartition (valeur : part des lignes)"],
            [[f"`{col}`", " · ".join(f"{fmt_number(v)} : {fmt_pct(p * 100)}" for v, p in dist.items())]
             for col, dist in discrete.items()],
        ), ""]
    else:
        L += [f"Aucune variable à {settings['max_discrete_unique']} valeurs distinctes ou moins.", ""]

    # Qualité --------------------------------------------------------------
    q = quality
    L += [f"### {sec}.5 Qualité des données", ""]
    if q["manquants_total"]:
        detail = ", ".join(f"`{c}` : {fmt_number(v)}" for c, v in q["manquants_par_colonne"].items())
        missing_txt = f"{fmt_number(q['manquants_total'])} ({detail})"
    else:
        missing_txt = "aucune"
    id_dups = ", ".join(f"`{c}` : {fmt_number(v)}" for c, v in q["doublons_identifiants"].items()) or "aucun identifiant déclaré"
    n = q["n_lignes"]
    L += [
        f"- **Valeurs manquantes** : {missing_txt}",
        f"- **Lignes entièrement vides** : {fmt_number(q['lignes_vides'])}",
        f"- **Doublons (toutes colonnes)** : {fmt_number(q['doublons_complets'])}",
        f"- **Doublons en ignorant les colonnes techniques ({_code_list(q['colonnes_techniques'])})** : "
        f"{fmt_number(q['doublons_hors_techniques'])}",
        f"- **Identifiants en double** : {id_dups}",
        f"- **Colonnes constantes** (aucune information) : {_code_list(q['colonnes_constantes'])}",
        f"- **Colonnes ressemblant à un index de ligne** : {_code_list(q['colonnes_type_index'])}",
        f"- **Entiers stockés en flottant** ({len(q['entiers_stockes_en_flottant'])} colonnes) : "
        f"{_code_list(q['entiers_stockes_en_flottant'])}",
        f"- **Colonnes absentes de la configuration** (à documenter) : {_code_list(q['colonnes_non_documentees'])}",
        f"- **Colonnes documentées mais absentes du batch** : {_code_list(q['colonnes_documentees_absentes'])}",
        f"- **Lignes avec au moins une valeur |z| > {fmt_number(z, 1)}** : "
        f"{fmt_count_pct(q['lignes_avec_z_eleve'], q['lignes_avec_z_eleve'] / n * 100 if n else 0)}, "
        f"calculé sur {len(q['colonnes_aberrants_evaluees'])} variables quantitatives",
        "",
        "Le seuil |z| > 3 isole moins de 0,3 % des valeurs d'une loi normale. Sur des comptages très "
        "asymétriques, il signale aussi des valeurs rares mais légitimes : un z-score élevé est une valeur "
        "à examiner, pas une erreur à supprimer automatiquement.",
        "",
    ]

    # Règles métier --------------------------------------------------------
    L += [f"### {sec}.6 Règles métier", ""]
    if rules.empty:
        L += ["Aucune règle déclarée dans `datacard_config.json`.", ""]
    else:
        L += ["Règles déclarées dans `datacard_config.json` (expression décrivant une ligne valide), "
              "évaluées sur l'ensemble du batch.", ""]
        L += [md_table(
            ["Règle", "Sévérité", "Violations", "Part des lignes", "Statut", "Justification"],
            [[r.regle, r.severite, fmt_number(r.violations), fmt_pct(r.pct_violations), r.statut, r.justification]
             for r in rules.itertuples(index=False)],
        ), ""]

    # Évolution ------------------------------------------------------------
    L += [f"### {sec}.7 Évolution par rapport au batch précédent", ""]
    c = comparison
    if c.get("premiere_generation"):
        L += [f"Première génération : aucun batch de référence enregistré (`{snapshot_name}`). "
              "La prochaine exécution sera comparée à celle-ci.", ""]
    elif c.get("identique"):
        prev = c["precedent"]
        L += [f"Contenu identique au batch précédent (même empreinte), généré le "
              f"{_fmt_iso(prev.get('generated_at'))} à partir de `{prev.get('source')}`.", ""]
    else:
        prev = c["precedent"]
        old_n, new_n = c["lignes"]
        delta = new_n - (old_n or 0)
        sign = "+" if delta >= 0 else ""
        L += [f"Batch de référence : généré le {_fmt_iso(prev.get('generated_at'))} à partir de `{prev.get('source')}`.", ""]
        L += [f"- **Lignes** : {fmt_number(old_n)} → {fmt_number(new_n)} ({sign}{fmt_number(delta)})"]
        L += [f"- **Colonnes ajoutées** : {_code_list(c['colonnes_ajoutees'])}"]
        L += [f"- **Colonnes supprimées** : {_code_list(c['colonnes_supprimees'])}"]
        if c["types_modifies"]:
            L += ["- **Types modifiés** : " + ", ".join(f"`{col}` ({a} → {b})" for col, a, b in c["types_modifies"])]
        old_rate, new_rate = c["taux_cible"]
        if old_rate is not None and new_rate is not None:
            L += [f"- **Part de la classe positive de `{target}`** : {fmt_pct(old_rate * 100)} → {fmt_pct(new_rate * 100)}"]
        if c["decalages_moyenne"]:
            L += [f"- **Moyennes décalées d'au moins {_fmt_param(settings['smd_threshold'])} écart-type** :"]
            L += [f"  - `{col}` : {fmt_number(a)} → {fmt_number(b)} (écart standardisé {fmt_number(smd)})"
                  for col, a, b, smd in c["decalages_moyenne"]]
        else:
            L += [f"- **Moyennes** : aucun décalage d'au moins {_fmt_param(settings['smd_threshold'])} écart-type"]
        if c["manquants_modifies"]:
            L += ["- **Valeurs manquantes** : " + ", ".join(
                f"`{col}` ({fmt_pct(a)} → {fmt_pct(b)})" for col, a, b in c["manquants_modifies"])]
        L += [""]

    L += [AUTO_END, ""]
    return "\n".join(L)


def _fmt_iso(value) -> str:
    try:
        return datetime.fromisoformat(value).strftime("%d/%m/%Y à %H:%M")
    except (TypeError, ValueError):
        return str(value)


def assemble_datacard(header_text: str, generated_text: str) -> str:
    """Entête manuelle + partie générée. L'entête reste la source de vérité de la partie manuelle."""
    header = header_text.split(AUTO_BEGIN)[0].rstrip()  # au cas où une ancienne partie générée y aurait été collée
    return f"{header}\n\n{generated_text}"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_datacard(
    source: str | Path | None = None,
    *,
    df: pd.DataFrame | None = None,
    source_label: str | None = None,
    config_path: str | Path = HERE / "datacard_config.json",
    header_path: str | Path = HERE / "datacard_header.md",
    output_path: str | Path = HERE / "DATACARD.md",
    dictionary_path: str | Path = HERE / "outputs" / "data_dictionary.csv",
    snapshot_path: str | Path = HERE / "outputs" / "datacard_stats.json",
    cache_dir: str | Path = HERE / "data" / "openml_cache",
    z_threshold: float | None = None,
    compare: bool = True,
) -> dict:
    """Calcule la partie données, l'assemble avec l'entête et écrit les fichiers.

    On peut fournir soit une source (fichier ou 'openml:<id>'), soit directement un DataFrame.
    """
    config = load_config(config_path)
    settings = settings_from(config)
    if z_threshold is not None:
        settings["z_threshold"] = z_threshold

    if df is None:
        source = source or config.get("source", {}).get("default")
        if not source:
            raise ValueError("Aucune source : passer --source ou renseigner source.default dans la configuration.")
        df, source_info = load_data(source, cache_dir=cache_dir)
    else:
        source_info = {"source": source_label or "DataFrame fourni en mémoire", "type": "dataframe"}

    config = {**config, "datacard": settings}
    generated_at = datetime.now()
    fingerprint = content_fingerprint(df)
    dictionary = build_data_dictionary(df, config)
    classes = class_distribution(df, config)
    discrete = discrete_distributions(df, config)
    quality = quality_report(df, config)
    rules = evaluate_rules(df, config)

    snapshot_path = Path(snapshot_path)
    snapshot = build_snapshot(df, config, dictionary, quality, source_info, fingerprint, generated_at)
    previous = None
    if compare and snapshot_path.exists():
        previous = json.loads(snapshot_path.read_text(encoding="utf-8"))
    comparison = compare_snapshots(previous, snapshot, settings["smd_threshold"])

    generated = render_generated_part(
        config=config, settings=settings, source_info=source_info, fingerprint=fingerprint,
        generated_at=generated_at, dictionary=dictionary, classes=classes, discrete=discrete,
        quality=quality, rules=rules, comparison=comparison, snapshot_name=snapshot_path.name,
    )
    header_text = Path(header_path).read_text(encoding="utf-8")
    datacard = assemble_datacard(header_text, generated)

    for path in (output_path, dictionary_path, snapshot_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(datacard, encoding="utf-8")
    dictionary.to_csv(dictionary_path, index=False, encoding="utf-8")
    snapshot_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "output_path": str(output_path),
        "dictionary_path": str(dictionary_path),
        "snapshot_path": str(snapshot_path),
        "n_lignes": quality["n_lignes"],
        "n_colonnes": quality["n_colonnes"],
        "fingerprint": fingerprint,
        "comparison": comparison,
        "generated_markdown": generated,
        "dictionary": dictionary,
        "rules": rules,
        "quality": quality,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Génère la partie données de la datacard (DATACARD.md).")
    parser.add_argument("--source", help="Fichier .csv/.parquet/.arff ou 'openml:<id>' (défaut : source de la configuration)")
    parser.add_argument("--config", default=str(HERE / "datacard_config.json"))
    parser.add_argument("--header", default=str(HERE / "datacard_header.md"))
    parser.add_argument("--output", default=str(HERE / "DATACARD.md"))
    parser.add_argument("--dictionary", default=str(HERE / "outputs" / "data_dictionary.csv"))
    parser.add_argument("--snapshot", default=str(HERE / "outputs" / "datacard_stats.json"))
    parser.add_argument("--z-threshold", type=float, default=None, help="Seuil du z-score (défaut : configuration, sinon 3)")
    parser.add_argument("--no-compare", action="store_true", help="Ne pas comparer avec le batch précédent")
    args = parser.parse_args(argv)

    result = generate_datacard(
        args.source, config_path=args.config, header_path=args.header, output_path=args.output,
        dictionary_path=args.dictionary, snapshot_path=args.snapshot,
        z_threshold=args.z_threshold, compare=not args.no_compare,
    )
    comp = result["comparison"]
    if comp.get("premiere_generation"):
        evolution = "première génération"
    elif comp.get("identique"):
        evolution = "données identiques au batch précédent"
    else:
        evolution = "différences détectées (voir la section Évolution)"
    print(f"Datacard écrite : {result['output_path']}")
    print(f"Dictionnaire    : {result['dictionary_path']}")
    print(f"Batch           : {result['n_lignes']} lignes, {result['n_colonnes']} colonnes, "
          f"empreinte {result['fingerprint'][:16]}")
    print(f"Évolution       : {evolution}")


if __name__ == "__main__":
    main()
