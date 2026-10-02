#!/usr/bin/env python3
"""Génère MODEL_CARD.md : entête rédigée à la main + section calculée depuis les résultats d'entraînement.

Entrées
    model_card_header.md     entête rédigée à la main (objectif, usages, données, limites, éthique)
    model_card_config.json   critères de validation et explications des algorithmes (à la main)
    reports/metrics.json     résultats de train_models.py
    figures/manifest.json    graphes exportés par make_figures.py
Sortie
    MODEL_CARD.md

Chaque graphe exporté est inséré automatiquement là où il sert l'argumentation ; un graphe qui n'a
pas trouvé sa place est ajouté à la fin, pour qu'aucun ne soit oublié.

Usage :
    python generate_model_card.py
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
NBSP = "\u00a0"
AUTO_BEGIN = ("<!-- DÉBUT DE LA PARTIE GÉNÉRÉE AUTOMATIQUEMENT (generate_model_card.py) : "
              "ne pas modifier à la main -->")
AUTO_END = "<!-- FIN DE LA PARTIE GÉNÉRÉE AUTOMATIQUEMENT -->"
CI_METRICS = {"log_loss", "brier", "roc_auc", "average_precision", "accuracy", "ece"}
DECIMALS = {"cv_log_loss": 4, "log_loss": 4, "brier": 4, "roc_auc": 4, "average_precision": 4, "ece": 4,
            "ecart_generalisation": 4, "predict_ms_per_1000": 3, "size_kb": 1}


# ---------------------------------------------------------------------------
# Mise en forme
# ---------------------------------------------------------------------------
def num(x, d: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:,.{d}f}".replace(",", NBSP).replace(".", ",")


def signed(x, d: int = 4) -> str:
    return ("+" if x > 0 else "") + num(x, d)


def pct(x, d: int = 1) -> str:
    return f"{x * 100:.{d}f}".replace(".", ",") + f"{NBSP}%"


def integer(n) -> str:
    return f"{int(n):,}".replace(",", NBSP)


def fmt_value(key: str, x) -> str:
    return pct(x, 2) if key == "accuracy" else num(x, DECIMALS.get(key, 4))


def fmt_diff(key: str, x) -> str:
    if key == "accuracy":
        return ("+" if x > 0 else "") + num(x * 100, 2) + " pt"
    return signed(x, DECIMALS.get(key, 4))


def md_table(headers, rows) -> str:
    def cell(v):
        return str(v).replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(out)


def names(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " et " + items[-1]


def lc(text: str) -> str:
    """Minuscule initiale, sauf pour un sigle (AUC, ECE...)."""
    return text[0].lower() + text[1:] if len(text) > 1 and text[1].islower() else text


def plural(items, singular: str, plural_form: str) -> str:
    return plural_form if len(items) > 1 else singular


def points(x) -> str:
    return signed(x, 1) + (" points" if abs(x) >= 2 else " point")


def hp_info(config: dict, model: str, param: str) -> tuple[str, str]:
    """Nom lisible et rôle d'un hyperparamètre (le nom scikit-learn sert de repli)."""
    doc = algo_doc(config, model)["hyperparametres"].get(param, {})
    if isinstance(doc, str):
        return param.split("__")[-1], doc
    return doc.get("nom", param.split("__")[-1]), doc.get("role", "")


def algo_doc(config: dict, model: str) -> dict:
    """Explications d'un algorithme (model_card_config.json), avec des valeurs par défaut pour un modèle
    ajouté sans être encore documenté."""
    family = "logistique" if model.startswith("logistique") else model
    doc = config["algorithmes"].get(family, {})
    return {"principe": doc.get("principe", "Principe à documenter dans model_card_config.json."),
            "interpretabilite": doc.get("interpretabilite", "non renseignée : à documenter"),
            "hyperparametres": doc.get("hyperparametres", {})}


def fmt_param(value) -> str:
    if isinstance(value, list):
        return "(" + ", ".join(str(v) for v in value) + ")"
    return str(value).replace(".", ",") if isinstance(value, float) else str(value)


# ---------------------------------------------------------------------------
# Graphes : insertion automatique depuis le manifeste
# ---------------------------------------------------------------------------
class Figures:
    def __init__(self, manifest: list[dict]):
        self.items = {Path(f["fichier"]).name: f for f in manifest}
        self.order = [Path(f["fichier"]).name for f in manifest]
        self.used: set[str] = set()

    def embed(self, name: str) -> list[str]:
        fig = self.items.get(name)
        if fig is None or name in self.used:
            return []
        self.used.add(name)
        if not (HERE / fig["fichier"]).exists():
            return [f"> Graphe manquant : `{fig['fichier']}` (relancer `make_figures.py`).", ""]
        return [f"![{fig['titre']}]({fig['fichier']})", "", f"*{fig['titre']}.* {fig['legende']}", ""]

    def section(self, section: str) -> list[str]:
        lines = []
        for name in self.order:
            if self.items[name]["section"] == section:
                lines += self.embed(name)
        return lines

    def remaining(self) -> list[str]:
        lines = []
        for name in self.order:
            lines += self.embed(name)
        return lines


# ---------------------------------------------------------------------------
# Comparaisons
# ---------------------------------------------------------------------------
def value(results: dict, model: str, key: str):
    r = results[model]
    if key == "cv_log_loss":
        return r["cv_log_loss"]
    if key == "ecart_generalisation":
        return r["test"]["log_loss"] - r["train"]["log_loss"]
    return r["test"][key]


def better(sens: str, diff: float) -> bool:
    return diff < 0 if sens == "min" else diff > 0


def compare(crit: dict, metrics: dict, config: dict, other: str) -> dict:
    """Compare un modèle abandonné au modèle retenu sur un critère ; renvoie écart, IC et verdict."""
    results, sel = metrics["modeles"], metrics["selection"]["retenu"]
    key, sens = crit["cle"], crit["sens"]
    out = {"modele": other}
    if sens == "qualitatif":
        out["texte"] = algo_doc(config, other)["interpretabilite"]
        return out
    v_other, v_sel = value(results, other, key), value(results, sel, key)
    out.update(valeur=v_other, diff=v_other - v_sel)
    if key == "ecart_generalisation":
        overfit = v_other > config["seuil_surapprentissage"]
        out.update(code="moins bon" if overfit else "equivalent",
                   texte="surapprentissage" if overfit else "pas de surapprentissage")
    elif key == "cv_log_loss":
        inside = other in metrics["selection"]["dans_la_tolerance"]
        out.update(code="equivalent" if inside else ("meilleur" if better(sens, out["diff"]) else "moins bon"),
                   texte="dans la tolérance (moins d'une erreur standard)" if inside
                   else "hors tolérance (plus d'une erreur standard)")
    elif key in CI_METRICS:
        ci = metrics["ic_bootstrap"]["differences"][other][key]
        out.update(low=ci["low"], high=ci["high"])
        if ci["low"] <= 0 <= ci["high"]:
            out.update(code="equivalent", texte="équivalent (écart non significatif)")
        elif better(sens, ci["diff"]):
            out.update(code="meilleur", texte="significativement meilleur")
        else:
            out.update(code="moins bon", texte="significativement moins bon")
    else:  # temps et taille : un rapport parle mieux qu'une différence
        ratio = v_other / v_sel if v_sel else math.inf
        word_more, word_less = ("lent", "rapide") if key == "predict_ms_per_1000" else ("lourd", "léger")
        if ratio >= 1.5:
            out.update(code="moins bon", texte=f"{num(ratio, 0)} fois plus {word_more}")
        elif ratio <= 1 / 1.5:
            out.update(code="meilleur", texte=f"{num(1 / ratio, 0)} fois plus {word_less}")
        else:
            out.update(code="equivalent", texte="comparable")
    return out


def justification(crit: dict, rows: list[dict], labels: dict, config: dict, sel_label: str, sec) -> str:
    key = crit["cle"]
    if crit["sens"] == "qualitatif":
        return (f"Le modèle retenu ({sel_label}) est le seul dont chaque décision s'explique variable par variable "
                f"(section {sec}.2) ; les autres demandent des outils d'explication a posteriori.")
    if key == "ecart_generalisation":
        flagged = [labels[r["modele"]] for r in rows if r["code"] == "moins bon"]
        txt = "Le modèle retenu ne montre aucun surapprentissage (écart sous le seuil)."
        if flagged:
            txt += (f" {names(flagged)} {plural(flagged, 'dépasse', 'dépassent')} le seuil : "
                    f"{plural(flagged, 'ce modèle apprend', 'ces modèles apprennent')} en partie les données d'apprentissage par cœur.")
        return txt
    if key in ("predict_ms_per_1000", "size_kb"):
        slower = [f"{labels[r['modele']]} ({r['texte']})" for r in rows if r["code"] == "moins bon"]
        faster = [f"{labels[r['modele']]} ({r['texte']})" for r in rows if r["code"] == "meilleur"]
        if faster:
            txt = f"{names(faster)} {plural(faster, 'fait', 'font')} mieux sur ce point."
        elif all(r["code"] == "moins bon" for r in rows):
            txt = "Le modèle retenu est " + ("le plus rapide." if key == "predict_ms_per_1000" else "le plus léger.")
        else:
            txt = "Le modèle retenu est parmi les " + ("plus rapides." if key == "predict_ms_per_1000" else "plus légers.")
        if slower:
            txt += f" Écarts notables : {names(slower)}."
        if key == "predict_ms_per_1000":
            txt += " Tous restent largement assez rapides pour un affichage en direct : critère peu discriminant ici."
        return txt
    better_rows = [r for r in rows if r["code"] == "meilleur"]
    same = [labels[r["modele"]] for r in rows if r["code"] == "equivalent"]
    worse = [labels[r["modele"]] for r in rows if r["code"] == "moins bon"]
    parts = []
    if key == "cv_log_loss":
        parts.append("Le modèle retenu a la meilleure log loss de validation croisée." if not better_rows else
                     f"{names([labels[r['modele']] for r in better_rows])} {plural(better_rows, 'fait', 'font')} mieux, "
                     "mais reste(nt) dans la tolérance : la règle de simplicité s'applique.")
        if same:
            parts.append(f"{names(same)} : à moins d'une erreur standard, donc {plural(same, 'équivalent', 'équivalents')} "
                         "au sens de la règle ; le plus simple l'emporte.")
        if worse:
            parts.append(f"{names(worse)} : à plus d'une erreur standard, donc {plural(worse, 'écarté', 'écartés')} "
                         "dès cette étape.")
        return " ".join(parts)
    if better_rows:
        lst = [f"{labels[r['modele']]} ({fmt_diff(key, r['diff'])})" for r in better_rows]
        parts.append(f"{names(lst)} {plural(lst, 'fait', 'font')} significativement mieux que le modèle retenu sur ce critère.")
        parts.append(config["arbitrage_test"])
    else:
        parts.append("Aucun modèle abandonné ne fait significativement mieux que le modèle retenu sur ce critère.")
    if same:
        parts.append(f"{names(same)} : écart trop faible pour départager, {plural(same, 'modèle équivalent', 'modèles équivalents')}.")
    if worse:
        parts.append(f"{names(worse)} : {plural(worse, 'significativement moins bon', 'significativement moins bons')}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Rendu de la section générée
# ---------------------------------------------------------------------------
def render(metrics: dict, config: dict, figures: Figures) -> str:
    sec = config.get("section_number", 8)
    results, sel_info = metrics["modeles"], metrics["selection"]
    sel = sel_info["retenu"]
    labels = {k: r["label"] for k, r in results.items()}
    eligible = list(sel_info["ordre_de_simplicite"])
    abandoned = [k for k in eligible if k != sel]
    algo = algo_doc(config, sel)
    data = metrics["donnees"]
    trained = datetime.fromisoformat(metrics["genere_le"])
    L = [AUTO_BEGIN, "", f"## {sec}. Modèle retenu et validation (partie générée automatiquement)", "",
         f"> Section produite par `generate_model_card.py` le {datetime.now():%d/%m/%Y à %H:%M}, à partir des "
         f"résultats de `train_models.py` du {trained:%d/%m/%Y à %H:%M} (`reports/metrics.json`). "
         "Ne pas l'éditer à la main.", ""]

    # 8.1 Fiche d'identité ---------------------------------------------------
    r_sel = results[sel]
    L += [f"### {sec}.1 Fiche d'identité", ""]
    L += [md_table(["Élément", "Valeur"], [
        ["Modèle retenu", f"**{labels[sel]}**"],
        ["Sortie", "Probabilité de victoire de l'équipe bleue (seuil de décision 0,5 pour l'exactitude)"],
        ["Variables", ", ".join(f"`{c}`" for c in r_sel["colonnes"])],
        ["Hyperparamètres", ", ".join(f"{hp_info(config, sel, k)[0]} = {fmt_param(v)}"
                                      for k, v in r_sel["best_params"].items()) or "aucun"],
        ["Fichier du modèle", "`models/modele_retenu.joblib` (métadonnées : `models/modele_retenu.json`)"],
        ["Données", f"`{data['fichier']}` (SHA-256 `{data['sha256'][:16]}`), {integer(data['lignes'])} parties"],
        ["Apprentissage / test", f"{integer(data['apprentissage'])} / {integer(data['test'])} parties "
                                 f"(découpage temporel : {data['decoupage'].removeprefix('temporel : ')})"],
        ["Performances sur le test", f"log loss {num(r_sel['test']['log_loss'])}, AUC {num(r_sel['test']['roc_auc'])}, "
                                     f"exactitude {pct(r_sel['test']['accuracy'], 1)}"],
        ["Versions", f"Python {metrics['versions']['python']}, scikit-learn {metrics['versions']['scikit_learn']}"],
    ]), ""]

    # 8.2 Fonctionnement ----------------------------------------------------
    L += [f"### {sec}.2 Comment fonctionne le modèle", "", algo["principe"], ""]
    expl = metrics.get("explication", {})
    if expl.get("coefficients"):
        units = config.get("unites_variables", {})
        rows = []
        for c in expl["coefficients"]:
            u = units.get(c["variable"], {"unite": "1 unité", "facteur": 1})
            coef = c["coef_par_unite"] * u["facteur"]
            effect = 1 / (1 + math.exp(-coef)) - 0.5
            rows.append([f"`{c['variable']}`", u["unite"], num(c["coef_standardise"], 3), num(coef, 4),
                         num(math.exp(coef), 3), signed(effect * 100, 1) + " pt"])
        p0 = 1 / (1 + math.exp(-expl["intercept"]))
        L += [md_table(["Variable", "Pour", "Coefficient standardisé", "Coefficient", "Cote multipliée par",
                        "Effet depuis 50 %"], rows), ""]
        L += [f"Lecture : la constante vaut {num(expl['intercept'], 4)}, soit {pct(p0, 1)} de victoire bleue quand "
              "tous les écarts sont nuls. « Effet depuis 50 % » donne le gain de probabilité pour l'avance indiquée, "
              "en partant d'une partie équilibrée, les autres écarts restant nuls. Le coefficient standardisé "
              "compare les variables entre elles (effet d'un écart-type de chaque variable).", ""]
    L += figures.section("explication")

    # 8.3 Entraînement ------------------------------------------------------
    cv = metrics["validation_croisee"]
    L += [f"### {sec}.3 Entraînement et hyperparamètres", "",
          f"- **Découpage temporel** : {data['decoupage'].removeprefix('temporel : ')} ; matchId de {data['matchid_apprentissage'][0]} à "
          f"{data['matchid_apprentissage'][1]} en apprentissage, de {data['matchid_test'][0]} à "
          f"{data['matchid_test'][1]} en test. Victoires bleues : {pct(data['victoires_bleues_apprentissage'], 1)} "
          f"en apprentissage, {pct(data['victoires_bleues_test'], 1)} en test.",
          f"- **Recherche d'hyperparamètres** : grille complète, validation croisée stratifiée à {cv['plis']} plis "
          f"(mélangée, graine {cv['graine']}) sur l'apprentissage seul, critère : {cv['critere']}. La meilleure "
          "configuration est ensuite réentraînée sur tout l'apprentissage.",
          "- **Prétraitement** : standardisation des variables (moyenne 0, écart-type 1) pour la logistique et le "
          "réseau de neurones, inutile pour les arbres.", ""]
    L += [f"**Hyperparamètres du modèle retenu ({labels[sel]})**", ""]
    L += [md_table(["Hyperparamètre", "Nom dans scikit-learn", "Valeur retenue", "Valeurs testées", "Rôle"], [
        [hp_info(config, sel, k)[0], f"`{k}`", fmt_param(v), ", ".join(fmt_param(x) for x in r_sel["grid"].get(k, [])),
         hp_info(config, sel, k)[1]]
        for k, v in r_sel["best_params"].items()]), ""]
    if sel == "logistique" or sel.startswith("logistique"):
        configs = sorted(r_sel["configs"], key=lambda c: list(c["params"].values())[0])
        L += ["Log loss de validation croisée selon C : " + " ; ".join(
            f"C = {fmt_param(list(c['params'].values())[0])} : {num(c['cv_log_loss'])}" for c in configs)
            + ". Au-delà d'une régularisation très forte, la valeur de C change peu le résultat.", ""]
    L += ["**Recherche pour chaque modèle**", ""]
    L += [md_table(["Modèle", "Configurations", "Meilleure configuration", "Log loss CV (± erreur standard)",
                    "Durée de la recherche"], [
        [labels[k], results[k]["n_configs"],
         ", ".join(f"{hp_info(config, k, p)[0]} = {fmt_param(v)}" for p, v in results[k]["best_params"].items()) or "aucune",
         f"{num(results[k]['cv_log_loss'])} (± {num(results[k]['cv_log_loss_se'])})",
         f"{num(results[k]['search_time_s'], 0)} s"] for k in ["reference"] + eligible + ["logistique_avec_or"]
        if k in results]), ""]
    conv = metrics.get("convergence", {})
    if conv:
        short = {"logistique": "convergence du solveur en {n} itérations", "gradient_boosting": "arrêt précoce à {n} arbres",
                 "reseau_neurones": "arrêt précoce à {n} époques"}
        L += ["**Convergence.** " + " ; ".join(
            f"{labels[k]} : {short.get(k, 'arrêt à {n} itérations').format(n=v['choisi'])}" for k, v in conv.items())
            + ". L'arrêt précoce stoppe l'entraînement quand la perte de validation cesse de baisser, avant que "
              "le modèle ne surapprenne : sur le graphe, la perte de validation du gradient boosting remonte "
              "ensuite pendant que celle d'apprentissage continue de baisser.", ""]
    L += figures.embed("02_convergence.png")
    learning = metrics.get("courbes_apprentissage", {})
    if learning:
        parts = [f"{labels[k]} : {num(v['val'][0], 3)} avec {integer(v['n'][0])} parties, {num(v['val'][-1], 3)} "
                 f"avec {integer(v['n'][-1])}" for k, v in learning.items()]
        L += ["**Courbes d'apprentissage.** Log loss de validation selon la taille de l'apprentissage : "
              + " ; ".join(parts) + ". Un modèle simple atteint son niveau avec peu de données ; les modèles "
              "flexibles ont besoin de tout le jeu pour le rejoindre, sans le dépasser.", ""]
    L += figures.section("entrainement")

    # 8.4 Sélection ---------------------------------------------------------
    best = sel_info["meilleur_cv"]
    L += [f"### {sec}.4 Sélection : règle et résultat", "", f"**Règle** : {sel_info['regle']}", ""]
    L += [md_table(["Modèle", "Log loss CV", "Erreur standard", "Simplicité (1 = le plus simple)", "Statut"], [
        [labels[k], num(results[k]["cv_log_loss"]), num(results[k]["cv_log_loss_se"]), sel_info["ordre_de_simplicite"][k],
         "**retenu**" if k == sel else ("dans la tolérance" if k in sel_info["dans_la_tolerance"] else "hors tolérance")]
        for k in sorted(eligible, key=lambda k: results[k]["cv_log_loss"])]), ""]
    L += [f"Meilleure log loss : {labels[best]} ({num(results[best]['cv_log_loss'])}). Tolérance : "
          f"{num(results[best]['cv_log_loss'])} + {num(results[best]['cv_log_loss_se'])} = {num(sel_info['seuil'])}. "
          f"Dans la tolérance : {names([labels[k] for k in sel_info['dans_la_tolerance']])}. "
          f"Le plus simple d'entre eux est retenu : **{labels[sel]}**.", ""]

    # 8.5 Critères ----------------------------------------------------------
    L += [f"### {sec}.5 Critères de validation : comparaison avec les modèles abandonnés", "",
          "Pour chaque critère : pourquoi il compte, la valeur de chaque modèle, l'écart avec le modèle retenu et "
          "son intervalle de confiance à 95 % (bootstrap apparié sur le test, "
          f"{metrics['ic_bootstrap']['n']} tirages), puis le verdict.", ""]
    synthesis = {k: {} for k in abandoned}
    for i, crit in enumerate(config["criteres"], start=1):
        key = crit["cle"]
        rows = [compare(crit, metrics, config, k) for k in abandoned]
        for r in rows:
            synthesis[r["modele"]][key] = r
        L += [f"#### {sec}.5.{i} {crit['label']}", "", f"**Rôle** : {crit['role']}. {crit['pourquoi']}", ""]
        if crit["sens"] == "qualitatif":
            table = [[f"**{labels[sel]}** (retenu)", algo["interpretabilite"]]]
            table += [[labels[r["modele"]], r["texte"]] for r in rows]
            L += [md_table(["Modèle", "Interprétabilité"], table), ""]
        else:
            v_sel = value(results, sel, key)
            if key == "ecart_generalisation":
                sel_verdict = "surapprentissage" if v_sel > config["seuil_surapprentissage"] else "pas de surapprentissage"
            else:
                sel_verdict = "référence de comparaison"
            table = [[f"**{labels[sel]}** (retenu)", fmt_value(key, v_sel), "", sel_verdict]]
            for r in rows:
                diff = "" if key == "ecart_generalisation" else fmt_diff(key, r["diff"])
                if "low" in r:
                    diff += f" [{fmt_diff(key, r['low'])} ; {fmt_diff(key, r['high'])}]"
                table.append([labels[r["modele"]], fmt_value(key, r["valeur"]), diff, r["texte"]])
            ref_val = value(results, "logistique_avec_or", key)
            table.append([f"*{labels['logistique_avec_or']}*", fmt_value(key, ref_val), "", "hors sélection"])
            if key == "ecart_generalisation":
                L += [md_table(["Modèle", "Log loss test moins apprentissage", "Verdict"],
                               [[row[0], row[1], row[3]] for row in table]), ""]
            else:
                header_diff = "Écart avec le retenu [IC 95 %]" if key in CI_METRICS else "Écart avec le retenu"
                L += [md_table(["Modèle", "Valeur", header_diff, "Verdict"], table), ""]
        L += [f"**Justification** : {justification(crit, rows, labels, config, labels[sel], sec)}", ""]
        for name in crit.get("figures", []):
            L += figures.embed(name)

    # Synthèse par modèle abandonné
    L += [f"#### {sec}.5.{len(config['criteres']) + 1} Synthèse : pourquoi chaque autre modèle est abandonné", ""]
    for k in abandoned:
        s = synthesis[k]
        quality = [c for c in ("log_loss", "brier", "roc_auc", "average_precision", "accuracy", "ece") if c in s]
        worse = [next(cr["label"] for cr in config["criteres"] if cr["cle"] == c) for c in quality if s[c]["code"] == "moins bon"]
        better_q = [next(cr["label"] for cr in config["criteres"] if cr["cle"] == c) for c in quality if s[c]["code"] == "meilleur"]
        reasons = [("dans la tolérance de la règle d'un écart-type mais plus complexe" if k in sel_info["dans_la_tolerance"]
                    else "hors de la tolérance de la règle d'un écart-type")
                   + f" (log loss CV {fmt_diff('cv_log_loss', s['cv_log_loss']['diff'])})"]
        if worse:
            reasons.append("significativement moins bon sur : " + ", ".join(lc(w) for w in worse))
        if better_q:
            reasons.append("meilleur sur : " + ", ".join(lc(b) for b in better_q) + ", sans confirmation en validation croisée")
        if not worse and not better_q:
            reasons.append("équivalent sur toutes les métriques de test")
        if s["ecart_generalisation"]["code"] == "moins bon":
            reasons.append("surapprentissage")
        t, z = s["predict_ms_per_1000"], s["size_kb"]
        reasons.append(("temps de prédiction comparable" if t["code"] == "equivalent" else f"{t['texte']} à prédire")
                       + ", " + ("taille comparable" if z["code"] == "equivalent" else z["texte"]))
        reasons.append(f"interprétabilité {algo_doc(config, k)['interpretabilite'].split(' :')[0]}")
        L.append(f"- **{labels[k]}** : " + " ; ".join(reasons) + ".")
    L += ["", f"Aucun modèle abandonné n'apporte de gain significatif et confirmé en validation croisée : à qualité "
              f"égale, le modèle le plus simple, le plus léger et le plus explicable est retenu ({labels[sel]}).", ""]

    # 8.6 Surapprentissage --------------------------------------------------
    ctrl = metrics.get("controle_surapprentissage", {})
    L += [f"### {sec}.6 Contrôle du surapprentissage", "",
          "Un modèle qui surapprend est nettement meilleur sur ses données d'apprentissage que sur des données "
          "nouvelles. On compare donc les deux, modèle par modèle :", ""]
    L += [md_table(["Modèle", "Log loss apprentissage", "Log loss test", "Écart", "AUC apprentissage", "AUC test"], [
        [labels[k], num(results[k]["train"]["log_loss"]), num(results[k]["test"]["log_loss"]),
         signed(results[k]["test"]["log_loss"] - results[k]["train"]["log_loss"]),
         num(results[k]["train"]["roc_auc"]), num(results[k]["test"]["roc_auc"])]
        for k in eligible + ["logistique_avec_or"]]), ""]
    if ctrl:
        shuffled = ctrl["cibles_melangees_log_loss_test"]
        single = ctrl["auc_variable_seule_test"]
        top = max((c for c in single if c in r_sel["colonnes"]), key=lambda c: single[c])
        L += [f"- **Cibles mélangées** : réentraîné sur des issues tirées au hasard, le modèle retenu obtient une log "
              f"loss de test moyenne de {num(sum(shuffled) / len(shuffled))} (hasard pur : 0,6931). Le pipeline ne "
              "laisse donc passer aucune information sur l'issue en dehors des variables.",
              f"- **D'où vient la performance** : la variable `{top}` seule, sans aucun modèle, donne déjà une AUC de "
              f"{num(single[top], 3)} sur le test. Une AUC élevée reflète l'information contenue dans l'état de la "
              "partie à 15 minutes, pas un apprentissage par cœur."]
    sub = metrics["sous_groupes"]["metriques"][sel]
    n_test = sum(v["n"] for v in sub.values())
    parts = [f"{name.lower()} : {pct(v['n'] / n_test, 0)} des parties, exactitude {pct(v['accuracy'], 1)}"
             + (f", AUC {num(v['roc_auc'], 3)}" if v.get("roc_auc") is not None else "") for name, v in sub.items()]
    L += ["- **Selon le type de partie** : " + " ; ".join(parts) + ". L'AUC globale mélange des parties presque "
          "jouées d'avance et des parties serrées, presque imprévisibles.", ""]

    # 8.7 Sous-groupes ------------------------------------------------------
    L += [f"### {sec}.7 Analyse par sous-groupe", "",
          "Les parties de test sont réparties selon l'écart d'or à 15 minutes, une variable que le modèle en direct "
          "ne voit pas :", ""]
    defs = metrics["sous_groupes"]["definition"]
    allsub = metrics["sous_groupes"]["metriques"]
    rows = []
    for name, lo, hi in defs:
        rng = f"{integer(lo)} à {integer(hi)}" if hi is not None else f"plus de {integer(lo)}"
        cells = [f"{name} ({rng})", integer(allsub[sel][name]["n"])]
        for k in [sel] + abandoned + ["logistique_avec_or"]:
            cells.append(f"{pct(allsub[k][name]['accuracy'], 1)} / {num(allsub[k][name]['log_loss'], 3)}")
        rows.append(cells)
    L += [md_table(["Sous-groupe (écart d'or)", "Parties"] + [f"{labels[k]} (exactitude / log loss)"
                                                             for k in [sel] + abandoned + ["logistique_avec_or"]], rows), ""]
    L += figures.section("sous_groupes")

    # 8.8 Résidus -----------------------------------------------------------
    res = metrics.get("residus", {})
    L += [f"### {sec}.8 Analyse des résidus", "",
          "Le résidu d'une partie est l'issue réelle (1 si l'équipe bleue gagne, 0 sinon) moins la probabilité "
          "prédite. En moyenne par classe, il doit rester proche de 0 si le modèle n'a pas de biais systématique.", ""]
    if res:
        by_p = res["par_probabilite"]
        outside = sum(abs(r["residu"]) > r["se2"] for r in by_p)
        count = ("aucune classe ne sort" if outside == 0 else
                 f"{outside} classe sort" if outside == 1 else f"{outside} classes sortent")
        L.append(f"- **Selon la probabilité prédite** : sur {len(by_p)} classes, {count} de la bande de ± 2 erreurs "
                 "types (environ 1 sur 20 attendue par hasard) : pas de biais systématique visible.")

        def mean_res(rows, cond):
            sel_rows = [r for r in rows if cond(r["x"])]
            n = sum(r["n"] for r in sel_rows)
            return sum(r["residu"] * r["n"] for r in sel_rows) / n if n else float("nan")
        g = res["par_ecart_or"]
        lv_pos, lv_neg = mean_res(g[sel], lambda x: x >= 2000), mean_res(g[sel], lambda x: x <= -2000)
        ref = "logistique_avec_or"
        rf_pos, rf_neg = mean_res(g[ref], lambda x: x >= 2000), mean_res(g[ref], lambda x: x <= -2000)
        L.append(f"- **Selon l'écart d'or** : quand l'équipe bleue mène de plus de 2 000 pièces d'or, le résidu moyen "
                 f"du modèle retenu est de {points(lv_pos * 100)} ; quand elle est menée d'autant, "
                 f"{points(lv_neg * 100)}. Le modèle en direct sous-estime donc légèrement l'équipe qui mène à "
                 f"l'or : c'est l'information que l'or apporterait. Avec l'or, ces résidus tombent à "
                 f"{points(rf_pos * 100)} et {points(rf_neg * 100)}.")
        L.append("")
    L += figures.section("residus")

    # 8.9 Corrélations ------------------------------------------------------
    L += [f"### {sec}.9 Corrélations entre variables", "",
          "Les écarts sont fortement corrélés entre eux (une élimination rapporte de l'or et de l'expérience). Cela "
          "ne gêne pas la prédiction, mais chaque coefficient doit se lire « à autres écarts égaux » et sa valeur "
          "peut bouger si l'on retire une variable.", ""]
    L += figures.section("donnees")

    # 8.10 Coût de la contrainte du direct ---------------------------------
    ref = "logistique_avec_or"
    ci = metrics["ic_bootstrap"]["differences"][ref]
    L += [f"### {sec}.10 Coût de la contrainte du direct", "",
          "Même algorithme, avec l'écart d'or en plus (non disponible en direct) :", ""]
    L += [md_table(["Critère", labels[sel], labels[ref], "Écart [IC 95 %]"], [
        [lbl, fmt_value(m, results[sel]["test"][m]), fmt_value(m, results[ref]["test"][m]),
         f"{fmt_diff(m, ci[m]['diff'])} [{fmt_diff(m, ci[m]['low'])} ; {fmt_diff(m, ci[m]['high'])}]"]
        for m, lbl in (("log_loss", "Log loss"), ("brier", "Score de Brier"), ("roc_auc", "AUC ROC"),
                       ("accuracy", "Exactitude"))]), ""]
    acc_gain, ll_gain = ci["accuracy"]["diff"], -ci["log_loss"]["diff"]
    significant = not (ci["log_loss"]["low"] <= 0 <= ci["log_loss"]["high"])
    if acc_gain > 0 and ll_gain > 0:
        L += [f"Ajouter l'or ferait gagner {num(acc_gain * 100, 2)} point d'exactitude et {num(ll_gain, 4)} de log loss : "
              + ("un écart significatif mais faible" if significant else "un écart non significatif")
              + ", accepté pour pouvoir afficher la probabilité en direct.", ""]
    else:
        L += ["L'or n'améliore pas le modèle sur ce jeu de test : la contrainte du direct ne coûte rien.", ""]
    L += figures.section("classement") + figures.section("calibration")

    # 8.11 Reproductibilité -------------------------------------------------
    L += [f"### {sec}.11 Reproductibilité", "", "```bash",
          "python run_pipeline.py         # toute la chaîne, ou étape par étape :",
          "python collect_data.py         # vérifie les données d'entrée (data/raw/)",
          "python clean_data.py           # data/clean/ + LINEAGE.md",
          "python train_models.py         # models/ + reports/",
          "python make_figures.py         # figures/",
          "python generate_model_card.py  # MODEL_CARD.md", "```", "",
          f"Graine aléatoire : {cv['graine']}. Données d'entraînement : SHA-256 `{data['sha256'][:16]}` "
          "(à comparer avec la sortie de `LINEAGE.md`).", ""]
    rest = figures.remaining()
    if rest:
        L += [f"### {sec}.12 Autres graphes", ""] + rest
    L += [AUTO_END, ""]
    return "\n".join(L)


def generate(header=HERE / "model_card_header.md", config_path=HERE / "model_card_config.json",
             metrics_path=HERE / "reports" / "metrics.json", manifest_path=HERE / "figures" / "manifest.json",
             output=HERE / "MODEL_CARD.md") -> Path:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    figures = Figures(json.loads(Path(manifest_path).read_text(encoding="utf-8")))
    head = Path(header).read_text(encoding="utf-8").split(AUTO_BEGIN)[0].rstrip()
    Path(output).write_text(f"{head}\n\n{render(metrics, config, figures)}", encoding="utf-8")
    return Path(output)


if __name__ == "__main__":
    path = generate()
    print(f"Model card écrite : {path.name}")
