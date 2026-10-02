# Model card : probabilité de victoire à 15 minutes (League of Legends)

| | |
|---|---|
| **Modèle** | Probabilité de victoire de l'équipe bleue, estimée à 15:00 |
| **Version** | 1.0 |
| **Auteur** | Louis, EPSI (2026-2027) |
| **Date de rédaction** | 01/10/2026 |
| **Données** | OpenML 43635 nettoyé : `DATACARD.md` (données brutes) et `LINEAGE.md` (nettoyage) |
| **Statut** | Prototype pédagogique, non déployé |

> Cette entête est rédigée à la main. La section 8 est générée par `generate_model_card.py` à partir
> des résultats de `train_models.py` et des graphes de `make_figures.py` : chiffres, tableaux et graphes
> y sont toujours à jour.

## 1. Résumé

Le modèle estime, à la 15e minute d'une partie classée, la probabilité que l'équipe bleue gagne. Il est
pensé pour un affichage en direct (overlay de stream) ou l'annotation d'une VOD : il n'utilise donc que
des informations disponibles pendant la partie, ce qui exclut l'or d'équipe.

Quatre familles de modèles ont été comparées (régression logistique, gradient boosting, forêt aléatoire,
réseau de neurones). Leurs performances sont presque identiques : à 15 minutes, la limite vient de
l'information disponible, pas du modèle. La régression logistique est retenue parce qu'elle obtient la
meilleure log loss en validation croisée, qu'elle reste au niveau des autres sur tous les critères de test,
et qu'elle est de loin la plus simple, la plus légère et la plus explicable.

## 2. Usage prévu

**Prévu**
- Afficher une probabilité de victoire à 15:00 sur un stream ou une VOD, à titre informatif.
- Support pédagogique : comparer des modèles, discuter calibration, surapprentissage et interprétabilité.

**Hors périmètre**
- Paris ou toute décision à enjeu financier.
- Autres rangs, autres serveurs ou version actuelle du jeu sans nouvelle validation : le modèle a appris
  sur des parties Diamant nord-américaines antérieures à la mi-2020.
- Prédire à un autre instant que 15:00 : le modèle ne connaît que l'état de la partie à 15 minutes.
- Juger un joueur : le modèle décrit des équipes, pas des individus.

## 3. Données d'entraînement

- **Jeu utilisé** : `data/clean/lol_diamond_15min_clean.csv`, produit par `clean_data.py`. Doublons,
  colonnes cassées et parties avec un joueur absent ont été retirés (détail et empreintes dans `LINEAGE.md`).
- **Découpage temporel** : les 80 % de parties les plus anciennes (selon `matchId`) servent à l'apprentissage
  et au choix des hyperparamètres, les 20 % les plus récentes au test final. Ce découpage imite l'usage
  réel : prédire des parties futures à partir du passé.
- **Le test n'est utilisé qu'une fois**, après la sélection, pour mesurer les performances annoncées.

## 4. Variables d'entrée

D'après le modèle du domaine (`DATACARD.md`, section 4), les deux équipes sont décrites par les mêmes
indicateurs : le modèle travaille sur les écarts bleu moins rouge. Chaque variable correspond à une
donnée que l'API locale du client de jeu (Live Client Data API) fournit pendant la partie.

| Variable | Définition | Source en direct |
|---|---|---|
| `ecart_eliminations` | Éliminations de l'équipe bleue moins celles de l'équipe rouge | `scores.kills` de chaque joueur, sommés par équipe |
| `ecart_cs` | Score de sbires (voie + jungle) bleu moins rouge | `scores.creepScore` de chaque joueur |
| `ecart_niveau` | Niveau moyen bleu moins niveau moyen rouge | `level` de chaque joueur |
| `ecart_monstres_epiques` | Dragons et Hérauts tués, bleu moins rouge | événements `DragonKill` et `HeraldKill` |
| `ecart_tours` | Tours détruites par l'équipe bleue moins par l'équipe rouge | événements `TurretKilled` |

L'écart d'or n'est pas utilisé : l'API ne donne l'or courant que pour le joueur actif. Un modèle « avec or »
est entraîné uniquement comme référence, pour mesurer ce que coûte cette contrainte (section 8.10).

## 5. Critères de validation et règle de sélection

- **Critère principal : la log loss.** Le modèle affiche une probabilité ; la log loss récompense les
  probabilités honnêtes et punit les certitudes fausses, là où l'exactitude ignore la confiance.
- **Sélection en validation croisée** (5 plis, sur l'apprentissage seul), pour ne pas ajuster le choix au test.
- **Règle d'un écart-type.** Les modèles sont très proches : on retient tous ceux dont la log loss de
  validation croisée est à moins d'une erreur standard du meilleur, puis le plus simple d'entre eux.
  Ordre de simplicité : régression logistique (quelques coefficients lisibles), gradient boosting
  (centaines de petits arbres), forêt aléatoire (centaines d'arbres profonds, modèle lourd), réseau de
  neurones (boîte noire, sensible à l'initialisation).
- **Critères secondaires** : score de Brier, AUC ROC, AUC précision-rappel, exactitude, calibration,
  écart entre apprentissage et test, temps de prédiction, taille, interprétabilité. Pour chacun, la
  section 8.5 compare le modèle retenu aux modèles abandonnés, avec un intervalle de confiance à 95 %
  obtenu par bootstrap apparié sur le test.

## 6. Limites et biais connus

1. **Pas de durée de partie dans les données.** Les parties abandonnées dès 15:00 ne peuvent pas être
   isolées : leur issue est presque connue au moment de la prédiction, ce qui gonfle un peu les scores,
   surtout dans les parties à grand écart.
2. **Époque et niveau.** Diamant, serveur NA1, parties antérieures à la mi-2020. Le jeu a changé depuis
   (Larves du Néant en 2024, notamment) et la façon dont un avantage se convertit en victoire dépend
   sans doute du niveau : recalibrer sur des parties récentes avant tout usage réel (projet 3).
3. **Monstres épiques.** Dragons et Hérauts sont confondus dans une même variable (colonne d'origine mal
   extraite).
4. **Pas de composition d'équipe.** Un même avantage ne vaut pas la même chose selon les champions : une
   composition faite pour la fin de partie peut gagner en partant de derrière.
5. **Parties serrées.** Quand l'écart est faible, le modèle n'est guère meilleur qu'une pièce
   (section 8.7) : l'affichage doit montrer cette incertitude plutôt qu'un chiffre trop précis.
6. **Parties avec joueur absent exclues** de l'apprentissage : le modèle n'a pas appris ces situations.

## 7. Éthique et maintenance

- **Données personnelles** : aucune en entrée du modèle ; les `matchId` de l'apprentissage restent des
  identifiants pseudonymes, à ne pas utiliser pour retrouver des joueurs.
- **Effet sur les joueurs** : un pourcentage affiché en direct peut décourager une équipe ou influencer le
  public. Le présenter comme une estimation, avec son incertitude.
- **Paris** : usage exclu.
- **Mise à jour** : `python run_pipeline.py` (ou, étape par étape, `clean_data.py`, `train_models.py`,
  `make_figures.py` puis `generate_model_card.py`). La section 8 est réécrite ; cette entête n'est modifiée
  qu'à la main.
