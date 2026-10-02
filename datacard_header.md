# Datacard : parties classées Diamant de League of Legends à 15 minutes

| | |
|---|---|
| **Jeu de données** | League-of-Legends-Diamond-Games-(First-15-Minutes) |
| **Plateforme** | OpenML, identifiant 43635, version 1 |
| **Lien** | https://www.openml.org/search?type=data&status=active&id=43635 |
| **Licence** | CC0 (domaine public) |
| **Auteur de la datacard** | Louis, EPSI (2026-2027) |
| **Dernière rédaction de l'entête** | 01/10/2026 |
| **Statut** | Exploration terminée, modélisation non commencée |

> Cette entête est rédigée à la main. La section 8 est générée par `generate_datacard.py`
> à chaque nouveau batch : les chiffres à jour (volumes, statistiques, contrôles) s'y trouvent.

## 1. Résumé

Chaque ligne décrit une partie classée de League of Legends, jouée au rang Diamant sur le serveur
nord-américain, observée à la 15e minute. Pour chacune des deux équipes (bleue et rouge), on dispose
de ses ressources (or, sbires, monstres de jungle, niveau moyen) et de ses actions (éliminations,
monstres épiques, tours), ainsi que du vainqueur de la partie (`blue_win`).

Question posée par l'auteur d'origine : peut-on prédire l'issue d'une partie à partir de ses
15 premières minutes ?

## 2. Provenance

- **Source d'origine** : jeu de données Kaggle « League of Legends Diamond Games (First 15 Minutes) »
  publié par Ben Fattori (collecte datée du 31/07/2020), repris sur OpenML le 24/03/2022 (format ARFF,
  copie Parquet disponible).
- **Inspiration** : le jeu de données Kaggle « League of Legends Diamond Ranked Games (10 min) », qui
  suit le même principe à 10 minutes. D'après l'auteur, l'exactitude des modèles y plafonnait autour
  de 70 % ; ce jeu de données vise à voir si cinq fois plus de parties améliorent ce résultat.
- **Liens** :
  - OpenML : https://www.openml.org/search?type=data&status=active&id=43635
  - Kaggle : https://www.kaggle.com/datasets/benfattori/league-of-legends-diamond-games-first-15-minutes

## 3. Collecte des données

| Élément | Ce qui est documenté |
|---|---|
| Méthode | Extraction automatisée (scraping) via l'API officielle de Riot Games, avec l'accord de Riot (remerciements dans la description) |
| Saison | « Season 13 » (identifiant de saison renvoyé par l'API) |
| Serveur | NA1 (Amérique du Nord) |
| Rang | Diamant, divisions I, II, III et IV |
| Volume annoncé | Environ 50 000 parties (volume exact en section 8) |
| Instant d'observation | Cumul depuis le début de la partie jusqu'à la 15e minute |
| Date de collecte | 31/07/2020 (métadonnée OpenML) |

**Ce qui n'est pas documenté et que je retiens comme incertitudes :**

- **La période exacte des parties.** « Season 13 » est un identifiant de l'API, pas l'année. La table
  de référence publiée par Riot (`seasons.json`) s'arrête à l'identifiant 13, libellé « SEASON 2019 »,
  alors que des saisons ont suivi : ce champ ne permet donc pas de dater les parties. On sait
  seulement qu'elles sont antérieures à la date de collecte.
- **La version du jeu (patch)** : absente des données.
- **La méthode d'échantillonnage** : on ne sait pas comment les joueurs et les parties ont été choisis
  (un joueur actif peut apparaître dans beaucoup de parties).
- **Les requêtes exactes.** Les noms de colonnes reprennent ceux de la chronologie des parties de l'API
  Match de l'époque (or total, sbires, monstres de jungle et niveau par joueur, minute par minute, plus
  les événements d'élimination, de monstres et de bâtiments), ce qui suggère que l'état à 15 minutes
  vient de cette chronologie.

## 4. Modèle du domaine

```
Partie (matchId), issue : blue_win (1 = victoire de l'équipe bleue)
 ├── Équipe bleue (5 joueurs), état cumulé à 15:00
 │    ├── Ressources : Gold, MinionsKilled, JungleMinionsKilled, AvgLevel
 │    └── Actions    : ChampKills, HeraldKills, DragonKills, TowersDestroyed
 └── Équipe rouge (5 joueurs), mêmes indicateurs avec le préfixe red
```

- Les deux équipes sont décrites par les mêmes indicateurs : un modèle peut travailler sur les écarts
  (bleu moins rouge), plus informatifs que les valeurs brutes.
- Les ressources sont des états (ce que l'équipe possède à 15:00), les actions sont des événements
  comptés depuis le début de la partie.
- Contraintes attendues d'après les règles de 2020 : 500 pièces d'or de départ par joueur, niveau de
  1 à 18, 11 tours par équipe, au plus 2 dragons et 2 Hérauts avant 15:00. Ces contraintes sont
  déclarées comme règles dans `datacard_config.json` et vérifiées automatiquement (section 8.6).

## 5. Usages

**Prévus**
- Apprentissage du Machine Learning : classification binaire (prédire le vainqueur à 15 minutes),
  calibration de probabilités, interprétation de modèles.
- Analyse de la valeur des avantages de début de partie (or, éliminations, objectifs).

**Hors périmètre**
- Généraliser à d'autres rangs, d'autres serveurs ou à la version actuelle du jeu sans nouvelles
  données : le jeu a beaucoup changé depuis 2020 (par exemple, les Larves du Néant ont remplacé
  le premier Héraut en 2024).
- Évaluer des joueurs individuellement : aucune donnée par joueur.
- Paris ou décisions à enjeu financier.

## 6. Limites et problèmes connus

Constats issus de l'exploration (notebook `exploration_lol_diamond_15min.ipynb`). Les volumes
concernés sont recalculés à chaque batch en section 8.

1. **`blueDragonKills` et `redDragonKills` valent toujours 0.** Impossible en Diamant, où le premier
   dragon est presque toujours tué avant 15 minutes : erreur d'extraction probable. Colonnes à exclure.
2. **`blueHeraldKills` et `redHeraldKills` ont un libellé trompeur.** Le total par partie dépasse
   souvent 2, ce que les règles interdisent pour les Hérauts seuls. La distribution (souvent 3, au
   maximum 4) correspond au total dragons + Hérauts : ces colonnes comptent très probablement tous
   les monstres épiques.
3. **`blueTowersDestroyed` et `redTowersDestroyed` ont un sens inversé.** `blueTowersDestroyed` est
   corrélée négativement à la victoire bleue et positivement à l'or de l'équipe rouge : elle compte
   les tours **perdues** par l'équipe bleue. Explication probable : l'événement de destruction d'un
   bâtiment dans l'API porte l'équipe propriétaire du bâtiment, pas celle qui l'a détruit.
4. **`Unnamed:_0` est un index de ligne** exporté avec les données : à supprimer. Attention, il
   masque les doublons (deux lignes identiques ont des index différents).
5. **Doublons exacts** (même `matchId`, mêmes valeurs) : à supprimer avant toute modélisation.
6. **Types** : le format ARFF stocke tout en flottant, y compris `matchId` et les comptages. Convertir
   en entiers **signés** avant de calculer des écarts (avec un type non signé, 6 - 12 donne 250).
7. **Parties anormales** : quelques parties ont un niveau moyen très bas à 15 minutes (joueur absent
   ou déconnecté probable). Elles sont signalées par une règle d'alerte, pas supprimées.
8. **Informations absentes** : champions, composition des équipes, rôles, joueurs, durée totale,
   version du jeu.
9. **Représentativité** : uniquement NA1, uniquement Diamant, parties antérieures au 31/07/2020.

## 7. Éthique, licence et maintenance

- **Licence CC0** : réutilisation libre.
- **Données personnelles** : aucun pseudo de joueur, mais `matchId` permet de retrouver la partie,
  donc les joueurs, via l'API de Riot. Les données sont pseudonymisées, pas anonymes : ne pas chercher
  à réidentifier les joueurs. Toute nouvelle collecte doit respecter les conditions d'utilisation de
  l'API de Riot Games.
- **Mise à jour** : déposer le nouveau batch dans `data/raw/`, puis lancer
  `python generate_datacard.py --source data/raw/<fichier>.csv`. La section 8 est réécrite et
  comparée au batch précédent ; cette entête n'est modifiée qu'à la main.
