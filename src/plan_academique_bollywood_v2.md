# Plan académique du projet

## Titre proposé
**Analyse du succès commercial et de la rentabilité des films Bollywood : comparaison entre Gaussian Naive Bayes et Random Forest**

## 1. Introduction
L'industrie cinématographique cherche à comprendre quels facteurs favorisent la réussite commerciale d'un film. Dans ce projet, nous analysons un jeu de données de films Bollywood afin d'étudier l'effet du **budget**, du **nombre d'écrans**, du **nombre de votes**, de la **note moyenne** et du **genre** sur deux dimensions complémentaires :
- le **succès commercial** ;
- la **rentabilité**.

L'objectif n'est pas seulement de décrire les films passés, mais de dégager des profils de films performants afin d'estimer le potentiel de nouveaux films.

## 2. Problématique
**Dans quelle mesure l'analyse du budget, du nombre d'écrans, du nombre de votes, de la note moyenne et du genre permet-elle de comprendre le succès commercial et la rentabilité des films Bollywood, puis d'estimer le potentiel de nouveaux films ?**

## 3. Objectifs
### Objectif général
Analyser les déterminants du succès commercial et de la rentabilité des films Bollywood, puis construire des modèles de classification capables d'estimer la classe de succès et la classe de rentabilité de nouveaux films.

### Objectifs spécifiques
1. Décrire le dataset et vérifier sa qualité.
2. Étudier la distribution des variables principales.
3. Mesurer le lien entre les variables explicatives et le revenu.
4. Créer une variable de **succès commercial** à partir du revenu.
5. Créer une variable de **rentabilité** à partir du ROI.
6. Construire un modèle **Gaussian Naive Bayes**.
7. Construire un modèle **Random Forest**.
8. Comparer les performances des deux approches.
9. Développer une interface interactive pour présenter les résultats et tester de nouveaux cas.

## 4. Hypothèses de recherche
- Les films avec un **budget élevé** ont davantage de chances de succès.
- Les films diffusés sur un **grand nombre d'écrans** obtiennent de meilleurs revenus.
- Les films ayant une **bonne note** et un **grand nombre de votes** ont plus de probabilité d'appartenir à la classe des films à succès.
- Certains **genres** sont plus fréquemment associés au succès ou à une rentabilité élevée.
- Le modèle **Random Forest** devrait obtenir de meilleures performances que **Gaussian Naive Bayes**, car il capture mieux les relations non linéaires et les interactions entre variables.

## 5. Description du dataset
Le dataset contient des informations sur des films Bollywood, notamment :
- `Genre`
- `Budget(INR)`
- `Revenue(INR)`
- `Number of Screens`
- `Rating(10)`
- `Votes`

### Variables retenues pour l'étude
#### Variables explicatives
- `Votes`
- `Budget(INR)`
- `Number of Screens`
- `Rating(10)`
- `Genre`

#### Variables cibles
1. **Success_Class** : classe de succès construite à partir de `Revenue(INR)`
2. **Profitability_Class** : classe de rentabilité construite à partir du ratio :
   \[
   ROI = \frac{Revenue(INR)}{Budget(INR)}
   \]

## 6. Méthodologie
### 6.1 Préparation des données
- Vérification des valeurs manquantes
- Contrôle des doublons
- Sélection des variables utiles
- Création du ROI
- Création des classes de succès et de rentabilité à partir des quantiles
- Encodage du genre

### 6.2 Analyse exploratoire
L'analyse exploratoire portera sur :
- la distribution du revenu ;
- la distribution du ROI ;
- les statistiques descriptives ;
- les corrélations entre les variables numériques ;
- les comparaisons par genre ;
- les boxplots par classe de succès ;
- les boxplots par classe de rentabilité.

### 6.3 Modélisation
Deux tâches de classification seront étudiées :
1. **Prédiction du succès commercial**
2. **Prédiction de la rentabilité**

Pour chacune :
- **Modèle 1 : Gaussian Naive Bayes**
- **Modèle 2 : Random Forest**

### 6.4 Évaluation
Les modèles seront comparés à l'aide de :
- Accuracy
- Precision
- Recall
- F1-score
- Matrice de confusion

## 7. Structure du notebook
### Partie A — Chargement et compréhension des données
- Import des bibliothèques
- Chargement du CSV
- Aperçu du dataset
- Types de variables

### Partie B — Préparation
- Création du ROI
- Création de `Success_Class`
- Création de `Profitability_Class`
- Encodage de `Genre`
- Séparation des ensembles d'entraînement et de test

### Partie C — Analyse exploratoire
- Histogrammes
- Boxplots
- Heatmap des corrélations
- Analyse par genre
- Analyse par classes de succès et de rentabilité

### Partie D — Modélisation
- Gaussian Naive Bayes
- Random Forest

### Partie E — Comparaison des modèles
- Tableau de synthèse
- Commentaire des performances
- Interprétation des erreurs

### Partie F — Conclusion
- Résumé des résultats
- Limites
- Perspectives

## 8. Structure de l'application Streamlit
### Onglet 1 — Présentation
- contexte ;
- problématique ;
- objectifs ;
- variables utilisées.

### Onglet 2 — Exploration des données
- aperçu du dataset ;
- distributions ;
- statistiques descriptives ;
- analyse par genre.

### Onglet 3 — Analyse du succès
- distribution des classes ;
- visualisations ;
- métriques des modèles.

### Onglet 4 — Analyse de la rentabilité
- distribution des classes ;
- visualisations ;
- métriques des modèles.

### Onglet 5 — Prédiction interactive
L'utilisateur renseigne :
- budget ;
- nombre d'écrans ;
- note ;
- votes ;
- genre.

L'application retourne :
- la classe de succès prédite ;
- la classe de rentabilité prédite ;
- les probabilités associées ;
- le modèle utilisé.

## 9. Discussion attendue
Dans la discussion, il faudra répondre à des questions telles que :
- Le budget a-t-il réellement un effet positif sur le succès ?
- Le nombre d'écrans est-il un facteur plus fort que la note ?
- Quels genres sont les plus rentables ?
- Le modèle Random Forest réduit-il les erreurs sur les classes intermédiaires ?
- Le succès commercial et la rentabilité conduisent-ils aux mêmes conclusions ?

## 10. Conclusion générale
Le projet doit montrer comment des indicateurs de production, de diffusion et de réception permettent à la fois de **comprendre** les films à succès et d'**estimer** le potentiel de nouveaux films. La comparaison entre un modèle simple (GaussianNB) et un modèle plus flexible (Random Forest) donnera une base méthodologique solide.
