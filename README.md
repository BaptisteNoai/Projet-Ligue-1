# Projet prédiction Ligue 1

Prédiction des résultats de Ligue 1 pour la saison 2025-2026.

Projet d'algorithmes d'apprentissage réalisé par **Baptiste Noailhac** et **Seydoux**.

## Objectif

Prédire l'issue de chaque match de Ligue 1 de la saison 2025-2026 à partir des données historiques (matchs depuis 2013, joueurs, valeurs marchandes, événements de match) :

| Classe | Signification |
|-------:|---------------|
| `1`  | Victoire de l'équipe à domicile |
| `0`  | Match nul |
| `-1` | Victoire de l'équipe à l'extérieur |

## Démarche

1. **Exploration des données** : distribution des résultats (44 % victoires domicile, 26 % nuls, 30 % victoires extérieur).
2. **Feature engineering** : agrégation des statistiques joueurs, des événements de match et des valeurs marchandes par équipe.
3. **Sélection de variables** via l'importance des features d'une Random Forest.
4. **Modélisation** avec scikit-learn :
   - Baseline (`DummyClassifier`)
   - Arbre de décision
   - Random Forest
   - SVM (plusieurs noyaux)
   - ACP + Random Forest
5. **Évaluation** : accuracy, F1 macro, validation croisée 5-fold.
6. **Prédictions** finales sur les matchs 2025-2026 (`predictions.csv`).

## Structure

```
.
├── README.md
├── .gitignore
└── Clean/
    ├── projet_ligue1_final_Seydoux_Noailhac.ipynb   # Notebook principal
    ├── explication.txt                               # Énoncé du projet
    ├── matchs_2013_2024.csv                          # Historique des matchs
    ├── match_2025.csv                                # Matchs à prédire
    ├── clubs_fr.csv                                  # Infos clubs
    ├── player_valuation_before_season.csv            # Valeurs marchandes
    ├── player_appearance.csv                         # Stats joueurs par match
    ├── game_lineups.csv                              # Compositions d'équipes
    ├── game_events_before2025.csv                    # Événements de match
    ├── sample_results.csv                            # Exemple de format attendu
    └── predictions.csv                               # Prédictions finales
```

## Lancer le projet

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Clean/projet_ligue1_final_Seydoux_Noailhac.ipynb
```

## Compétences techniques

### Langage & outils

- Python
- Jupyter Notebook
- scikit-learn (`model_selection`, `preprocessing`, `tree`, `ensemble`, `svm`, `decomposition`, `metrics`, `dummy`)

### Manipulation de données

- pandas : lecture de CSV, conversion de dates (`to_datetime`), filtres booléens, `groupby` / `unstack`, `concat`, `value_counts`
- NumPy : calculs vectorisés, moyennes ignorant les valeurs manquantes (`nanmean`), variance cumulée
- Agrégation de données joueurs au niveau club : valeur marchande de l'effectif (dernière valuation de chaque joueur sur une fenêtre glissante de 18 mois)
- Mise en cache des valeurs marchandes par couple (saison, club)
- Imputation des valeurs manquantes par la médiane

### Visualisation

- Matplotlib : diagrammes en barres (verticales, horizontales, empilées), camembert, histogrammes, boîtes à moustaches, courbes
- Seaborn : heatmap de matrice de corrélation (triangle masqué)
- Matrices de confusion (`ConfusionMatrixDisplay`)
- Courbes d'accuracy entraînement / test selon la profondeur de l'arbre
- Courbe de variance expliquée cumulée de l'ACP

### Modèles de ML

- Classification multiclasse (victoire domicile / nul / victoire extérieur)
- Baseline `DummyClassifier` (classe majoritaire)
- Arbre de décision (profondeur maximale de 1 à 15, `min_samples_leaf`)
- Random Forest (`n_estimators`, `max_depth`, `min_samples_leaf`)
- SVM (noyaux linéaire, RBF avec C=1 et C=5, polynomial de degré 3)
- ACP (composantes expliquant 90 % de la variance) suivie d'une Random Forest

### Évaluation & validation

- Découpage train/test 75/25 stratifié
- Validation croisée 5-fold (`cross_val_score`)
- Accuracy, F1 macro, `classification_report` (précision, rappel, F1 par classe)
- Matrices de confusion
- Comparaison systématique à la baseline
- Analyse du sur-apprentissage selon la profondeur de l'arbre

### Méthodologie

- Analyse exploratoire : distribution des classes, stabilité de l'avantage domicile par saison, buts marqués, taux de victoire par club
- Analyse de corrélation pour repérer les features redondantes et privilégier les features différentielles (différence, ratio)
- Feature engineering temporel sans fuite de données : forme sur 5 et 10 matchs, face-à-face, moyennes de buts marqués et concédés, calculés uniquement sur les matchs antérieurs à la date du match ; valeur marchande calculée avant le début de saison
- Sélection de features par importance de Gini (Random Forest, seuil de 4 %) et comparaison des performances selon le nombre de features retenues
- Gestion du déséquilibre des classes : `class_weight='balanced'`, stratification, F1 macro
- Standardisation (`StandardScaler`) ajustée sur le seul jeu d'entraînement avant SVM et ACP
- Prédiction séquentielle : matchs 2025-2026 prédits dans l'ordre chronologique, chaque prédiction étant réinjectée dans l'historique pour mettre à jour les features des matchs suivants
- Reproductibilité (`random_state` fixé)
