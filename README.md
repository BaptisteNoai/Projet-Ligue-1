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
Clean/
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

## Technologies

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Jupyter
