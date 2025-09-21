# FootballPredictor

Modèle de prédiction robuste qui s'appuie sur les statistiques de l'API v3.football.api-sports.io pour prédire l'issue des matchs de football. Le système utilise des algorithmes d'apprentissage automatique pour analyser les performances des équipes et générer des prédictions fiables.

## Fonctionnalités

- 🏈 **Extraction de statistiques** via l'API v3.football.api-sports.io
- 🤖 **Modèle d'apprentissage automatique** basé sur Random Forest
- 📊 **Analyse complète** des performances d'équipe (forme récente, historique H2H, statistiques générales)
- 💾 **Export CSV** des prédictions
- 🔄 **Entraînement automatique** à partir de données historiques
- 🎯 **Prédictions avec niveau de confiance**

## Installation

1. Clonez le repository:
```bash
git clone https://github.com/badostan2506-hash/FootballPredictor.git
cd FootballPredictor
```

2. Installez les dépendances:
```bash
pip install -r requirements.txt
```

3. Configurez votre clé API:
```bash
cp .env.example .env
# Éditez .env et ajoutez votre clé API de api-football.com
```

## Configuration

Obtenez une clé API gratuite sur [api-football.com](https://rapidapi.com/api-sports/api/api-football/) et ajoutez-la dans le fichier `.env`:

```
FOOTBALL_API_KEY=votre_cle_api_ici
```

## Utilisation

### 1. Entraîner le modèle

Entraînez le modèle avec les données de la Ligue 1 française:
```bash
python main.py --train --league 61 --season 2023
```

### 2. Prédire les matchs à venir

Prédisez les prochains matchs de la ligue:
```bash
python main.py --predict --league 61 --season 2023 --max-matches 10
```

### 3. Prédire un match spécifique

Prédisez le résultat entre deux équipes:
```bash
python main.py --predict-match --team1 85 --team2 81 --league 61 --season 2023
```

### Options disponibles

- `--train`: Entraîner le modèle
- `--predict`: Prédire les matchs à venir
- `--predict-match`: Prédire un match spécifique
- `--league ID`: ID de la ligue (défaut: 61 pour Ligue 1)
- `--season YEAR`: Année de la saison (défaut: 2023)
- `--team1 ID`, `--team2 ID`: IDs des équipes pour un match spécifique
- `--max-matches N`: Nombre maximum de matchs à prédire
- `--output FILE`: Nom du fichier CSV de sortie

## IDs de ligues populaires

- **61**: Ligue 1 (France)
- **39**: Premier League (Angleterre)
- **140**: La Liga (Espagne)
- **135**: Serie A (Italie)
- **78**: Bundesliga (Allemagne)

## Structure du projet

```
FootballPredictor/
├── main.py              # Script principal
├── config.py            # Configuration et paramètres
├── api_client.py        # Client API pour v3.football.api-sports.io
├── data_processor.py    # Traitement des données statistiques
├── prediction_model.py  # Modèle d'apprentissage automatique
├── predictor.py         # Moteur de prédiction principal
├── requirements.txt     # Dépendances Python
├── .env.example         # Exemple de configuration
└── predictions/         # Dossier des résultats (créé automatiquement)
    ├── football_predictions.csv
    └── football_model.pkl
```

## Exemple de sortie CSV

Les prédictions sont sauvegardées dans un fichier CSV avec les colonnes suivantes:

```csv
team1_name,team2_name,predicted_outcome,confidence,prediction_date,match_date,team1_id,team2_id,league_id,season
Paris Saint Germain,Olympique Marseille,Home Win,0.75,2023-12-01T10:30:00,2023-12-03T20:00:00,85,81,61,2023
```

## Métriques utilisées

Le modèle analyse plusieurs métriques pour chaque équipe:

- **Performance générale**: Victoires, nuls, défaites, ratio de victoires
- **Statistiques de buts**: Buts marqués/encaissés par match
- **Forme récente**: Performance sur les 5 derniers matchs
- **Historique H2H**: Confrontations directes entre les équipes
- **Performance domicile/extérieur**: Statistiques spécifiques au lieu

## Exemples d'utilisation

Voir tous les exemples disponibles:
```bash
python main.py --examples
```

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à:

1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Ouvrir une Pull Request

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## Notes importantes

- Les prédictions sont basées sur des données statistiques et ne garantissent pas le résultat réel
- L'API a des limites de taux pour les comptes gratuits
- Les performances du modèle dépendent de la qualité et quantité des données d'entraînement
