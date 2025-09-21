# FootballPredictor

Modèle de prédiction qui s'appuie sur des statistiques de deux équipes pour prédire l'issue d'un match de football.

## 🎯 Fonctionnalités

- **Analyse des équipes** : Calcul automatique des statistiques de performance (forme, points par match, buts marqués/encaissés)
- **Prédiction de matchs** : Probabilités de victoire, match nul et défaite
- **Score prédit** : Estimation du score final basée sur les performances offensives et défensives
- **Classement** : Ranking des équipes par force relative
- **Avantage du terrain** : Prise en compte de l'avantage de jouer à domicile

## 🚀 Utilisation rapide

```python
from team import Team
from predictor import FootballPredictor

# Créer des équipes avec leurs statistiques
psg = Team("Paris Saint-Germain", 20, 15, 3, 2, 45, 15)
om = Team("Olympique de Marseille", 20, 12, 5, 3, 38, 22)

# Initialiser le prédicteur
predictor = FootballPredictor()
predictor.add_team(psg)
predictor.add_team(om)

# Prédire un match
result = predictor.predict_match("Paris Saint-Germain", "Olympique de Marseille")
print(f"Prédiction: {result['prédiction']}")
print(f"Confiance: {result['confiance']}%")
print(f"Score prédit: {result['score_prédit']}")
```

## 📊 Structure du projet

```
FootballPredictor/
├── team.py              # Classe Team pour les statistiques d'équipe
├── predictor.py         # Classe FootballPredictor pour les prédictions
├── main.py             # Démonstration avec des données d'exemple
├── test_predictor.py   # Tests unitaires
├── requirements.txt    # Dépendances (aucune externe)
└── README.md          # Cette documentation
```

## 🏗️ Classes principales

### Team
Représente une équipe avec ses statistiques de performance :
- Matchs joués, victoires, nuls, défaites
- Buts marqués et encaissés
- Calculs automatiques : points, différence de buts, taux de victoire, forme

### FootballPredictor
Moteur de prédiction qui analyse les équipes et prédit les résultats :
- Ajout d'équipes à la base de données
- Prédiction de matchs avec probabilités détaillées
- Classement des équipes par force
- Prise en compte de l'avantage du terrain

## 📈 Algorithme de prédiction

Le système utilise plusieurs métriques pour évaluer la force des équipes :

1. **Score de forme** (0-100) :
   - Taux de victoire (40% du score)
   - Performance offensive (30% du score)
   - Performance défensive (20% du score)
   - Différence de buts (10% du score)

2. **Force d'équipe** :
   - Score de forme pondéré
   - Force offensive et défensive
   - Bonus de +5 points pour l'équipe à domicile

3. **Probabilités de résultat** :
   - Calcul basé sur la force relative des équipes
   - Ajustement pour l'avantage du terrain
   - Distribution équilibrée entre victoire domicile/extérieur/nul

## 🧪 Tests

Lancer les tests unitaires :

```bash
python test_predictor.py
```

Les tests couvrent :
- Création et calculs des équipes
- Logique de prédiction
- Structure des résultats
- Gestion des erreurs
- Workflow complet

## 🎮 Démonstration

Voir le système en action :

```bash
python main.py
```

Cela affiche :
- Statistiques des équipes d'exemple
- Classement par force
- Prédictions de plusieurs matchs avec analyses détaillées

## 📋 Format des résultats

Une prédiction retourne un dictionnaire avec :

```python
{
    "équipe_domicile": "Paris Saint-Germain",
    "équipe_extérieur": "Olympique de Marseille", 
    "prédiction": "Victoire domicile",
    "confiance": 65.2,
    "probabilités": {
        "victoire_domicile": 65.2,
        "match_nul": 24.8,
        "victoire_extérieur": 10.0
    },
    "score_prédit": "2-1",
    "analyse": {
        "force_domicile": 85.6,
        "force_extérieur": 72.3,
        "forme_domicile": 75.0,
        "forme_extérieur": 62.0
    }
}
```

## 🔧 Configuration requise

- Python 3.6+
- Aucune dépendance externe (utilise uniquement les modules standard)

## 🎯 Exemples d'utilisation

### Créer une équipe personnalisée

```python
# Team(nom, matchs_joués, victoires, nuls, défaites, buts_pour, buts_contre)
mon_equipe = Team("Mon Club", 15, 8, 4, 3, 24, 18)
print(f"Forme de l'équipe: {mon_equipe.get_form_score():.1f}")
```

### Analyser un championnat

```python
predictor = FootballPredictor()

# Ajouter toutes les équipes du championnat
for equipe in mes_equipes:
    predictor.add_team(equipe)

# Obtenir le classement par force
classement = predictor.get_team_ranking()
for position, (nom, force) in enumerate(classement, 1):
    print(f"{position}. {nom} (Force: {force:.1f})")
```

## 🚀 Extensions possibles

Le système peut être étendu avec :
- Historique des confrontations directes
- Forme récente (derniers 5 matchs)
- Conditions météorologiques
- Blessures de joueurs clés
- Importance du match (championnat, coupe, etc.)
- Machine learning pour optimiser les pondérations

## 📝 Licence

Ce projet est libre d'utilisation pour des fins éducatives et non commerciales.