"""
Module principal pour démontrer l'utilisation du système de prédiction de football.
"""

from team import Team
from predictor import FootballPredictor


def create_sample_teams():
    """
    Crée des équipes d'exemple avec des statistiques réalistes.
    
    Returns:
        list: Liste d'équipes avec leurs statistiques
    """
    teams = [
        Team("Paris Saint-Germain", 20, 15, 3, 2, 45, 15),
        Team("Olympique de Marseille", 20, 12, 5, 3, 38, 22),
        Team("AS Monaco", 20, 11, 4, 5, 35, 25),
        Team("OGC Nice", 20, 10, 6, 4, 32, 28),
        Team("Stade Rennais", 20, 9, 7, 4, 30, 26),
        Team("Olympique Lyonnais", 20, 8, 8, 4, 28, 24),
        Team("RC Lens", 20, 8, 6, 6, 26, 24),
        Team("LOSC Lille", 20, 7, 9, 4, 25, 23),
        Team("Montpellier HSC", 20, 6, 8, 6, 22, 26),
        Team("FC Nantes", 20, 5, 10, 5, 20, 25)
    ]
    return teams


def demonstrate_predictions():
    """Démontre l'utilisation du système de prédiction."""
    print("=" * 60)
    print("SYSTÈME DE PRÉDICTION DE MATCHS DE FOOTBALL")
    print("=" * 60)
    
    # Créer le prédicteur et ajouter les équipes
    predictor = FootballPredictor()
    teams = create_sample_teams()
    
    for team in teams:
        predictor.add_team(team)
    
    print("\n📊 STATISTIQUES DES ÉQUIPES:")
    print("-" * 40)
    for team in teams[:5]:  # Afficher les 5 premières équipes
        print(f"{team.name:25} | Matchs: {team.matches_played:2} | "
              f"V-N-D: {team.wins:2}-{team.draws:2}-{team.losses:2} | "
              f"Buts: {team.goals_for:2}-{team.goals_against:2} | "
              f"Forme: {team.get_form_score():5.1f}")
    
    print("\n🏆 CLASSEMENT PAR FORCE:")
    print("-" * 40)
    ranking = predictor.get_team_ranking()
    for i, (team_name, strength) in enumerate(ranking[:8], 1):
        print(f"{i:2}. {team_name:25} | Force: {strength:5.1f}")
    
    print("\n⚽ PRÉDICTIONS DE MATCHS:")
    print("-" * 40)
    
    # Quelques matchs d'exemple
    matches = [
        ("Paris Saint-Germain", "Olympique de Marseille"),
        ("AS Monaco", "OGC Nice"),
        ("Stade Rennais", "Olympique Lyonnais"),
        ("RC Lens", "LOSC Lille")
    ]
    
    for home_team, away_team in matches:
        print(f"\n🏟️  {home_team} vs {away_team}")
        try:
            result = predictor.predict_match(home_team, away_team)
            
            print(f"   Prédiction: {result['prédiction']}")
            print(f"   Confiance: {result['confiance']}%")
            print(f"   Score prédit: {result['score_prédit']}")
            print(f"   Probabilités:")
            print(f"     - Victoire domicile: {result['probabilités']['victoire_domicile']}%")
            print(f"     - Match nul: {result['probabilités']['match_nul']}%")
            print(f"     - Victoire extérieur: {result['probabilités']['victoire_extérieur']}%")
            print(f"   Analyse:")
            print(f"     - Forme domicile: {result['analyse']['forme_domicile']}")
            print(f"     - Forme extérieur: {result['analyse']['forme_extérieur']}")
            
        except ValueError as e:
            print(f"   Erreur: {e}")


def interactive_prediction():
    """Mode interactif pour faire des prédictions personnalisées."""
    print("\n" + "=" * 60)
    print("MODE INTERACTIF - PRÉDICTION PERSONNALISÉE")
    print("=" * 60)
    
    predictor = FootballPredictor()
    teams = create_sample_teams()
    
    for team in teams:
        predictor.add_team(team)
    
    print("\nÉquipes disponibles:")
    for i, team in enumerate(teams, 1):
        print(f"{i:2}. {team.name}")
    
    try:
        print("\nChoisissez les équipes pour le match:")
        home_choice = int(input("Équipe à domicile (numéro): ")) - 1
        away_choice = int(input("Équipe à l'extérieur (numéro): ")) - 1
        
        if 0 <= home_choice < len(teams) and 0 <= away_choice < len(teams):
            home_team = teams[home_choice].name
            away_team = teams[away_choice].name
            
            print(f"\n🏟️  Match: {home_team} vs {away_team}")
            result = predictor.predict_match(home_team, away_team)
            
            print(f"\n🎯 Résultat de la prédiction:")
            print(f"   Prédiction: {result['prédiction']}")
            print(f"   Confiance: {result['confiance']}%")
            print(f"   Score prédit: {result['score_prédit']}")
            print(f"\n📊 Probabilités détaillées:")
            for outcome, prob in result['probabilités'].items():
                print(f"   {outcome.replace('_', ' ').title()}: {prob}%")
            
        else:
            print("Choix invalide.")
            
    except (ValueError, KeyboardInterrupt):
        print("\nAu revoir!")


if __name__ == "__main__":
    # Démonstration du système
    demonstrate_predictions()
    
    # Mode interactif (commenté pour éviter l'input en mode automatique)
    # interactive_prediction()