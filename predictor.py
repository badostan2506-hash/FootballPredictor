"""
Module pour prédire les résultats de matchs de football.
"""

import math
from typing import Dict, Tuple
from team import Team


class FootballPredictor:
    """Classe pour prédire les résultats de matchs de football."""
    
    def __init__(self):
        """Initialise le prédicteur."""
        self.teams = {}
    
    def add_team(self, team: Team):
        """
        Ajoute une équipe au prédicteur.
        
        Args:
            team (Team): L'équipe à ajouter
        """
        self.teams[team.name] = team
    
    def predict_match(self, home_team_name: str, away_team_name: str) -> Dict:
        """
        Prédit le résultat d'un match entre deux équipes.
        
        Args:
            home_team_name (str): Nom de l'équipe à domicile
            away_team_name (str): Nom de l'équipe à l'extérieur
            
        Returns:
            Dict: Dictionnaire contenant les probabilités et la prédiction
            
        Raises:
            ValueError: Si une des équipes n'existe pas
        """
        if home_team_name not in self.teams:
            raise ValueError(f"Équipe '{home_team_name}' non trouvée")
        if away_team_name not in self.teams:
            raise ValueError(f"Équipe '{away_team_name}' non trouvée")
        
        home_team = self.teams[home_team_name]
        away_team = self.teams[away_team_name]
        
        # Calcul des forces relatives des équipes
        home_strength = self._calculate_team_strength(home_team, is_home=True)
        away_strength = self._calculate_team_strength(away_team, is_home=False)
        
        # Calcul des probabilités
        total_strength = home_strength + away_strength
        if total_strength == 0:
            # Cas où les deux équipes n'ont pas de données
            home_win_prob = 0.4  # Légère favorisation de l'équipe à domicile
            away_win_prob = 0.3
            draw_prob = 0.3
        else:
            # Probabilité basée sur la force relative
            strength_ratio = home_strength / total_strength
            
            # Ajustement pour tenir compte de l'avantage du terrain
            home_advantage = 0.1
            adjusted_ratio = min(max(strength_ratio + home_advantage, 0.1), 0.9)
            
            # Distribution des probabilités
            if adjusted_ratio > 0.6:
                home_win_prob = 0.3 + (adjusted_ratio - 0.6) * 1.5
                away_win_prob = max(0.1, 0.1 + (0.6 - adjusted_ratio) * 0.5)
            elif adjusted_ratio < 0.4:
                home_win_prob = max(0.1, 0.1 + adjusted_ratio * 0.5)
                away_win_prob = 0.3 + (0.4 - adjusted_ratio) * 1.5
            else:
                home_win_prob = 0.2 + adjusted_ratio * 0.5
                away_win_prob = 0.2 + (1 - adjusted_ratio) * 0.5
            
            # S'assurer que les probabilités sont positives
            home_win_prob = max(0.05, home_win_prob)
            away_win_prob = max(0.05, away_win_prob)
            
            draw_prob = max(0.1, 1 - home_win_prob - away_win_prob)
            
            # Normalisation pour s'assurer que la somme = 1
            total_prob = home_win_prob + away_win_prob + draw_prob
            home_win_prob /= total_prob
            away_win_prob /= total_prob
            draw_prob /= total_prob
        
        # Détermination de la prédiction
        if home_win_prob > away_win_prob and home_win_prob > draw_prob:
            prediction = "Victoire domicile"
            confidence = home_win_prob
        elif away_win_prob > home_win_prob and away_win_prob > draw_prob:
            prediction = "Victoire extérieur"
            confidence = away_win_prob
        else:
            prediction = "Match nul"
            confidence = draw_prob
        
        # Prédiction du score approximatif
        home_goals = self._predict_goals(home_team, away_team, is_home=True)
        away_goals = self._predict_goals(away_team, home_team, is_home=False)
        
        return {
            "équipe_domicile": home_team_name,
            "équipe_extérieur": away_team_name,
            "prédiction": prediction,
            "confiance": round(confidence * 100, 1),
            "probabilités": {
                "victoire_domicile": round(home_win_prob * 100, 1),
                "match_nul": round(draw_prob * 100, 1),
                "victoire_extérieur": round(away_win_prob * 100, 1)
            },
            "score_prédit": f"{home_goals}-{away_goals}",
            "analyse": {
                "force_domicile": round(home_strength, 2),
                "force_extérieur": round(away_strength, 2),
                "forme_domicile": round(home_team.get_form_score(), 1),
                "forme_extérieur": round(away_team.get_form_score(), 1)
            }
        }
    
    def _calculate_team_strength(self, team: Team, is_home: bool) -> float:
        """
        Calcule la force d'une équipe pour un match.
        
        Args:
            team (Team): L'équipe
            is_home (bool): True si l'équipe joue à domicile
            
        Returns:
            float: Force de l'équipe
        """
        if team.matches_played == 0:
            return 50.0  # Force neutre si pas de données
        
        # Composantes de la force
        form_score = team.get_form_score()
        attacking_strength = min(team.goals_per_game * 20, 40)
        defensive_strength = max(40 - team.goals_conceded_per_game * 15, 0)
        
        # Bonus pour l'équipe à domicile
        home_bonus = 5 if is_home else 0
        
        total_strength = form_score * 0.4 + attacking_strength * 0.3 + defensive_strength * 0.3 + home_bonus
        return total_strength
    
    def _predict_goals(self, team: Team, opponent: Team, is_home: bool) -> int:
        """
        Prédit le nombre de buts qu'une équipe va marquer.
        
        Args:
            team (Team): L'équipe qui attaque
            opponent (Team): L'équipe qui défend
            is_home (bool): True si l'équipe attaque à domicile
            
        Returns:
            int: Nombre de buts prédits
        """
        if team.matches_played == 0:
            base_goals = 1.0
        else:
            base_goals = team.goals_per_game
        
        # Ajustement basé sur la défense adverse
        if opponent.matches_played > 0:
            defensive_factor = opponent.goals_conceded_per_game / 2.0
            adjusted_goals = (base_goals + defensive_factor) / 2
        else:
            adjusted_goals = base_goals
        
        # Bonus domicile
        if is_home:
            adjusted_goals *= 1.1
        
        # Arrondi au nombre entier le plus proche, minimum 0
        return max(0, round(adjusted_goals))
    
    def get_team_ranking(self) -> list:
        """
        Retourne le classement des équipes par ordre de force.
        
        Returns:
            list: Liste des équipes triées par force décroissante
        """
        team_strengths = []
        for team in self.teams.values():
            strength = self._calculate_team_strength(team, is_home=False)
            team_strengths.append((team, strength))
        
        team_strengths.sort(key=lambda x: x[1], reverse=True)
        return [(team.name, strength) for team, strength in team_strengths]