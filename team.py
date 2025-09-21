"""
Module pour représenter les statistiques d'une équipe de football.
"""

class Team:
    """Classe représentant une équipe de football avec ses statistiques."""
    
    def __init__(self, name, matches_played=0, wins=0, draws=0, losses=0, 
                 goals_for=0, goals_against=0):
        """
        Initialise une équipe avec ses statistiques.
        
        Args:
            name (str): Nom de l'équipe
            matches_played (int): Nombre de matchs joués
            wins (int): Nombre de victoires
            draws (int): Nombre de matchs nuls
            losses (int): Nombre de défaites
            goals_for (int): Buts marqués
            goals_against (int): Buts encaissés
        """
        self.name = name
        self.matches_played = matches_played
        self.wins = wins
        self.draws = draws
        self.losses = losses
        self.goals_for = goals_for
        self.goals_against = goals_against
    
    @property
    def points(self):
        """Calcule le nombre de points (3 pour victoire, 1 pour nul)."""
        return self.wins * 3 + self.draws * 1
    
    @property
    def goal_difference(self):
        """Calcule la différence de buts."""
        return self.goals_for - self.goals_against
    
    @property
    def win_rate(self):
        """Calcule le pourcentage de victoires."""
        if self.matches_played == 0:
            return 0.0
        return self.wins / self.matches_played
    
    @property
    def points_per_game(self):
        """Calcule la moyenne de points par match."""
        if self.matches_played == 0:
            return 0.0
        return self.points / self.matches_played
    
    @property
    def goals_per_game(self):
        """Calcule la moyenne de buts marqués par match."""
        if self.matches_played == 0:
            return 0.0
        return self.goals_for / self.matches_played
    
    @property
    def goals_conceded_per_game(self):
        """Calcule la moyenne de buts encaissés par match."""
        if self.matches_played == 0:
            return 0.0
        return self.goals_against / self.matches_played
    
    def get_form_score(self):
        """
        Calcule un score de forme basé sur les statistiques de l'équipe.
        
        Returns:
            float: Score de forme entre 0 et 100
        """
        if self.matches_played == 0:
            return 50.0  # Score neutre si pas de données
        
        # Facteurs de performance
        win_factor = self.win_rate * 40  # Max 40 points
        goal_factor = min(self.goals_per_game * 10, 30)  # Max 30 points
        defense_factor = max(20 - self.goals_conceded_per_game * 10, 0)  # Max 20 points
        goal_diff_factor = min(max(self.goal_difference, -10), 10)  # Entre -10 et 10 points
        
        total_score = win_factor + goal_factor + defense_factor + goal_diff_factor
        return min(max(total_score, 0), 100)  # Entre 0 et 100
    
    def __str__(self):
        return f"{self.name} - Points: {self.points}, Forme: {self.get_form_score():.1f}"
    
    def __repr__(self):
        return f"Team('{self.name}', matches={self.matches_played}, wins={self.wins})"