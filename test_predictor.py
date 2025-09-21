"""
Tests unitaires pour le système de prédiction de football.
"""

import unittest
from team import Team
from predictor import FootballPredictor


class TestTeam(unittest.TestCase):
    """Tests pour la classe Team."""
    
    def setUp(self):
        """Configuration des tests."""
        self.team = Team("Test FC", 10, 6, 2, 2, 20, 10)
    
    def test_team_creation(self):
        """Test la création d'une équipe."""
        self.assertEqual(self.team.name, "Test FC")
        self.assertEqual(self.team.matches_played, 10)
        self.assertEqual(self.team.wins, 6)
        self.assertEqual(self.team.draws, 2)
        self.assertEqual(self.team.losses, 2)
    
    def test_points_calculation(self):
        """Test le calcul des points."""
        # 6 victoires * 3 + 2 nuls * 1 = 20 points
        self.assertEqual(self.team.points, 20)
    
    def test_goal_difference(self):
        """Test le calcul de la différence de buts."""
        self.assertEqual(self.team.goal_difference, 10)  # 20 - 10
    
    def test_win_rate(self):
        """Test le calcul du taux de victoire."""
        self.assertEqual(self.team.win_rate, 0.6)  # 6/10
    
    def test_points_per_game(self):
        """Test le calcul de la moyenne de points par match."""
        self.assertEqual(self.team.points_per_game, 2.0)  # 20/10
    
    def test_goals_per_game(self):
        """Test le calcul de la moyenne de buts par match."""
        self.assertEqual(self.team.goals_per_game, 2.0)  # 20/10
    
    def test_form_score(self):
        """Test le calcul du score de forme."""
        form_score = self.team.get_form_score()
        self.assertIsInstance(form_score, float)
        self.assertGreaterEqual(form_score, 0)
        self.assertLessEqual(form_score, 100)
    
    def test_empty_team_stats(self):
        """Test une équipe sans statistiques."""
        empty_team = Team("Empty FC")
        self.assertEqual(empty_team.win_rate, 0.0)
        self.assertEqual(empty_team.points_per_game, 0.0)
        self.assertEqual(empty_team.get_form_score(), 50.0)


class TestFootballPredictor(unittest.TestCase):
    """Tests pour la classe FootballPredictor."""
    
    def setUp(self):
        """Configuration des tests."""
        self.predictor = FootballPredictor()
        self.team1 = Team("Strong FC", 10, 8, 1, 1, 25, 8)
        self.team2 = Team("Weak FC", 10, 2, 3, 5, 10, 20)
        self.predictor.add_team(self.team1)
        self.predictor.add_team(self.team2)
    
    def test_add_team(self):
        """Test l'ajout d'équipes."""
        self.assertIn("Strong FC", self.predictor.teams)
        self.assertIn("Weak FC", self.predictor.teams)
    
    def test_predict_match_structure(self):
        """Test la structure du résultat de prédiction."""
        result = self.predictor.predict_match("Strong FC", "Weak FC")
        
        # Vérifier les clés principales
        required_keys = [
            "équipe_domicile", "équipe_extérieur", "prédiction", 
            "confiance", "probabilités", "score_prédit", "analyse"
        ]
        for key in required_keys:
            self.assertIn(key, result)
        
        # Vérifier les probabilités
        probabilities = result["probabilités"]
        prob_keys = ["victoire_domicile", "match_nul", "victoire_extérieur"]
        for key in prob_keys:
            self.assertIn(key, probabilities)
            self.assertIsInstance(probabilities[key], (int, float))
            self.assertGreaterEqual(probabilities[key], 0)
            self.assertLessEqual(probabilities[key], 100)
        
        # Vérifier que les probabilités somment à 100% (avec tolérance)
        total_prob = sum(probabilities.values())
        self.assertAlmostEqual(total_prob, 100.0, delta=0.1)
    
    def test_predict_match_logic(self):
        """Test la logique de prédiction."""
        result = self.predictor.predict_match("Strong FC", "Weak FC")
        
        # L'équipe forte à domicile devrait avoir plus de chances de gagner
        probs = result["probabilités"]
        self.assertGreater(probs["victoire_domicile"], probs["victoire_extérieur"])
    
    def test_predict_nonexistent_team(self):
        """Test la prédiction avec une équipe inexistante."""
        with self.assertRaises(ValueError):
            self.predictor.predict_match("Strong FC", "Nonexistent FC")
        
        with self.assertRaises(ValueError):
            self.predictor.predict_match("Nonexistent FC", "Weak FC")
    
    def test_team_ranking(self):
        """Test le classement des équipes."""
        ranking = self.predictor.get_team_ranking()
        
        self.assertEqual(len(ranking), 2)
        self.assertIsInstance(ranking, list)
        
        # Vérifier que Strong FC est mieux classé que Weak FC
        team_names = [team_name for team_name, _ in ranking]
        strong_position = team_names.index("Strong FC")
        weak_position = team_names.index("Weak FC")
        self.assertLess(strong_position, weak_position)
    
    def test_goal_prediction(self):
        """Test la prédiction de buts."""
        home_goals = self.predictor._predict_goals(self.team1, self.team2, True)
        away_goals = self.predictor._predict_goals(self.team2, self.team1, False)
        
        self.assertIsInstance(home_goals, int)
        self.assertIsInstance(away_goals, int)
        self.assertGreaterEqual(home_goals, 0)
        self.assertGreaterEqual(away_goals, 0)
        
        # L'équipe forte devrait marquer plus de buts
        self.assertGreaterEqual(home_goals, away_goals)


class TestIntegration(unittest.TestCase):
    """Tests d'intégration du système complet."""
    
    def test_complete_workflow(self):
        """Test un workflow complet."""
        # Créer des équipes réalistes
        teams = [
            Team("Barcelona", 20, 15, 3, 2, 50, 20),
            Team("Real Madrid", 20, 14, 4, 2, 48, 22),
            Team("Atletico Madrid", 20, 12, 6, 2, 35, 18)
        ]
        
        # Créer le prédicteur
        predictor = FootballPredictor()
        for team in teams:
            predictor.add_team(team)
        
        # Faire des prédictions
        result1 = predictor.predict_match("Barcelona", "Real Madrid")
        result2 = predictor.predict_match("Real Madrid", "Atletico Madrid")
        
        # Vérifier que les résultats sont cohérents
        self.assertIsInstance(result1["confiance"], (int, float))
        self.assertIsInstance(result2["confiance"], (int, float))
        
        # Vérifier le classement
        ranking = predictor.get_team_ranking()
        self.assertEqual(len(ranking), 3)


def run_tests():
    """Lance tous les tests."""
    # Créer une suite de tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Ajouter les classes de test
    suite.addTests(loader.loadTestsFromTestCase(TestTeam))
    suite.addTests(loader.loadTestsFromTestCase(TestFootballPredictor))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Lancer les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Lancement des tests du système de prédiction...")
    print("=" * 60)
    success = run_tests()
    print("=" * 60)
    if success:
        print("✅ Tous les tests sont passés avec succès!")
    else:
        print("❌ Certains tests ont échoué.")