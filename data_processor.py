"""Data processing module for football statistics analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from api_client import FootballAPIClient

class FootballDataProcessor:
    """Process and analyze football statistics for prediction modeling."""
    
    def __init__(self, api_client: FootballAPIClient):
        self.api_client = api_client
    
    def extract_team_features(self, team_stats: Dict) -> Dict[str, float]:
        """Extract numerical features from team statistics."""
        if not team_stats:
            return self._get_default_features()
        
        fixtures = team_stats.get('fixtures', {})
        goals = team_stats.get('goals', {})
        
        features = {
            # Basic match statistics
            'matches_played': fixtures.get('played', {}).get('total', 0),
            'wins': fixtures.get('wins', {}).get('total', 0),
            'draws': fixtures.get('draws', {}).get('total', 0),
            'losses': fixtures.get('loses', {}).get('total', 0),
            
            # Goals statistics
            'goals_for': goals.get('for', {}).get('total', {}).get('total', 0),
            'goals_against': goals.get('against', {}).get('total', {}).get('total', 0),
            
            # Home/Away performance
            'home_wins': fixtures.get('wins', {}).get('home', 0),
            'away_wins': fixtures.get('wins', {}).get('away', 0),
            'home_goals_for': goals.get('for', {}).get('total', {}).get('home', 0),
            'away_goals_for': goals.get('for', {}).get('total', {}).get('away', 0),
            'home_goals_against': goals.get('against', {}).get('total', {}).get('home', 0),
            'away_goals_against': goals.get('against', {}).get('total', {}).get('away', 0),
        }
        
        # Calculate derived metrics
        matches_played = features['matches_played']
        if matches_played > 0:
            features['win_rate'] = features['wins'] / matches_played
            features['goals_per_match'] = features['goals_for'] / matches_played
            features['goals_conceded_per_match'] = features['goals_against'] / matches_played
            features['goal_difference'] = features['goals_for'] - features['goals_against']
            features['goal_difference_per_match'] = features['goal_difference'] / matches_played
        else:
            features.update({
                'win_rate': 0.0,
                'goals_per_match': 0.0,
                'goals_conceded_per_match': 0.0,
                'goal_difference': 0.0,
                'goal_difference_per_match': 0.0
            })
        
        return features
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when no statistics are available."""
        return {
            'matches_played': 0, 'wins': 0, 'draws': 0, 'losses': 0,
            'goals_for': 0, 'goals_against': 0, 'home_wins': 0, 'away_wins': 0,
            'home_goals_for': 0, 'away_goals_for': 0, 'home_goals_against': 0,
            'away_goals_against': 0, 'win_rate': 0.0, 'goals_per_match': 0.0,
            'goals_conceded_per_match': 0.0, 'goal_difference': 0.0,
            'goal_difference_per_match': 0.0
        }
    
    def analyze_h2h_history(self, h2h_matches: List[Dict], team1_id: int, team2_id: int) -> Dict[str, float]:
        """Analyze head-to-head match history between two teams."""
        if not h2h_matches:
            return {
                'h2h_total_matches': 0,
                'h2h_team1_wins': 0,
                'h2h_team2_wins': 0,
                'h2h_draws': 0,
                'h2h_team1_win_rate': 0.0,
                'h2h_avg_goals': 0.0
            }
        
        team1_wins = 0
        team2_wins = 0
        draws = 0
        total_goals = 0
        
        for match in h2h_matches:
            fixture = match.get('fixture', {})
            teams = match.get('teams', {})
            goals = match.get('goals', {})
            
            if fixture.get('status', {}).get('short') != 'FT':
                continue
                
            home_id = teams.get('home', {}).get('id')
            away_id = teams.get('away', {}).get('id')
            home_goals = goals.get('home', 0) or 0
            away_goals = goals.get('away', 0) or 0
            
            total_goals += home_goals + away_goals
            
            if home_goals > away_goals:
                if home_id == team1_id:
                    team1_wins += 1
                else:
                    team2_wins += 1
            elif away_goals > home_goals:
                if away_id == team1_id:
                    team1_wins += 1
                else:
                    team2_wins += 1
            else:
                draws += 1
        
        total_matches = team1_wins + team2_wins + draws
        
        return {
            'h2h_total_matches': total_matches,
            'h2h_team1_wins': team1_wins,
            'h2h_team2_wins': team2_wins,
            'h2h_draws': draws,
            'h2h_team1_win_rate': team1_wins / total_matches if total_matches > 0 else 0.0,
            'h2h_avg_goals': total_goals / total_matches if total_matches > 0 else 0.0
        }
    
    def calculate_form_features(self, recent_matches: List[Dict], team_id: int, num_matches: int = 5) -> Dict[str, float]:
        """Calculate recent form features for a team."""
        if not recent_matches:
            return {
                'recent_wins': 0,
                'recent_draws': 0,
                'recent_losses': 0,
                'recent_win_rate': 0.0,
                'recent_goals_for': 0,
                'recent_goals_against': 0,
                'recent_form_points': 0
            }
        
        # Sort matches by date (most recent first)
        sorted_matches = sorted(recent_matches, 
                              key=lambda x: x.get('fixture', {}).get('date', ''), 
                              reverse=True)
        
        # Take only the specified number of recent matches
        recent = sorted_matches[:num_matches]
        
        wins = draws = losses = 0
        goals_for = goals_against = 0
        
        for match in recent:
            teams = match.get('teams', {})
            goals = match.get('goals', {})
            fixture = match.get('fixture', {})
            
            if fixture.get('status', {}).get('short') != 'FT':
                continue
            
            home_id = teams.get('home', {}).get('id')
            away_id = teams.get('away', {}).get('id')
            home_goals = goals.get('home', 0) or 0
            away_goals = goals.get('away', 0) or 0
            
            if home_id == team_id:
                goals_for += home_goals
                goals_against += away_goals
                if home_goals > away_goals:
                    wins += 1
                elif home_goals == away_goals:
                    draws += 1
                else:
                    losses += 1
            elif away_id == team_id:
                goals_for += away_goals
                goals_against += home_goals
                if away_goals > home_goals:
                    wins += 1
                elif away_goals == home_goals:
                    draws += 1
                else:
                    losses += 1
        
        total_matches = wins + draws + losses
        form_points = wins * 3 + draws * 1  # Standard football points system
        
        return {
            'recent_wins': wins,
            'recent_draws': draws,
            'recent_losses': losses,
            'recent_win_rate': wins / total_matches if total_matches > 0 else 0.0,
            'recent_goals_for': goals_for,
            'recent_goals_against': goals_against,
            'recent_form_points': form_points
        }
    
    def prepare_match_data(self, team1_id: int, team2_id: int, league_id: int, season: int) -> Optional[Dict]:
        """Prepare comprehensive match data for prediction."""
        try:
            # Get team statistics
            team1_stats = self.api_client.get_team_statistics(team1_id, league_id, season)
            team2_stats = self.api_client.get_team_statistics(team2_id, league_id, season)
            
            # Extract basic features
            team1_features = self.extract_team_features(team1_stats)
            team2_features = self.extract_team_features(team2_stats)
            
            # Get head-to-head history
            h2h_matches = self.api_client.get_h2h_matches(team1_id, team2_id)
            h2h_features = self.analyze_h2h_history(h2h_matches, team1_id, team2_id)
            
            # Get recent fixtures for form analysis
            all_fixtures = self.api_client.get_fixtures(league_id, season, status="FT")
            
            # Filter fixtures for each team
            team1_recent = [f for f in all_fixtures 
                           if f.get('teams', {}).get('home', {}).get('id') == team1_id or 
                              f.get('teams', {}).get('away', {}).get('id') == team1_id]
            
            team2_recent = [f for f in all_fixtures 
                           if f.get('teams', {}).get('home', {}).get('id') == team2_id or 
                              f.get('teams', {}).get('away', {}).get('id') == team2_id]
            
            # Calculate form features
            team1_form = self.calculate_form_features(team1_recent, team1_id)
            team2_form = self.calculate_form_features(team2_recent, team2_id)
            
            # Combine all features
            match_data = {
                'team1_id': team1_id,
                'team2_id': team2_id,
                **{f'team1_{k}': v for k, v in team1_features.items()},
                **{f'team2_{k}': v for k, v in team2_features.items()},
                **{f'team1_{k}': v for k, v in team1_form.items()},
                **{f'team2_{k}': v for k, v in team2_form.items()},
                **h2h_features
            }
            
            return match_data
            
        except Exception as e:
            print(f"Error preparing match data: {e}")
            return None