"""Main prediction engine for football match predictions."""

import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from config import Config
from api_client import FootballAPIClient
from data_processor import FootballDataProcessor
from prediction_model import FootballPredictionModel

class FootballPredictor:
    """Main class for football match prediction system."""
    
    def __init__(self):
        self.api_client = FootballAPIClient()
        self.data_processor = FootballDataProcessor(self.api_client)
        self.model = FootballPredictionModel()
        self.predictions = []
        
    def initialize(self) -> bool:
        """Initialize the prediction system."""
        print("Initializing Football Predictor...")
        
        # Validate API connection
        if not self.api_client.validate_connection():
            print("Failed to connect to football API")
            return False
        
        print("API connection validated successfully")
        return True
    
    def train_model_from_league(self, league_id: int, season: int, max_teams: int = 10) -> bool:
        """Train the model using historical data from a league."""
        print(f"Training model with data from league {league_id}, season {season}")
        
        try:
            # Get teams from the league
            teams_data = self.api_client.get_teams(league_id, season)
            if not teams_data:
                print("No teams found for the specified league and season")
                return False
            
            teams = teams_data[:max_teams]  # Limit teams to avoid too many API calls
            print(f"Found {len(teams)} teams")
            
            # Get historical fixtures
            print("Fetching historical fixtures...")
            fixtures = self.api_client.get_fixtures(league_id, season, status="FT")
            
            if len(fixtures) < Config.MIN_MATCHES_FOR_PREDICTION:
                print(f"Insufficient fixtures found ({len(fixtures)} < {Config.MIN_MATCHES_FOR_PREDICTION})")
                return False
            
            print(f"Found {len(fixtures)} completed fixtures")
            
            # Prepare training data with team statistics
            print("Preparing training data...")
            training_data = []
            
            for fixture in fixtures[:50]:  # Limit to avoid too many API calls for demo
                try:
                    teams_info = fixture.get('teams', {})
                    home_team_id = teams_info.get('home', {}).get('id')
                    away_team_id = teams_info.get('away', {}).get('id')
                    
                    if not home_team_id or not away_team_id:
                        continue
                    
                    # Get match data with statistics
                    match_data = self.data_processor.prepare_match_data(
                        home_team_id, away_team_id, league_id, season
                    )
                    
                    if match_data:
                        # Add actual outcome
                        goals = fixture.get('goals', {})
                        home_goals = goals.get('home', 0) or 0
                        away_goals = goals.get('away', 0) or 0
                        
                        if home_goals > away_goals:
                            outcome = 2  # Home win
                        elif home_goals < away_goals:
                            outcome = 0  # Away win
                        else:
                            outcome = 1  # Draw
                        
                        match_data['outcome'] = outcome
                        training_data.append(match_data)
                        
                except Exception as e:
                    print(f"Error processing fixture: {e}")
                    continue
            
            if len(training_data) < 10:
                print(f"Insufficient training data prepared ({len(training_data)} samples)")
                return False
            
            print(f"Prepared {len(training_data)} training samples")
            
            # Train the model
            success = self.model.train_model(training_data)
            
            if success:
                # Save the trained model
                model_path = Config.get_output_path("football_model.pkl")
                self.model.save_model(model_path)
                print("Model training completed successfully")
            
            return success
            
        except Exception as e:
            print(f"Error during model training: {e}")
            return False
    
    def predict_match(self, team1_id: int, team2_id: int, league_id: int, season: int) -> Optional[Dict]:
        """Predict the outcome of a match between two teams."""
        try:
            print(f"Predicting match: Team {team1_id} vs Team {team2_id}")
            
            # Prepare match data
            match_data = self.data_processor.prepare_match_data(
                team1_id, team2_id, league_id, season
            )
            
            if not match_data:
                print("Could not prepare match data")
                return None
            
            # Make prediction
            prediction = self.model.predict_match(match_data)
            
            if prediction:
                # Get team names
                teams_data = self.api_client.get_teams(league_id, season)
                team1_name = "Unknown"
                team2_name = "Unknown"
                
                for team_info in teams_data:
                    team = team_info.get('team', {})
                    if team.get('id') == team1_id:
                        team1_name = team.get('name', f"Team {team1_id}")
                    elif team.get('id') == team2_id:
                        team2_name = team.get('name', f"Team {team2_id}")
                
                # Add team information to prediction
                prediction.update({
                    'team1_id': team1_id,
                    'team2_id': team2_id,
                    'team1_name': team1_name,
                    'team2_name': team2_name,
                    'league_id': league_id,
                    'season': season,
                    'prediction_date': datetime.now().isoformat()
                })
                
                self.predictions.append(prediction)
                
                print(f"Prediction: {prediction['predicted_outcome']} "
                      f"(Confidence: {prediction['confidence']:.2f})")
                
            return prediction
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def predict_upcoming_matches(self, league_id: int, season: int, max_matches: int = 10) -> List[Dict]:
        """Predict outcomes for upcoming matches in a league."""
        print(f"Predicting upcoming matches for league {league_id}")
        
        try:
            # Get upcoming fixtures
            upcoming_fixtures = self.api_client.get_fixtures(league_id, season, status="NS")
            
            if not upcoming_fixtures:
                print("No upcoming fixtures found")
                return []
            
            predictions = []
            processed = 0
            
            for fixture in upcoming_fixtures:
                if processed >= max_matches:
                    break
                
                try:
                    teams = fixture.get('teams', {})
                    home_team_id = teams.get('home', {}).get('id')
                    away_team_id = teams.get('away', {}).get('id')
                    
                    if not home_team_id or not away_team_id:
                        continue
                    
                    prediction = self.predict_match(home_team_id, away_team_id, league_id, season)
                    
                    if prediction:
                        # Add fixture information
                        fixture_info = fixture.get('fixture', {})
                        prediction.update({
                            'fixture_id': fixture_info.get('id'),
                            'match_date': fixture_info.get('date'),
                            'venue': fixture_info.get('venue', {}).get('name', 'Unknown')
                        })
                        predictions.append(prediction)
                        processed += 1
                
                except Exception as e:
                    print(f"Error predicting fixture: {e}")
                    continue
            
            print(f"Generated {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            print(f"Error predicting upcoming matches: {e}")
            return []
    
    def save_predictions_to_csv(self, filename: Optional[str] = None) -> bool:
        """Save all predictions to a CSV file."""
        if not self.predictions:
            print("No predictions to save")
            return False
        
        try:
            if filename is None:
                filename = Config.CSV_FILENAME
            
            filepath = Config.get_output_path(filename)
            
            # Convert predictions to DataFrame
            df = pd.DataFrame(self.predictions)
            
            # Reorder columns for better readability
            column_order = [
                'team1_name', 'team2_name', 'predicted_outcome',
                'confidence', 'prediction_date', 'match_date',
                'team1_id', 'team2_id', 'league_id', 'season'
            ]
            
            # Add columns that exist in the data
            available_columns = [col for col in column_order if col in df.columns]
            remaining_columns = [col for col in df.columns if col not in available_columns]
            final_columns = available_columns + remaining_columns
            
            df = df[final_columns]
            
            # Save to CSV
            df.to_csv(filepath, index=False, encoding='utf-8')
            
            print(f"Predictions saved to {filepath}")
            print(f"Total predictions: {len(df)}")
            
            return True
            
        except Exception as e:
            print(f"Error saving predictions to CSV: {e}")
            return False
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load a pre-trained model."""
        if model_path is None:
            model_path = Config.get_output_path("football_model.pkl")
        
        return self.model.load_model(model_path)
    
    def get_prediction_summary(self) -> Dict:
        """Get a summary of predictions made."""
        if not self.predictions:
            return {"total": 0}
        
        df = pd.DataFrame(self.predictions)
        
        summary = {
            "total": len(df),
            "outcomes": df['predicted_outcome'].value_counts().to_dict(),
            "average_confidence": df['confidence'].mean(),
            "high_confidence": len(df[df['confidence'] > Config.PREDICTION_CONFIDENCE_THRESHOLD])
        }
        
        return summary