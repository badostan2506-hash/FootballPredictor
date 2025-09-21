#!/usr/bin/env python3
"""
Test script for Football Predictor without requiring API key.
This script tests the core functionality with mock data.
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prediction_model import FootballPredictionModel
from data_processor import FootballDataProcessor
from config import Config

def test_prediction_model():
    """Test the prediction model with mock data."""
    print("Testing Football Prediction Model...")
    
    # Create mock training data
    mock_training_data = []
    
    for i in range(50):
        # Create synthetic team statistics
        team1_wins = 10 + i % 5
        team1_losses = 5 + i % 3
        team1_draws = 3 + i % 2
        team1_matches = team1_wins + team1_losses + team1_draws
        
        team2_wins = 8 + i % 4
        team2_losses = 6 + i % 3
        team2_draws = 4 + i % 2
        team2_matches = team2_wins + team2_losses + team2_draws
        
        match_data = {
            'team1_matches_played': team1_matches,
            'team1_wins': team1_wins,
            'team1_draws': team1_draws,
            'team1_losses': team1_losses,
            'team1_goals_for': team1_wins * 2 + team1_draws + i % 3,
            'team1_goals_against': team1_losses * 2 + i % 2,
            'team1_win_rate': team1_wins / team1_matches,
            'team1_goals_per_match': (team1_wins * 2 + team1_draws + i % 3) / team1_matches,
            'team1_goals_conceded_per_match': (team1_losses * 2 + i % 2) / team1_matches,
            'team1_goal_difference_per_match': 0.5 + i % 2,
            
            'team2_matches_played': team2_matches,
            'team2_wins': team2_wins,
            'team2_draws': team2_draws,
            'team2_losses': team2_losses,
            'team2_goals_for': team2_wins * 2 + team2_draws + i % 2,
            'team2_goals_against': team2_losses * 2 + i % 3,
            'team2_win_rate': team2_wins / team2_matches,
            'team2_goals_per_match': (team2_wins * 2 + team2_draws + i % 2) / team2_matches,
            'team2_goals_conceded_per_match': (team2_losses * 2 + i % 3) / team2_matches,
            'team2_goal_difference_per_match': 0.3 + i % 2,
            
            'team1_recent_wins': 3 + i % 2,
            'team1_recent_draws': 1,
            'team1_recent_losses': 1 + i % 2,
            'team1_recent_win_rate': 0.6 + (i % 3) * 0.1,
            'team1_recent_form_points': 9 + i % 3,
            
            'team2_recent_wins': 2 + i % 2,
            'team2_recent_draws': 2,
            'team2_recent_losses': 1 + i % 2,
            'team2_recent_win_rate': 0.4 + (i % 3) * 0.1,
            'team2_recent_form_points': 6 + i % 4,
            
            'h2h_total_matches': 5 + i % 3,
            'h2h_team1_wins': 2 + i % 2,
            'h2h_team2_wins': 1 + i % 2,
            'h2h_draws': 1,
            'h2h_team1_win_rate': 0.4 + (i % 3) * 0.1,
            'h2h_avg_goals': 2.5,
            
            # Determine outcome based on relative strength
            'outcome': 2 if team1_wins > team2_wins else (0 if team2_wins > team1_wins else 1)
        }
        
        mock_training_data.append(match_data)
    
    # Test model training
    model = FootballPredictionModel()
    success = model.train_model(mock_training_data)
    
    if success:
        print("✓ Model training successful")
        
        # Test prediction
        test_match = mock_training_data[0].copy()
        del test_match['outcome']  # Remove outcome for prediction
        
        prediction = model.predict_match(test_match)
        
        if prediction:
            print("✓ Prediction successful")
            print(f"  Predicted outcome: {prediction['predicted_outcome']}")
            print(f"  Confidence: {prediction['confidence']:.3f}")
            print(f"  Probabilities: {prediction['probabilities']}")
            
            # Test model saving/loading
            test_model_path = "/tmp/test_model.pkl"
            if model.save_model(test_model_path):
                print("✓ Model saving successful")
                
                # Load model
                new_model = FootballPredictionModel()
                if new_model.load_model(test_model_path):
                    print("✓ Model loading successful")
                    
                    # Test prediction with loaded model
                    loaded_prediction = new_model.predict_match(test_match)
                    if loaded_prediction:
                        print("✓ Prediction with loaded model successful")
                    else:
                        print("✗ Prediction with loaded model failed")
                else:
                    print("✗ Model loading failed")
            else:
                print("✗ Model saving failed")
        else:
            print("✗ Prediction failed")
    else:
        print("✗ Model training failed")

def test_data_processor():
    """Test the data processor with mock data."""
    print("\nTesting Football Data Processor...")
    
    # Create mock API client (we won't actually use it)
    class MockAPIClient:
        pass
    
    processor = FootballDataProcessor(MockAPIClient())
    
    # Test feature extraction
    mock_team_stats = {
        'fixtures': {
            'played': {'total': 20},
            'wins': {'total': 12, 'home': 8, 'away': 4},
            'draws': {'total': 5, 'home': 2, 'away': 3},
            'loses': {'total': 3, 'home': 1, 'away': 2}
        },
        'goals': {
            'for': {'total': {'total': 35, 'home': 20, 'away': 15}},
            'against': {'total': {'total': 15, 'home': 8, 'away': 7}}
        }
    }
    
    features = processor.extract_team_features(mock_team_stats)
    
    if features and features['matches_played'] == 20:
        print("✓ Feature extraction successful")
        print(f"  Matches played: {features['matches_played']}")
        print(f"  Win rate: {features['win_rate']:.3f}")
        print(f"  Goals per match: {features['goals_per_match']:.3f}")
    else:
        print("✗ Feature extraction failed")
    
    # Test H2H analysis
    mock_h2h_matches = [
        {
            'fixture': {'status': {'short': 'FT'}},
            'teams': {'home': {'id': 1}, 'away': {'id': 2}},
            'goals': {'home': 2, 'away': 1}
        },
        {
            'fixture': {'status': {'short': 'FT'}},
            'teams': {'home': {'id': 2}, 'away': {'id': 1}},
            'goals': {'home': 1, 'away': 1}
        }
    ]
    
    h2h_features = processor.analyze_h2h_history(mock_h2h_matches, 1, 2)
    
    if h2h_features and h2h_features['h2h_total_matches'] == 2:
        print("✓ H2H analysis successful")
        print(f"  Total H2H matches: {h2h_features['h2h_total_matches']}")
        print(f"  Team 1 wins: {h2h_features['h2h_team1_wins']}")
    else:
        print("✗ H2H analysis failed")

def test_csv_output():
    """Test CSV output functionality."""
    print("\nTesting CSV Output...")
    
    # Create mock predictions
    mock_predictions = [
        {
            'team1_name': 'Team A',
            'team2_name': 'Team B',
            'predicted_outcome': 'Home Win',
            'confidence': 0.75,
            'prediction_date': datetime.now().isoformat(),
            'match_date': '2023-12-03T20:00:00',
            'team1_id': 1,
            'team2_id': 2,
            'league_id': 61,
            'season': 2023
        },
        {
            'team1_name': 'Team C',
            'team2_name': 'Team D',
            'predicted_outcome': 'Away Win',
            'confidence': 0.68,
            'prediction_date': datetime.now().isoformat(),
            'match_date': '2023-12-04T15:00:00',
            'team1_id': 3,
            'team2_id': 4,
            'league_id': 61,
            'season': 2023
        }
    ]
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs('/tmp/test_predictions', exist_ok=True)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(mock_predictions)
        csv_path = '/tmp/test_predictions/test_predictions.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Verify CSV was created
        if os.path.exists(csv_path):
            # Read back and verify content
            df_read = pd.read_csv(csv_path)
            if len(df_read) == 2 and 'predicted_outcome' in df_read.columns:
                print("✓ CSV output successful")
                print(f"  File saved to: {csv_path}")
                print(f"  Predictions saved: {len(df_read)}")
                
                # Show sample content
                print("\nSample CSV content:")
                print(df_read[['team1_name', 'team2_name', 'predicted_outcome', 'confidence']].to_string(index=False))
            else:
                print("✗ CSV content verification failed")
        else:
            print("✗ CSV file not created")
    
    except Exception as e:
        print(f"✗ CSV output failed: {e}")

def main():
    """Run all tests."""
    print("Football Predictor - Test Suite")
    print("=" * 40)
    
    test_data_processor()
    test_prediction_model()
    test_csv_output()
    
    print("\n" + "=" * 40)
    print("Test suite completed!")
    print("\nNote: This test uses mock data. To use real football data,")
    print("you need to set up an API key from api-football.com")

if __name__ == "__main__":
    main()