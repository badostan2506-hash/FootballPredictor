#!/usr/bin/env python3
"""
Demo script for Football Predictor.
This script demonstrates the system's capabilities with mock data when no API key is available.
"""

import sys
import os
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prediction_model import FootballPredictionModel
import pandas as pd

def create_demo_data():
    """Create realistic demo data for demonstration."""
    # French Ligue 1 teams with realistic stats
    teams = {
        85: {"name": "Paris Saint Germain", "strength": 0.9},
        81: {"name": "Olympique Marseille", "strength": 0.7},
        80: {"name": "Olympique Lyonnais", "strength": 0.75},
        82: {"name": "AS Monaco", "strength": 0.72},
        96: {"name": "OGC Nice", "strength": 0.65},
        84: {"name": "Lille OSC", "strength": 0.68},
        83: {"name": "Stade Rennais FC", "strength": 0.62},
        79: {"name": "RC Lens", "strength": 0.60}
    }
    
    training_data = []
    predictions_data = []
    
    # Generate training data
    team_ids = list(teams.keys())
    
    for i in range(100):  # Generate 100 matches
        home_id = team_ids[i % len(team_ids)]
        away_id = team_ids[(i + 1) % len(team_ids)]
        
        if home_id == away_id:
            continue
            
        home_strength = teams[home_id]["strength"]
        away_strength = teams[away_id]["strength"]
        
        # Create realistic match data based on team strength
        home_advantage = 0.1  # Home teams have slight advantage
        
        # Generate realistic team statistics
        match_data = {
            'team1_matches_played': 20,
            'team1_wins': int(20 * home_strength * 0.6),
            'team1_draws': int(20 * 0.25),
            'team1_losses': 20 - int(20 * home_strength * 0.6) - int(20 * 0.25),
            'team1_goals_for': int(40 * home_strength),
            'team1_goals_against': int(25 * (1 - home_strength)),
            'team1_win_rate': home_strength * 0.6,
            'team1_goals_per_match': 2.0 * home_strength,
            'team1_goals_conceded_per_match': 1.25 * (1 - home_strength),
            'team1_goal_difference_per_match': (2.0 * home_strength) - (1.25 * (1 - home_strength)),
            
            'team2_matches_played': 20,
            'team2_wins': int(20 * away_strength * 0.5),  # Away teams perform slightly worse
            'team2_draws': int(20 * 0.25),
            'team2_losses': 20 - int(20 * away_strength * 0.5) - int(20 * 0.25),
            'team2_goals_for': int(35 * away_strength),
            'team2_goals_against': int(30 * (1 - away_strength)),
            'team2_win_rate': away_strength * 0.5,
            'team2_goals_per_match': 1.75 * away_strength,
            'team2_goals_conceded_per_match': 1.5 * (1 - away_strength),
            'team2_goal_difference_per_match': (1.75 * away_strength) - (1.5 * (1 - away_strength)),
            
            'team1_recent_wins': min(5, int(5 * home_strength)),
            'team1_recent_draws': 1,
            'team1_recent_losses': max(0, 4 - int(5 * home_strength)),
            'team1_recent_win_rate': home_strength,
            'team1_recent_form_points': int(15 * home_strength),
            
            'team2_recent_wins': min(5, int(5 * away_strength * 0.8)),
            'team2_recent_draws': 1,
            'team2_recent_losses': max(0, 4 - int(5 * away_strength * 0.8)),
            'team2_recent_win_rate': away_strength * 0.8,
            'team2_recent_form_points': int(12 * away_strength),
            
            'h2h_total_matches': 5,
            'h2h_team1_wins': int(5 * (home_strength - away_strength + 1) / 2),
            'h2h_team2_wins': int(5 * (away_strength - home_strength + 1) / 2),
            'h2h_draws': 1,
            'h2h_team1_win_rate': (home_strength - away_strength + 1) / 2,
            'h2h_avg_goals': 2.8,
        }
        
        # Determine outcome based on relative strength
        total_strength = home_strength + home_advantage - away_strength
        if total_strength > 0.3:
            outcome = 2  # Home win
        elif total_strength < -0.2:
            outcome = 0  # Away win
        else:
            outcome = 1  # Draw
            
        match_data['outcome'] = outcome
        training_data.append(match_data)
    
    # Generate upcoming matches for prediction
    for i in range(5):
        home_id = team_ids[i]
        away_id = team_ids[i + 3]
        
        home_strength = teams[home_id]["strength"]
        away_strength = teams[away_id]["strength"]
        
        match_data = {
            'team1_id': home_id,
            'team2_id': away_id,
            'team1_name': teams[home_id]["name"],
            'team2_name': teams[away_id]["name"],
            'match_date': (datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d %H:%M:%S"),
            'team1_matches_played': 20,
            'team1_wins': int(20 * home_strength * 0.6),
            'team1_draws': int(20 * 0.25),
            'team1_losses': 20 - int(20 * home_strength * 0.6) - int(20 * 0.25),
            'team1_goals_for': int(40 * home_strength),
            'team1_goals_against': int(25 * (1 - home_strength)),
            'team1_win_rate': home_strength * 0.6,
            'team1_goals_per_match': 2.0 * home_strength,
            'team1_goals_conceded_per_match': 1.25 * (1 - home_strength),
            'team1_goal_difference_per_match': (2.0 * home_strength) - (1.25 * (1 - home_strength)),
            
            'team2_matches_played': 20,
            'team2_wins': int(20 * away_strength * 0.5),
            'team2_draws': int(20 * 0.25),
            'team2_losses': 20 - int(20 * away_strength * 0.5) - int(20 * 0.25),
            'team2_goals_for': int(35 * away_strength),
            'team2_goals_against': int(30 * (1 - away_strength)),
            'team2_win_rate': away_strength * 0.5,
            'team2_goals_per_match': 1.75 * away_strength,
            'team2_goals_conceded_per_match': 1.5 * (1 - away_strength),
            'team2_goal_difference_per_match': (1.75 * away_strength) - (1.5 * (1 - away_strength)),
            
            'team1_recent_wins': min(5, int(5 * home_strength)),
            'team1_recent_draws': 1,
            'team1_recent_losses': max(0, 4 - int(5 * home_strength)),
            'team1_recent_win_rate': home_strength,
            'team1_recent_form_points': int(15 * home_strength),
            
            'team2_recent_wins': min(5, int(5 * away_strength * 0.8)),
            'team2_recent_draws': 1,
            'team2_recent_losses': max(0, 4 - int(5 * away_strength * 0.8)),
            'team2_recent_win_rate': away_strength * 0.8,
            'team2_recent_form_points': int(12 * away_strength),
            
            'h2h_total_matches': 5,
            'h2h_team1_wins': int(5 * (home_strength - away_strength + 1) / 2),
            'h2h_team2_wins': int(5 * (away_strength - home_strength + 1) / 2),
            'h2h_draws': 1,
            'h2h_team1_win_rate': (home_strength - away_strength + 1) / 2,
            'h2h_avg_goals': 2.8,
        }
        
        predictions_data.append(match_data)
    
    return training_data, predictions_data

def main():
    """Run the demo."""
    print("🏈 Football Predictor - DEMO MODE")
    print("=" * 50)
    print("This demo shows the system working with realistic mock data")
    print("(French Ligue 1 teams with simulated statistics)")
    print("")
    
    # Create demo data
    print("📊 Generating realistic demo data...")
    training_data, upcoming_matches = create_demo_data()
    print(f"✓ Created {len(training_data)} training matches")
    print(f"✓ Created {len(upcoming_matches)} upcoming matches")
    print("")
    
    # Train the model
    print("🤖 Training prediction model...")
    model = FootballPredictionModel()
    success = model.train_model(training_data)
    
    if not success:
        print("❌ Failed to train model")
        return
    
    print("✅ Model training completed!")
    print("")
    
    # Make predictions
    print("🔮 Making predictions for upcoming matches...")
    print("")
    
    predictions = []
    
    for i, match in enumerate(upcoming_matches, 1):
        match_copy = match.copy()
        # Remove non-feature data for prediction
        for key in ['team1_id', 'team2_id', 'team1_name', 'team2_name', 'match_date']:
            match_copy.pop(key, None)
        
        prediction = model.predict_match(match_copy)
        
        if prediction:
            # Add match info back
            prediction.update({
                'team1_name': match['team1_name'],
                'team2_name': match['team2_name'],
                'match_date': match['match_date'],
                'prediction_date': datetime.now().isoformat()
            })
            
            predictions.append(prediction)
            
            # Display prediction
            print(f"Match {i}: {match['team1_name']} vs {match['team2_name']}")
            print(f"  📅 Date: {match['match_date']}")
            print(f"  🏆 Prediction: {prediction['predicted_outcome']}")
            print(f"  📊 Confidence: {prediction['confidence']:.1%}")
            print(f"  📈 Probabilities:")
            print(f"     Home Win: {prediction['probabilities']['home_win']:.1%}")
            print(f"     Draw:     {prediction['probabilities']['draw']:.1%}")
            print(f"     Away Win: {prediction['probabilities']['away_win']:.1%}")
            print("")
    
    # Save to CSV
    if predictions:
        print("💾 Saving predictions to CSV...")
        
        # Create output directory
        os.makedirs('demo_output', exist_ok=True)
        
        df = pd.DataFrame(predictions)
        csv_path = 'demo_output/demo_predictions.csv'
        
        # Reorder columns for better readability
        column_order = [
            'team1_name', 'team2_name', 'predicted_outcome',
            'confidence', 'match_date', 'prediction_date'
        ]
        
        available_columns = [col for col in column_order if col in df.columns]
        remaining_columns = [col for col in df.columns if col not in available_columns]
        final_columns = available_columns + remaining_columns
        
        df[final_columns].to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"✅ Predictions saved to: {csv_path}")
        print(f"📈 Total predictions: {len(predictions)}")
        
        # Show CSV preview
        print("\\n📋 CSV Preview:")
        print(df[['team1_name', 'team2_name', 'predicted_outcome', 'confidence']].to_string(index=False))
    
    print("")
    print("=" * 50)
    print("🎯 Demo completed successfully!")
    print("")
    print("To use with real football data:")
    print("1. Get an API key from api-football.com")
    print("2. Create a .env file with: FOOTBALL_API_KEY=your_key")
    print("3. Run: python main.py --train --league 61 --season 2023")
    print("4. Then: python main.py --predict --league 61 --season 2023")

if __name__ == "__main__":
    main()