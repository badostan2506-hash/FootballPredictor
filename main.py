#!/usr/bin/env python3
"""
Football Predictor - Main script for predicting football match outcomes.

This script provides a complete football prediction system that:
1. Extracts statistics from v3.football.api-sports.io API
2. Builds a machine learning model based on team statistics
3. Generates predictions for matches
4. Saves predictions to CSV files

Usage:
    python main.py --train --league 61 --season 2023  # Train model
    python main.py --predict --league 61 --season 2023  # Make predictions
    python main.py --help  # Show help
"""

import argparse
import sys
from predictor import FootballPredictor
from config import Config

def main():
    """Main function to run the football predictor."""
    parser = argparse.ArgumentParser(description='Football Match Predictor')
    
    # Main actions
    parser.add_argument('--train', action='store_true', 
                       help='Train the prediction model')
    parser.add_argument('--predict', action='store_true',
                       help='Make predictions for upcoming matches')
    parser.add_argument('--predict-match', action='store_true',
                       help='Predict a specific match between two teams')
    
    # League and season parameters
    parser.add_argument('--league', type=int, default=61,
                       help='League ID (default: 61 for Ligue 1)')
    parser.add_argument('--season', type=int, default=2023,
                       help='Season year (default: 2023)')
    
    # Team parameters for specific match prediction
    parser.add_argument('--team1', type=int,
                       help='Home team ID for specific match prediction')
    parser.add_argument('--team2', type=int,
                       help='Away team ID for specific match prediction')
    
    # Optional parameters
    parser.add_argument('--max-matches', type=int, default=10,
                       help='Maximum number of matches to predict (default: 10)')
    parser.add_argument('--max-teams', type=int, default=10,
                       help='Maximum number of teams for training (default: 10)')
    parser.add_argument('--output', type=str,
                       help='Custom output filename for CSV')
    
    # Model management
    parser.add_argument('--load-model', type=str,
                       help='Load a pre-trained model from file')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.train, args.predict, args.predict_match]):
        print("Error: Must specify one of --train, --predict, or --predict-match")
        parser.print_help()
        return 1
    
    if args.predict_match and (not args.team1 or not args.team2):
        print("Error: --predict-match requires both --team1 and --team2")
        return 1
    
    # Initialize predictor
    predictor = FootballPredictor()
    
    if not predictor.initialize():
        print("Failed to initialize predictor")
        return 1
    
    # Load model if specified
    if args.load_model:
        if not predictor.load_model(args.load_model):
            print(f"Failed to load model from {args.load_model}")
            return 1
    
    try:
        if args.train:
            print(f"Training model with League {args.league}, Season {args.season}")
            success = predictor.train_model_from_league(
                args.league, 
                args.season, 
                max_teams=args.max_teams
            )
            
            if not success:
                print("Model training failed")
                return 1
            
            print("Model training completed successfully")
        
        elif args.predict_match:
            print(f"Predicting match: Team {args.team1} vs Team {args.team2}")
            
            # Load model if not already loaded
            if not predictor.model.is_trained:
                model_path = Config.get_output_path("football_model.pkl")
                if not predictor.load_model(model_path):
                    print("No trained model found. Please train the model first with --train")
                    return 1
            
            prediction = predictor.predict_match(
                args.team1, 
                args.team2, 
                args.league, 
                args.season
            )
            
            if prediction:
                print("\\nPrediction Results:")
                print(f"Match: {prediction['team1_name']} vs {prediction['team2_name']}")
                print(f"Predicted outcome: {prediction['predicted_outcome']}")
                print(f"Confidence: {prediction['confidence']:.2f}")
                print("\\nProbabilities:")
                print(f"  Home Win: {prediction['probabilities']['home_win']:.3f}")
                print(f"  Draw: {prediction['probabilities']['draw']:.3f}")
                print(f"  Away Win: {prediction['probabilities']['away_win']:.3f}")
                
                # Save single prediction
                if predictor.save_predictions_to_csv(args.output):
                    print(f"Prediction saved to CSV")
            else:
                print("Failed to generate prediction")
                return 1
        
        elif args.predict:
            print(f"Predicting upcoming matches for League {args.league}, Season {args.season}")
            
            # Load model if not already loaded
            if not predictor.model.is_trained:
                model_path = Config.get_output_path("football_model.pkl")
                if not predictor.load_model(model_path):
                    print("No trained model found. Please train the model first with --train")
                    return 1
            
            predictions = predictor.predict_upcoming_matches(
                args.league, 
                args.season, 
                max_matches=args.max_matches
            )
            
            if predictions:
                print(f"\\nGenerated {len(predictions)} predictions")
                
                # Show summary
                summary = predictor.get_prediction_summary()
                print(f"\\nPrediction Summary:")
                print(f"Total predictions: {summary['total']}")
                print(f"Average confidence: {summary['average_confidence']:.3f}")
                print(f"High confidence predictions: {summary['high_confidence']}")
                print(f"Outcome distribution: {summary['outcomes']}")
                
                # Save predictions
                if predictor.save_predictions_to_csv(args.output):
                    print("All predictions saved to CSV")
            else:
                print("No predictions generated")
                return 1
        
        print("\\nOperation completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\\nOperation interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

def show_examples():
    """Show usage examples."""
    examples = """
Usage Examples:

1. Train a model with French Ligue 1 data:
   python main.py --train --league 61 --season 2023

2. Predict upcoming matches in Ligue 1:
   python main.py --predict --league 61 --season 2023 --max-matches 5

3. Predict a specific match (PSG vs Marseille):
   python main.py --predict-match --team1 85 --team2 81 --league 61 --season 2023

4. Train with Premier League data:
   python main.py --train --league 39 --season 2023

5. Save predictions to custom file:
   python main.py --predict --league 61 --season 2023 --output my_predictions.csv

Common League IDs:
  - 61: French Ligue 1
  - 39: English Premier League
  - 140: Spanish La Liga
  - 135: Italian Serie A
  - 78: German Bundesliga

Note: You need a valid API key from api-football.com
Set it in a .env file: FOOTBALL_API_KEY=your_key_here
"""
    print(examples)

if __name__ == "__main__":
    if len(sys.argv) == 1 or "--examples" in sys.argv:
        show_examples()
        sys.exit(0)
    
    sys.exit(main())