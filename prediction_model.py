"""Machine learning model for football match prediction."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from typing import Dict, List, Tuple, Optional, Any
import pickle
import os

class FootballPredictionModel:
    """Machine learning model for predicting football match outcomes."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
    def prepare_training_data(self, fixtures: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data from historical fixtures."""
        training_data = []
        
        for fixture in fixtures:
            try:
                teams = fixture.get('teams', {})
                goals = fixture.get('goals', {})
                fixture_info = fixture.get('fixture', {})
                
                # Skip if not finished
                if fixture_info.get('status', {}).get('short') != 'FT':
                    continue
                
                home_goals = goals.get('home', 0) or 0
                away_goals = goals.get('away', 0) or 0
                
                # Determine outcome (0: away win, 1: draw, 2: home win)
                if home_goals > away_goals:
                    outcome = 2  # Home win
                elif home_goals < away_goals:
                    outcome = 0  # Away win
                else:
                    outcome = 1  # Draw
                
                # For now, create basic features from available fixture data
                # In a real implementation, this would be combined with team statistics
                match_data = {
                    'home_team_id': teams.get('home', {}).get('id', 0),
                    'away_team_id': teams.get('away', {}).get('id', 0),
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    'outcome': outcome
                }
                
                training_data.append(match_data)
                
            except Exception as e:
                print(f"Error processing fixture: {e}")
                continue
        
        if not training_data:
            return pd.DataFrame(), pd.Series()
        
        df = pd.DataFrame(training_data)
        
        # Create basic features
        features_df = df[['home_team_id', 'away_team_id']].copy()
        
        # Add simple team performance features based on historical data
        for team_col in ['home_team_id', 'away_team_id']:
            team_stats = df.groupby(team_col).agg({
                'home_goals': 'mean',
                'away_goals': 'mean',
                'outcome': lambda x: (x == 2).sum() if team_col == 'home_team_id' else (x == 0).sum()
            }).add_prefix(f'{team_col}_avg_')
            
            features_df = features_df.merge(
                team_stats, 
                left_on=team_col, 
                right_index=True, 
                how='left'
            )
        
        # Fill missing values
        features_df = features_df.fillna(0)
        
        return features_df, df['outcome']
    
    def create_match_features(self, match_data: Dict) -> pd.DataFrame:
        """Create feature vector from match data."""
        if not match_data:
            return pd.DataFrame()
        
        # Define the feature columns we want to use for prediction
        feature_columns = [
            'team1_matches_played', 'team1_wins', 'team1_draws', 'team1_losses',
            'team1_goals_for', 'team1_goals_against', 'team1_win_rate',
            'team1_goals_per_match', 'team1_goals_conceded_per_match',
            'team1_goal_difference_per_match',
            'team2_matches_played', 'team2_wins', 'team2_draws', 'team2_losses',
            'team2_goals_for', 'team2_goals_against', 'team2_win_rate',
            'team2_goals_per_match', 'team2_goals_conceded_per_match',
            'team2_goal_difference_per_match',
            'team1_recent_wins', 'team1_recent_draws', 'team1_recent_losses',
            'team1_recent_win_rate', 'team1_recent_form_points',
            'team2_recent_wins', 'team2_recent_draws', 'team2_recent_losses',
            'team2_recent_win_rate', 'team2_recent_form_points',
            'h2h_total_matches', 'h2h_team1_wins', 'h2h_team2_wins',
            'h2h_draws', 'h2h_team1_win_rate', 'h2h_avg_goals'
        ]
        
        # Create feature vector
        features = {}
        for col in feature_columns:
            features[col] = match_data.get(col, 0)
        
        # Add relative strength features
        if match_data.get('team1_matches_played', 0) > 0 and match_data.get('team2_matches_played', 0) > 0:
            features['win_rate_difference'] = features['team1_win_rate'] - features['team2_win_rate']
            features['goals_difference'] = features['team1_goals_per_match'] - features['team2_goals_per_match']
            features['defense_difference'] = features['team2_goals_conceded_per_match'] - features['team1_goals_conceded_per_match']
            features['form_difference'] = features['team1_recent_form_points'] - features['team2_recent_form_points']
        else:
            features.update({
                'win_rate_difference': 0,
                'goals_difference': 0,
                'defense_difference': 0,
                'form_difference': 0
            })
        
        return pd.DataFrame([features])
    
    def train_model(self, training_data: List[Dict]) -> bool:
        """Train the prediction model with historical data."""
        if not training_data:
            print("No training data provided")
            return False
        
        try:
            # Convert training data to features and targets
            features_list = []
            targets = []
            
            for data in training_data:
                if 'outcome' not in data:
                    continue
                    
                features_df = self.create_match_features(data)
                if not features_df.empty:
                    features_list.append(features_df)
                    targets.append(data['outcome'])
            
            if not features_list:
                print("No valid training features created")
                return False
            
            # Combine all features
            X = pd.concat(features_list, ignore_index=True)
            y = np.array(targets)
            
            self.feature_columns = X.columns.tolist()
            
            # Handle missing values
            X = X.fillna(0)
            
            # Split data
            if len(X) < 10:
                print("Insufficient training data")
                return False
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train ensemble model
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            gb_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            
            # Use Random Forest as primary model
            rf_model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = rf_model.score(X_train_scaled, y_train)
            test_score = rf_model.score(X_test_scaled, y_test)
            
            print(f"Model training completed:")
            print(f"Training accuracy: {train_score:.3f}")
            print(f"Test accuracy: {test_score:.3f}")
            
            self.model = rf_model
            self.is_trained = True
            
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def predict_match(self, match_data: Dict) -> Optional[Dict[str, Any]]:
        """Predict the outcome of a match."""
        if not self.is_trained or self.model is None:
            print("Model is not trained")
            return None
        
        try:
            # Create features
            features_df = self.create_match_features(match_data)
            
            if features_df.empty:
                print("Could not create features for prediction")
                return None
            
            # Ensure all required columns are present
            for col in self.feature_columns:
                if col not in features_df.columns:
                    features_df[col] = 0
            
            # Reorder columns to match training data
            features_df = features_df[self.feature_columns]
            
            # Scale features
            features_scaled = self.scaler.transform(features_df)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Map prediction to outcome
            outcome_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
            predicted_outcome = outcome_map[prediction]
            
            # Get confidence
            confidence = max(probabilities)
            
            result = {
                'predicted_outcome': predicted_outcome,
                'prediction_code': int(prediction),
                'confidence': float(confidence),
                'probabilities': {
                    'home_win': float(probabilities[2]),
                    'draw': float(probabilities[1]),
                    'away_win': float(probabilities[0])
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model to disk."""
        if not self.is_trained:
            print("No trained model to save")
            return False
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model from disk."""
        if not os.path.exists(filepath):
            print(f"Model file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = True
            
            print(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False