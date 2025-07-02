import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from RandomForest.RandomForest import RandomForest
import json
import os
from dotenv import load_dotenv

class OddsAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        
        # Initialize the Random Forest model with optimized parameters
        self.model = RandomForest(
            n_features=7,
            n_estimators=200,
            tree_params=dict(
                max_depth=15,
                min_samples_split=20,
                min_gini_change=0.01,
                bagging=True
            )
        )
        
        # Train the model with historical data
        self.train_model()
    
    def train_model(self):
        """Train the model with historical tennis data"""
        try:
            # Load and prepare training data
            print("Loading training data...")
            
            # Combine all match files
            all_matches = []
            data_dir = 'data/all'
            for file in os.listdir(data_dir):
                if file.startswith('atp_matches_') and file.endswith('.csv'):
                    file_path = os.path.join(data_dir, file)
                    matches_df = pd.read_csv(file_path)
                    all_matches.append(matches_df)
            
            matches_df = pd.concat(all_matches, ignore_index=True)
            print(f"Loaded {len(matches_df)} matches from {len(all_matches)} files")
            
            # Prepare features
            # Convert surface to numeric
            surface_map = {'Hard': 0, 'Clay': 1, 'Grass': 2}
            matches_df['surface_numeric'] = matches_df['surface'].map(surface_map)
            
            # Calculate form (win percentage in last 10 matches)
            matches_df['player1_form'] = 0.5  # Default value
            matches_df['player2_form'] = 0.5  # Default value
            
            # Prepare features for model
            features = [
                'winner_rank', 'loser_rank',  # Rankings
                'surface_numeric',  # Surface type
                'winner_ht', 'loser_ht',  # Height
                'winner_age', 'loser_age'  # Age
            ]
            
            # Create feature matrix
            X = matches_df[features].values
            
            # Create target variable (1 for winner, 0 for loser)
            y = np.ones(len(matches_df))
            
            # Combine features and target
            data = np.hstack((X, y.reshape(-1, 1)))
            
            print("Training Random Forest model...")
            self.model.build_forest(data)
            print("Model training completed!")
            
        except Exception as e:
            print(f"Error training model: {e}")
            raise
    
    def get_tennis_matches(self):
        """Get upcoming tennis matches from The Odds API"""
        try:
            response = requests.get(
                f"{self.base_url}/sports/tennis/odds",
                params={
                    'apiKey': self.api_key,
                    'regions': 'us',
                    'markets': 'h2h',
                    'oddsFormat': 'decimal',
                    'dateFormat': 'iso'
                }
            )
            response.raise_for_status()
            matches = response.json()
            
            # Filter for Cincinnati Open matches
            cincinnati_matches = [
                match for match in matches 
                if 'Cincinnati' in match.get('sport_title', '') 
                or 'Western & Southern' in match.get('sport_title', '')
            ]
            
            if cincinnati_matches:
                print(f"\nFound {len(cincinnati_matches)} Cincinnati Open matches:")
                for match in cincinnati_matches:
                    print(f"- {match['home_team']} vs {match['away_team']}")
            else:
                print("\nNo Cincinnati Open matches found at this time.")
            
            return cincinnati_matches
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching tennis matches: {e}")
            return None
    
    def prepare_match_data(self, match_data):
        """Prepare match data for model prediction"""
        # Extract relevant features from match data
        features = {
            'winner_rank': match_data.get('home_team_rank', 100),
            'loser_rank': match_data.get('away_team_rank', 100),
            'surface_numeric': 0,  # Default to Hard court
            'winner_ht': match_data.get('home_team_height', 180),
            'loser_ht': match_data.get('away_team_height', 180),
            'winner_age': match_data.get('home_team_age', 25),
            'loser_age': match_data.get('away_team_age', 25)
        }
        return pd.DataFrame([features])
    
    def find_value_bets(self, min_probability_diff=0.1):
        """Find potential value bets by comparing model predictions with bookmaker odds"""
        matches = self.get_tennis_matches()
        if not matches:
            return []
        
        value_bets = []
        
        for match in matches:
            # Get the best odds from all bookmakers
            best_home_odds = max(float(bookmaker['markets'][0]['outcomes'][0]['price']) 
                               for bookmaker in match['bookmakers'])
            best_away_odds = max(float(bookmaker['markets'][0]['outcomes'][1]['price']) 
                               for bookmaker in match['bookmakers'])
            
            # Prepare match data for prediction
            match_data = self.prepare_match_data(match)
            
            # Get model prediction
            prediction = self.model.predict(match_data)
            model_probability = prediction[0][0]
            
            # Calculate implied probability from odds
            bookmaker_probability = 1 / best_home_odds
            
            # Check if there's a significant difference between model and bookmaker probabilities
            if abs(model_probability - bookmaker_probability) > min_probability_diff:
                value_bet = {
                    'match_id': match['id'],
                    'home_team': match['home_team'],
                    'away_team': match['away_team'],
                    'model_probability': model_probability,
                    'bookmaker_probability': bookmaker_probability,
                    'odds': best_home_odds,
                    'value': (model_probability - bookmaker_probability) * 100
                }
                value_bets.append(value_bet)
        
        return sorted(value_bets, key=lambda x: abs(x['value']), reverse=True)
    
    def monitor_odds(self, interval_minutes=5):
        """Continuously monitor odds for value betting opportunities"""
        print("\nMonitoring Cincinnati Open matches for value bets...")
        while True:
            print(f"\nChecking for value bets at {datetime.now()}")
            value_bets = self.find_value_bets()
            
            if value_bets:
                print("\nPotential value bets found in Cincinnati Open:")
                for bet in value_bets:
                    print(f"\nMatch: {bet['home_team']} vs {bet['away_team']}")
                    print(f"Model probability: {bet['model_probability']:.2%}")
                    print(f"Bookmaker probability: {bet['bookmaker_probability']:.2%}")
                    print(f"Best odds: {bet['odds']}")
                    print(f"Value: {bet['value']:.2f}%")
            else:
                print("No value bets found at this time.")
            
            time.sleep(interval_minutes * 60)

def main():
    # Load API key from environment variable
    load_dotenv()
    api_key = os.getenv('ODDS_API_KEY')
    
    if not api_key:
        print("Error: ODDS_API_KEY environment variable not set")
        return
    
    print("Initializing Odds Analyzer...")
    analyzer = OddsAnalyzer(api_key)
    
    print("\nStarting odds monitoring...")
    analyzer.monitor_odds()

if __name__ == "__main__":
    main() 