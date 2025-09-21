"""API client for football statistics using v3.football.api-sports.io"""

import requests
import time
from typing import Dict, List, Optional, Any
from config import Config

class FootballAPIClient:
    """Client for interacting with the v3.football.api-sports.io API."""
    
    def __init__(self):
        self.base_url = Config.API_BASE_URL
        self.headers = Config.DEFAULT_HEADERS
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a request to the API with error handling and rate limiting."""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if not data.get('response'):
                print(f"Warning: No data returned for {endpoint}")
                return {'response': []}
                
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {endpoint}: {e}")
            return {'response': []}
        
        # Rate limiting - free tier usually has limits
        time.sleep(0.1)
    
    def get_leagues(self, country: str = "France") -> List[Dict]:
        """Get available leagues for a country."""
        params = {"country": country}
        data = self._make_request("leagues", params)
        return data.get('response', [])
    
    def get_teams(self, league_id: int, season: int) -> List[Dict]:
        """Get teams for a specific league and season."""
        params = {"league": league_id, "season": season}
        data = self._make_request("teams", params)
        return data.get('response', [])
    
    def get_team_statistics(self, team_id: int, league_id: int, season: int) -> Dict:
        """Get detailed statistics for a team in a specific league and season."""
        params = {
            "team": team_id,
            "league": league_id,
            "season": season
        }
        data = self._make_request("teams/statistics", params)
        return data.get('response', {})
    
    def get_fixtures(self, league_id: int, season: int, status: str = "FT") -> List[Dict]:
        """Get fixtures (matches) for a league and season."""
        params = {
            "league": league_id,
            "season": season,
            "status": status
        }
        data = self._make_request("fixtures", params)
        return data.get('response', [])
    
    def get_h2h_matches(self, team1_id: int, team2_id: int, last: int = 10) -> List[Dict]:
        """Get head-to-head matches between two teams."""
        params = {
            "h2h": f"{team1_id}-{team2_id}",
            "last": last
        }
        data = self._make_request("fixtures/headtohead", params)
        return data.get('response', [])
    
    def get_standings(self, league_id: int, season: int) -> List[Dict]:
        """Get league standings."""
        params = {
            "league": league_id,
            "season": season
        }
        data = self._make_request("standings", params)
        return data.get('response', [])
    
    def validate_connection(self) -> bool:
        """Test API connection and key validity."""
        if not Config.validate_api_key():
            print("Error: No API key configured. Please set FOOTBALL_API_KEY in .env file")
            return False
            
        try:
            # Test with a simple request
            data = self._make_request("status")
            return bool(data.get('response'))
        except Exception as e:
            print(f"Error validating API connection: {e}")
            return False