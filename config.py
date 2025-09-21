"""Configuration management for the Football Predictor."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for API settings and parameters."""
    
    # API Configuration
    API_BASE_URL = "https://v3.football.api-sports.io"
    API_KEY = os.getenv("FOOTBALL_API_KEY", "")
    
    # Default headers for API requests
    DEFAULT_HEADERS = {
        "X-RapidAPI-Key": API_KEY,
        "X-RapidAPI-Host": "v3.football.api-sports.io"
    }
    
    # Model parameters
    MIN_MATCHES_FOR_PREDICTION = 5
    PREDICTION_CONFIDENCE_THRESHOLD = 0.6
    
    # Output settings
    OUTPUT_DIR = "predictions"
    CSV_FILENAME = "football_predictions.csv"
    
    @classmethod
    def validate_api_key(cls) -> bool:
        """Validate that API key is configured."""
        return bool(cls.API_KEY and cls.API_KEY.strip())
    
    @classmethod
    def get_output_path(cls, filename: Optional[str] = None) -> str:
        """Get the full output path for a file."""
        if not os.path.exists(cls.OUTPUT_DIR):
            os.makedirs(cls.OUTPUT_DIR)
        
        if filename is None:
            filename = cls.CSV_FILENAME
            
        return os.path.join(cls.OUTPUT_DIR, filename)