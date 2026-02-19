import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Stock Prediction API"
    API_V1_STR: str = "/api/v1"
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    
    MODEL_PATH: str = os.path.join(DATA_DIR, "lstm_stock_model_fixed.keras")
    PREPROCESSOR_PATH: str = os.path.join(DATA_DIR, "preprocessor.pkl")

    class Config:
        case_sensitive = True

settings = Settings()
