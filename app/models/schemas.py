from pydantic import BaseModel
from typing import List
from datetime import datetime

class Candle(BaseModel):
    Date: datetime
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float

class PredictionRequest(BaseModel):
    ticker: str = "SPY"

class PredictionResponse(BaseModel):
    ticker: str
    predicted_close: float
    timestamp: datetime

class HistoryResponse(BaseModel):
    ticker: str
    history: List[Candle]
