from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import numpy as np
from app.models.schemas import PredictionRequest, PredictionResponse, HistoryResponse, Candle
from app.core.containers import ModelContainer
from app.services.preprocessor import InferencePreprocessor
from app.services.data_loader import DataLoader
from datetime import timedelta

router = APIRouter()

@router.get(
    "/history",
    response_model=HistoryResponse,
    summary="Busca de Histórico de Ações",
    description="Recupera os últimos **100 dias úteis** de dados históricos consolidados para o ativo selecionado direto do Yahoo Finance."
)
def get_history(ticker: str = Query("SPY", description="Ticker symbol")):
    """
    Retorna os últimos 100 dias de dados históricos para o ticker informado.
    """
    try:
        # Busca mais dados para garantir que os últimos 100 sejam válidos e úteis
        df = DataLoader.fetch_data(ticker, days_back=200)
        
        # Pega os últimos 100
        last_100 = df.tail(100).to_dict(orient='records')
        
        return HistoryResponse(
            ticker=ticker,
            history=last_100
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar histórico: {str(e)}")

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predição do Preço de Fechamento",
    description="Calcula e infere o valor do preço de fechamento para o **próximo dia útil** através da arquitetura LSTM Neural Network de 60 timestamps de lag."
)
def predict_stock(request: PredictionRequest):
    """
    Realiza a predição para o ticker informado.
    Coleta dados automaticamente via yfinance.
    """
    ticker = request.ticker
    
    # 1. Carregar Recursos
    try:
        model = ModelContainer.get_model()
        scaler = ModelContainer.get_scaler()
        feature_columns = ModelContainer.get_feature_columns()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recursos do modelo não disponíveis: {str(e)}")

    # 2. Buscar Dados (Automation)
    try:
        # Busca ~200 dias para garantir indicadores + 60 dias de sequência
        df = DataLoader.fetch_data(ticker, days_back=300)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao buscar dados para {ticker}: {str(e)}")

    # 3. Pré-processar
    preprocessor = InferencePreprocessor(scaler, feature_columns)
    try:
        # O preprocessor vai calcular indicadores e pegar a última sequência de 60
        X_input = preprocessor.prepare_inference_data(df, seq_length=60)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no pré-processamento: {str(e)}")

    # 4. Predizer
    try:
        prediction_scaled = model.predict(X_input, verbose=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")

    # 5. Transformação Inversa (Desnormalizar)
    try:
        prediction_value = preprocessor.inverse_transform_prediction(prediction_scaled)
        predicted_close = float(prediction_value[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no pós-processamento: {str(e)}")

    # 6. Resposta
    # A predição é para o dia seguinte ao último dado coletado
    last_date = df.iloc[-1]['Date']
    next_date = last_date + timedelta(days=1)

    return PredictionResponse(
        ticker=ticker,
        predicted_close=predicted_close,
        timestamp=next_date
    )
