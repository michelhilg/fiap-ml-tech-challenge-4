import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class DataLoader:
    @staticmethod
    def fetch_data(ticker: str, days_back: int = 200) -> pd.DataFrame:
        """
        Coleta dados históricos do Yahoo Finance.
        Busca um buffer maior (days_back) para garantir que temos
        dados suficientes para os indicadores (ex: MA_50) e sequência (60).
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back * 1.5) # Margem de segurança para fins de semana/feriados

        # Download
        df = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1d',
            progress=False,
            auto_adjust=True
        )

        if len(df) == 0:
            raise ValueError(f"Nenhum dado encontrado para o ticker {ticker}")

        # O yfinance retorna MultiIndex nas colunas se for apenas 1 ticker nas versoes recentes, ou nao.
        # Vamos garantir que as colunas sejam simples: Open, High, Low, Close, Volume
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Reset index para ter Date como coluna
        df = df.reset_index()
        
        # Selecionar e renomear colunas para o padrão esperado, se necessário
        # Normalmente vem como: Date, Open, High, Low, Close, Volume
        expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Verificar se as colunas existem
        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
             raise ValueError(f"Colunas ausentes no retorno do yahoo finance: {missing}")

        return df[expected_cols]

    @staticmethod
    def get_last_n_days(df: pd.DataFrame, n: int = 100) -> list[dict]:
        """
        Retorna os últimos N dias no formato de dicionário.
        """
        return df.tail(n).to_dict(orient='records')
