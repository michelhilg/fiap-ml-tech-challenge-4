import pandas as pd
import numpy as np

class InferencePreprocessor:
    def __init__(self, scaler, feature_columns):
        self.scaler = scaler
        self.feature_columns = feature_columns

    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria indicadores técnicos idênticos aos do treinamento via Pandas.
        Requer dados históricos no DataFrame.
        """
        df = df.copy()
        
        # Garantir tipos numéricos
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Tratar valores ausentes (ffill)
        df = df.ffill()

        # --- Indicadores Técnicos (Copiados da lógica do Notebook) ---
        # Médias Móveis (Moving Averages)
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_21'] = df['Close'].rolling(window=21).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Médias Móveis Exponenciais (EMA)
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bandas de Bollinger (Bollinger Bands)
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Média de Volume
        df['Volume_MA_7'] = df['Volume'].rolling(window=7).mean()
        
        # Mudanças de Preço
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        
        # Volatilidade e ATR
        df['Volatility'] = df['Close'].rolling(window=20).std()
        df['ATR'] = df['High'] - df['Low']
        
        # Remover linhas com NaN criados pelos indicadores (ex: primeiras 50 linhas)
        # Para inferência, precisamos que a última linha seja válida.
        df.dropna(inplace=True)
        
        return df

    def prepare_inference_data(self, df: pd.DataFrame, seq_length=60):
        """
        Prepara a última sequência de dados para predição.
        """
        # 1. Criar indicadores
        df_processed = self.create_technical_indicators(df)
        
        if len(df_processed) < seq_length:
             raise ValueError(f"Dados insuficientes após processamento. Necessário pelo menos {seq_length} linhas, obtido {len(df_processed)}. Forneça mais histórico.")

        # 2. Selecionar colunas
        try:
             df_features = df_processed[self.feature_columns]
        except KeyError as e:
             missing = list(set(self.feature_columns) - set(df_processed.columns))
             raise ValueError(f"Colunas ausentes: {missing}")

        # 3. Obter última sequência
        last_sequence = df_features.values[-seq_length:]
        
        # 4. Normalizar (Apenas transform)
        # Reshape para o scaler: (n_samples, n_features) -> (seq_length, n_features)
        # O Scaler espera array 2D.
        sequence_scaled = self.scaler.transform(last_sequence)
        
        # 5. Reshape para LSTM: (1, seq_length, n_features)
        X_input = sequence_scaled.reshape(1, seq_length, len(self.feature_columns))
        
        return X_input

    def inverse_transform_prediction(self, scaled_prediction):
        """
        Reverte a normalização para a coluna alvo ('Close').
        """
        # O Scaler foi treinado (fit) em TODAS as features.
        # A predição é apenas 1 valor (Close).
        # Precisamos criar uma linha dummy com shape (1, n_features) para usar inverse_transform.
        
        close_col_idx = self.feature_columns.index('Close')
        
        dummy = np.zeros(shape=(len(scaled_prediction), len(self.feature_columns)))
        dummy[:, close_col_idx] = scaled_prediction
        
        unscaled = self.scaler.inverse_transform(dummy)
        return unscaled[:, close_col_idx]
