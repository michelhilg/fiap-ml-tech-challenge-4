import pickle
import tensorflow as tf
from app.core.config import settings
import os

class ModelContainer:
    model = None
    scaler = None
    feature_columns = None

    @classmethod
    def load_resources(cls):
        print(f"Carregando recursos de {settings.DATA_DIR}...")
        
        # Carregar Modelo Keras
        if os.path.exists(settings.MODEL_PATH):
            cls.model = tf.keras.models.load_model(settings.MODEL_PATH)
            print("✅ Modelo carregado com sucesso.")
        else:
            print(f"❌ Modelo não encontrado em {settings.MODEL_PATH}")
            raise FileNotFoundError(f"Modelo não encontrado em {settings.MODEL_PATH}")

        # Carregar Preprocessador (Pickle)
        if os.path.exists(settings.PREPROCESSOR_PATH):
            with open(settings.PREPROCESSOR_PATH, "rb") as f:
                data = pickle.load(f)
                cls.scaler = data['scaler']
                cls.feature_columns = data['feature_columns']
            print("✅ Preprocessador carregado com sucesso.")
        else:
             print(f"❌ Preprocessador não encontrado em {settings.PREPROCESSOR_PATH}")
             raise FileNotFoundError(f"Preprocessador não encontrado em {settings.PREPROCESSOR_PATH}")

    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.load_resources()
        return cls.model

    @classmethod
    def get_scaler(cls):
        if cls.scaler is None:
            cls.load_resources()
        return cls.scaler

    @classmethod
    def get_feature_columns(cls):
        if cls.feature_columns is None:
            cls.load_resources()
        return cls.feature_columns
