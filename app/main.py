from fastapi import FastAPI
from app.core.config import settings
from app.api.v1.endpoints import prediction
from app.core.containers import ModelContainer

description = """
Esta API realiza predições de preços de fechamento futuro de ações utilizando um modelo de Deep Learning **LSTM (Long Short-Term Memory)**.

### Funcionalidades
* **Histórico**: Busca de dados dos últimos dias.
* **Predição**: Cálculo do preço esperado de fechamento para o próximo dia útil.
* **Observabilidade**: Métricas da API expostas ativamente para o formato Prometheus.
"""

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=description,
    version="1.0.0",
    contact={
        "name": "FIAP - ML Engineering",
        "url": "https://github.com/michelhilg/fiap-ml-tech-challenge-4",
    },
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)

# Carregar recursos na inicialização
@app.on_event("startup")
def startup_event():
    print("Iniciando... Carregando recursos do modelo.")
    try:
        ModelContainer.load_resources()
    except Exception as e:
        print(f"Falha ao carregar recursos: {e}")

app.include_router(prediction.router, prefix=settings.API_V1_STR, tags=["prediction"])

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
