# API de Previsão de Ações (Stock Prediction API) com LSTM

Este repositório contém o código-fonte desenvolvido para a **Fase 4 do Tech Challenge** da Pós-Graduação em Machine Learning Engineering (FIAP). O objetivo do projeto é desenvolver e implantar um modelo preditivo baseado em **Redes Neurais Recorrentes (LSTM)** para prever os movimentos de fechamento de ações diariamente, disponibilizado por meio de uma API estruturada e servido com observabilidade.

O projeto foi construído sobre uma arquitetura moderna e escalável utilizando **FastAPI** e **TensorFlow/Keras**.

---

## Estrutura do Projeto

O modelo e a API estão divididos da seguinte forma para facilitar o desenvolvimento, deploy e experimentos:

- `app/`: Aplicação principal em FastAPI, totalmente modularizada.
  - `api/v1/endpoints/`: Contém as rotas da API (ex: `/predict`, `/history`).
  - `core/`: Configurações centrais do projeto.
  - `models/`: Definições de schemas Pydantic (Validação de Dados de Entrada e Saída).
  - `services/`: Regras de negócio, busca de dados via `yfinance` e pré-processamento.
- `notebook/`: **[Descoberta e Treinamento]** Contém o notebook base (`fiap-last-project.ipynb`) utilizado para prototipação, análise exploratória (EDA), criação dos indicadores técnicos e treinamento inicial da rede LSTM.
- `data/`: Aloca os artefatos gerados pelo notebook e que são consumidos pela API:
  - `lstm_stock_model_fixed.keras` (O modelo treinado compilado).
  - `preprocessor.pkl` (O preenchedor/padronizador baseado no `scikit-learn`).

---

## Funcionalidades e Requisitos Atendidos

1. **A Escolha do Ticker (`SPY`)**: O modelo foi treinado tendo como base principal o índice **S&P 500 ETF Trust (SPY)**. A escolha se deve à altíssima liquidez, robustez contra volatilidades isoladas de uma única empresa e ao seu vasto histórico de dados, tornando as predições do LSTM mais consistentes.
2. **Desenvolvimento do Modelo**: Utilizamos uma rede LSTM treinada para entender dependências temporais em dados de séries financeiras.
   > **Nota sobre Dados Históricos Mínimos:** O modelo exige uma janela estrita de dias retroativos de preço para calcular a próxima previsão. A API foi configurada para buscar um *buffer* extra (100 dias de histórico). Isso é **vital** porque os indicadores técnicos fundamentais do modelo (como as médias móveis `MA_50`) exigem blocos de dezenas de dias sequenciais autênticos apenas para começarem a funcionar. Uma vez calculados, a LSTM consome os 60 dias mais recentes de dados e os estabiliza para não gerar "ruídos" decorrentes de falta de histórico.
3. **Dados Históricos em Tempo Real**: O endpoint realiza a coleta atualizada dos dados da bolsa (via módulo `yfinance`), dispensando a necessidade de arquivos CSVs manuais pelo usuário no momento da predição.
4. **API Preditiva**: Criada com FastAPI. Suporta checagem de saúde (`/health`), busca de histórico de operações (`/api/v1/history`) e inferência do próximo encerramento para o ticker fornecido (`/api/v1/predict`).
5. **Monitoramento de Performance**: Implementado via `prometheus-fastapi-instrumentator`. A API expõe nativamente uma rota `/metrics` no formato texto para que soluções robustas (como o ecossistema Prometheus + Grafana) possam medir tempos de resposta, throughput e utilização.

---

## Como Executar Localmente

### Pré-requisitos
- Python 3.9+
- Gerenciador de pacotes (`uv` ou `pip`)

### 1. Clonando e preparando o ambiente

```bash
# Clone o repositório
git clone https://github.com/SeuUsuario/fiap-ml-tech-challenge-4.git
cd fiap-ml-tech-challenge-4/ml_api

# Crie e ative um ambiente virtual (Opcional, mas recomendado)
# Windows: .venv\Scripts\activate | Mac/Linux: source .venv/bin/activate
python -m venv .venv

# Instale as dependências
pip install -r requirements.txt
# Ou utilizando uv, caso prefira: uv pip install -r requirements.txt
```

### 2. Inicializando a API

A API foi projetada para iniciar diretamente utilizando o servidor ASGI Uvicorn:

```bash
python -m app.main
```

Se preferir rodar via CLI diretamente:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Quando você visualizar `Application startup complete`, o modelo de Deep Learning (Keras) e o Scikit-Learn e Preprocessor (`.pkl`) já terão sido carregados em memória usando o ciclo de vida dinâmico (`lifespan`) do FastAPI.

### 3. Utilizando e Testando as Rotas

> **Acesso Web (Swagger UI):** Acesse interativamente pela URL: `http://localhost:8000/docs`.

**Fazendo Inferencia via cURL (Predição):**

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "ticker": "SPY"
}'
```

**Checando Indicadores (Métricas Prometheus):**

A API já expõe nativamente uma rota `/metrics` no formato texto do Prometheus.

#### Monitorando com Prometheus e Grafana

Para criar um acompanhamento profissional da API (Uso de CPU, Memória, Latência das Requisições):

1. **Prometheus (Porta 9090):** Configure o job do Prometheus para escutar os *targets* da API (Ex: `localhost:8000`). Acesse `http://localhost:9090` para verificar se o target está ativo.
2. **Grafana (Porta 3000):** Acesse `http://localhost:3000`, adicione o Prometheus como sua fonte de dados (*Data Source*).
3. **Dashboards:** Crie novos painéis consumindo as métricas do Prometheus construídas pelo `prometheus-fastapi-instrumentator`.

---

## Considerações do Tech Challenge
A implementação focou em **Boas Práticas de Engenharia de Software** e modularização, garantindo que o modelo treinado (Keras) se acople à arquitetura web sem fricções. Foram descartadas práticas defasadas de importação contínua global; agora os dados operam de forma isolada na classe `app.state`, deixando a API extensível e veloz.
