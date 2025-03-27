# SEC Risk Factor Tracker

A quantitative framework for analyzing risk factors in SEC 10-K filings using large language models, designed with industry-standard methodologies mirroring hedge fund workflows.

## Overview

This project extracts, processes, and analyzes the "Item 1A: Risk Factors" and "Item 7: Management's Discussion & Analysis" sections from SEC 10-K filings to identify emerging risks, measure risk factor novelty, and predict market volatility. The system is built to address common quant trading challenges like factor decay and adverse selection.

## Key Features

- **Data Pipeline**: Extracts and processes text from SEC EDGAR 10-K filings (2010-2024)
- **NLP Analysis**: Clusters risks via BERTopic and scores novelty using cosine similarity
- **Retrieval-Augmented Generation**: Contextualizes new filings with historical risk factors
- **Predictive Modeling**: Targets 30-day post-filing stock volatility and alpha decay
- **Backtesting Framework**: Implements walk-forward analysis with factor decay adjustments
- **Production-Ready API**: Provides real-time risk scoring via FastAPI

## Project Architecture

```
sec-risk-tracker/
├── data/
│   ├── raw/                # Raw 10-K filings (PDF/HTML)
│   ├── processed/          # Cleaned text, risk factor embeddings
│   └── lobster/            # Limit Order Book data (for adverse selection)
├── models/
│   ├── finbert/            # Fine-tuned FinBERT model
│   └── lightgbm/           # Trained GBDT models
├── notebooks/
│   ├── 1_sec_data_pipeline.ipynb
│   ├── 2_risk_clustering.ipynb
│   └── 3_backtesting.ipynb
├── scripts/
│   ├── data_pipeline.py    # SEC-API + PDF extraction
│   ├── feature_engineer.py # QuantLib integration
│   └── deploy_api.py       # FastAPI endpoint
├── config/
│   └── sectors.yml         # Target sectors configuration
└── docker/
    └── Dockerfile          # Containerization (MLflow + FastAPI)
```

## Methodology

### 1. Data Pipeline

- **Source**: SEC EDGAR 10-K filings via sec-api.io 
- **Extraction**: Uses pdfplumber to parse Item 1A (Risk Factors) and Item 7 (MD&A)
- **Storage**: Compressed Parquet files in AWS S3

### 2. NLP & Feature Engineering

- **LLM Integration**: Utilizes financial LLMs (BloombergGPT/FinBERT) to:
  - Cluster risks into categories (e.g., "Supply Chain," "Regulatory")
  - Score novelty by comparing risk factors to prior years
- **RAG Implementation**:
  - Retrieves similar historical risk factors to contextualize new filings
  - Fine-tunes an LLM (Llama-3, BloombergGPT) for risk classification

### 3. Predictive Modeling

- **Target Variables**: 30-day post-filing stock volatility (IVOL) and alpha decay
- **Model Implementation**: LightGBM with SHAP explainability
- **Features**: Risk cluster prevalence, sentiment scores, sector-specific keywords
- **Adverse Selection Mitigation**: Simulates order book impact using LOBSTER data

### 4. Backtesting & Validation

- **Strategy**: Long stocks with declining risk scores, short deteriorating ones
- **Factor Decay Fix**: Walk-forward analysis with rolling 3-year training windows
- **Benchmarks**: SPY, sector ETFs

### 5. Deployment

- **API**: FastAPI endpoint for real-time risk scoring
- **Monitoring**: MLflow tracking for model drift

## Tools & Technology Stack

| Task | Tools |
|------|-------|
| Text Extraction | pdfplumber, sec-api.io |
| NLP | spaCy, FinBERT, BERTopic |
| Feature Store | AWS S3 + Delta Lake |
| Modeling | LightGBM, SHAP, QuantLib |
| Backtesting | Backtrader, vectorized Pandas |
| Deployment | Docker, FastAPI, MLflow |

## Dependencies

Major dependencies include:

```
finbert==1.3.2                # Bloomberg's financial BERT
bertopic==0.15.0              # Risk clustering
spacy==3.7.4                  # Entity extraction
quantlib==1.31                # Factor mapping
backtrader==1.9.76.123        # Institutional backtester
lightgbm==4.1.0               # GBDT implementation
shap==0.44.1                  # Model explainability
sec-api==2.6.1                # SEC EDGAR client
fastapi==0.109.0              # Production API
mlflow==2.9.2                 # Model monitoring
```

## Expected Results

- **Alpha Signal**: Risk factor-driven strategy with Sharpe > 1.5 and controlled drawdowns (<15%)
- **Factor Decay Analysis**: Performance comparison between 1-month vs. 6-month strategies
- **Adverse Selection Mitigation**: Improved PnL with LOBSTER liquidity filters
- **Explainability**: SHAP plots highlighting top risk drivers

## References

The project draws inspiration from academic literature on using LLMs for financial document analysis, including:

- "A Scalable Data-Driven Framework for Systematic Analysis of SEC 10-K Filings Using Large Language Models" (Daimi & Iqbal, 2024)
- "Enhancing Financial Sentiment Analysis via Retrieval Augmented Large Language Models" (Zhang et al., 2023)

## License

MIT Liscense

## Contributors

Lawrence Lin 