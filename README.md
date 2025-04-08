# SEC Risk Factor Quant Framework

**A quantitative framework for constructing and backtesting predictive signals derived from SEC 10-K risk disclosures (Item 1A) and MD&A (Item 7) using NLP and machine learning.** This project mirrors systematic hedge fund workflows for transforming unstructured text data into actionable trading insights, specifically addressing factor decay and implementation shortfall.

## Overview

This project implements an end-to-end pipeline to:
1.  **Extract and process** text from SEC EDGAR 10-K filings (Item 1A, Item 7) via sec-api.io.
2.  **Engineer quantitative features** from the text using NLP techniques (sentiment analysis via FinBERT, topic modeling via BERTopic, embedding-based novelty scoring).
3.  **Develop predictive models** (LightGBM) targeting forward-looking stock volatility and potential alpha signals based on the engineered text features.
4.  **Rigorously backtest** trading strategies derived from these models using industry-standard methodologies (walk-forward analysis, transaction cost simulation, factor decay analysis).
5.  **Simulate liquidity constraints** and adverse selection impact using high-frequency LOBSTER data.
6.  Provide a framework for deployment (API) and monitoring (MLflow).

## Key Quantitative Features & Methodologies

*   **Systematic Data Pipeline**: Automated extraction and processing ensures reproducibility and scalability (Targeting 2010-2024 initially).
*   **NLP Feature Engineering**:
    *   **Sentiment Dynamics**: Calculates not just average sentiment (FinBERT), but also sentiment *volatility* and *distribution shifts* within risk sections as potential factors.
    *   **Topic Modeling (BERTopic)**: Clusters risks dynamically and quantifies the *proportion* of discussion allocated to key themes (e.g., Regulatory, Supply Chain, Competition) year-over-year. Changes in topic concentration serve as input features.
    *   **Novelty Scoring**: Measures the textual difference (1 - max cosine similarity of sentence embeddings) between the current filing's risks and the prior year's, identifying potentially unpriced information.
*   **Predictive Modeling**:
    *   **Targets**: 30-day post-filing realized volatility and residual returns (alpha) relative to a factor model.
    *   **Model**: LightGBM chosen for performance and efficiency, with SHAP for feature importance analysis.
    *   **Feature Set**: Combines NLP-derived features (sentiment dynamics, topic shifts, novelty scores) with market-based features (e.g., historical volatility, beta) and sector controls.
*   **Robust Backtesting (Backtrader & Vectorized)**:
    *   **Walk-Forward Analysis**: Employs rolling training windows (e.g., 3-years) and out-of-sample testing periods to mitigate lookahead bias and adapt to changing market regimes.
    *   **Factor Decay Analysis**: Explicitly compares strategy performance based on signal holding periods (e.g., 1-month vs. 3-month rebalancing) to quantify decay.
    *   **Transaction Cost & Slippage**: Incorporates estimated trading costs and slippage based on historical spreads / volatility.
    *   **Adverse Selection Simulation (LOBSTER)**: Uses limit order book data to estimate potential market impact cost for strategy trades, adjusting PnL for implementation shortfall.
*   **QuantLib Integration**: Used for potential financial calculations or factor mapping (e.g., aligning risk factors to standard risk model categories).
*   **RAG for Context (Current Scope)**: Implements Retrieval-Augmented Generation primarily for *retrieving* similar historical risk disclosures to provide context during analysis, rather than large-scale fine-tuning for classification in the initial phase due to data constraints. *(Self-Correction: Fine-tuning large models like Llama-3/BloombergGPT requires substantial data beyond the initial scope; this is deferred.)*

## Project Architecture
sec-risk-quant/ # Renamed for clarity
├── data/
│ ├── raw/
│ │ └── sec_filings/ # Raw 10-K text/HTML via API
│ ├── processed/
│ │ └── sec_filings/ # Sentence-level data, embeddings, sentiment (Parquet)
│ ├── features/ # Aggregated filing-level features (Parquet)
│ └── lobster/ # LOBSTER sample data (if used locally)
├── models/
│ ├── bertopic/ # Saved BERTopic models
│ └── lightgbm/ # Trained LightGBM models & SHAP explainers
├── notebooks/ # Exploratory analysis, visualization
│ ├── 1_Data_Exploration.ipynb
│ ├── 2_NLP_Feature_Analysis.ipynb
│ ├── 3_Model_Training_Validation.ipynb
│ └── 4_Backtest_Analysis.ipynb
├── scripts/
│ ├── sec_ingestion.py # sec-api.io interaction, text processing
│ ├── feature_engineer_nlp.py # Embedding, Topic Modeling, Novelty, Aggregation
│ ├── train_predictive_model.py # LightGBM training and prediction pipeline
│ ├── backtest_strategy.py # Backtrader/vectorized backtesting logic
│ └── deploy_api.py # Optional: FastAPI endpoint structure
├── config/
│ └── parameters.yml # Model/backtest parameters, file paths
└── docker/ # Optional: Docker setup for API/MLflow
└── Dockerfile

*(Adjusted directory names for clarity)*

## Tools & Technology Stack

| Category | Tools & Libraries | Purpose |
|---|---|---|
| Data Acquisition | `sec-api` | SEC EDGAR API client |
| Data Processing | `pandas`, `pyarrow`, `numpy` | Data manipulation, storage |
| Core NLP | `transformers` (FinBERT), `sentence-transformers` | Sentiment, Embeddings |
| Topic Modeling | `bertopic`, `umap-learn`, `hdbscan` | Risk Clustering |
| ML Modeling | `lightgbm`, `scikit-learn`, `shap` | Prediction, Feature Importance |
| Backtesting | `backtrader` or Vectorized `pandas` | Strategy simulation |
| Quant Finance | `quantlib-python` | Optional: Financial calculations |
| API Deployment | `fastapi`, `uvicorn` | Real-time scoring endpoint |
| MLOps | `mlflow` | Experiment tracking, model logging |
| Infrastructure | `docker`, AWS S3 (optional) | Containerization, Cloud Storage |
| Core Utils | `python-dotenv`, `tenacity`, `tqdm`, `pyyaml` | Config, retries, progress |

*(Grouped tools by category)*

## Dependencies

Key packages listed in `requirements.txt`. Includes `pandas`, `pyarrow`, `scikit-learn`, `transformers`, `sentence-transformers`, `bertopic`, `lightgbm`, `shap`, `sec-api`, `backtrader` (or equivalent), `fastapi`, `mlflow`, `quantlib-python`.

## Potential Alpha Signals & Expected Results

*   **Volatility Prediction**: Demonstrate that NLP features (sentiment volatility, topic shifts, novelty) significantly improve predictions of 30-day realized volatility compared to baseline models using only historical volatility. Target R-squared improvement > 5-10%.
*   **Risk-Based Strategy**: Construct a long-short equity strategy based on a composite score derived from predictive model outputs (e.g., predicted low vol/improving risk vs. high vol/deteriorating risk). Target Sharpe Ratio > 1.0-1.5 (post-cost) in walk-forward backtests.
*   **Factor Decay**: Quantify the decay of the NLP-derived signals by comparing performance across different rebalancing frequencies (e.g., monthly vs. quarterly).
*   **Implementation Shortfall Analysis**: Show the estimated PnL impact of incorporating liquidity constraints using LOBSTER data simulation.
*   **Explainability**: Utilize SHAP plots to identify the most influential NLP features driving volatility predictions and strategy performance in different periods.

## References

*(Keep relevant references)*
- "A Scalable Data-Driven Framework for Systematic Analysis of SEC 10-K Filings Using Large Language Models" (Daimi & Iqbal, 2024)
- "Enhancing Financial Sentiment Analysis via Retrieval Augmented Large Language Models" (Zhang et al., 2023)
- Relevant papers on text-based factors, volatility forecasting, alpha decay, and transaction cost analysis.

## License

MIT License

## Contributors

Lawrence Lin