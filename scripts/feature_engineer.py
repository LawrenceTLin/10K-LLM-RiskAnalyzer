#!/usr/bin/env python3
"""
Feature Engineering Script for SEC Risk Factor Analysis

Reads processed sentence-level data, generates NLP features (embeddings, topics, novelty),
aggregates features to the filing level, and saves the final feature set.

Future Work:
- Integrate configuration loading from YAML (e.g., config/parameters.yml).
- Add support for processing MD&A (Item 7) in addition to Item 1A.
- Implement caching for embeddings to avoid re-computation.
- Explore alternative sentence splitting methods (e.g., spaCy, NLTK) for robustness.
- Enhance text cleaning (e.g., handling complex tables, specific SEC artifacts).
- Parallelize data loading and processing using Dask or Spark for larger datasets.
"""

import logging
import time
from pathlib import Path
import pandas as pd
import numpy as np
# import yaml # For loading config if needed
from typing import List, Dict, Optional, Tuple, Any # Added typing

# --- NLP Libraries (Import conditionally or ensure installed) ---
# Ensure these are in requirements.txt: sentence-transformers, bertopic, umap-learn, hdbscan, scikit-learn
NLP_LIBRARIES_AVAILABLE = True
try:
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    # Future Work: Import libraries for more advanced NLP like spaCy for NER/dependency parsing.
except ImportError as e:
    NLP_LIBRARIES_AVAILABLE = False
    print(f"WARNING: Required NLP library missing ({e}). Clustering, Embedding, Novelty features will fail.")
    SentenceTransformer, BERTopic, UMAP, HDBSCAN, cosine_similarity = None, None, None, None, None
# --- End NLP Libraries ---

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("feature_engineer_nlp.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)
# --- End Logging Setup ---


# --- Configuration ---
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
     BASE_DIR = Path(".") # Fallback for interactive use

PROCESSED_DIR = BASE_DIR / "data" / "processed" / "sec_filings"
FEATURES_DIR = BASE_DIR / "data" / "features"
MODELS_DIR = BASE_DIR / "models"
BERTOPIC_MODELS_DIR = MODELS_DIR / "bertopic"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
BERTOPIC_MODELS_DIR.mkdir(parents=True, exist_ok=True)

COMPANIES_TO_PROCESS = ["NVDA", "MSFT", "META", "TGT", "JPM", "SCHW", "XOM"]
YEARS_TO_PROCESS = list(range(2021, 2023 + 1))

# Model/Parameter Config
# Future Work: Experiment with different embedding models (e.g., domain-specific like FinBERT adapted for sentences, larger models like mpnet, multilingual models).
SENTENCE_MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_BATCH_SIZE = 128
# Future Work: Add support for multi-GPU embedding generation.
try:
     import torch
     DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
     DEVICE = 'cpu'
     print("WARNING: PyTorch not found. Embeddings will run on CPU.")

# Future Work: Extensive hyperparameter tuning for BERTopic (n_neighbors, n_components, min_dist, min_cluster_size, cluster_selection_method).
BERTOPIC_NR_TOPICS = "auto"
BERTOPIC_MIN_TOPIC_SIZE = 15
BERTOPIC_CALCULATE_PROBABILITIES = False # Set True if needed for downstream tasks, but slower.
# Future Work: Explore dynamic topic modeling (DTM) within BERTopic to track topic evolution over time explicitly.
# --- End Configuration ---


# =============================================================================
# Helper Functions
# =============================================================================

def load_processed_data(ticker: str, years: List[int], data_dir: Path) -> Optional[pd.DataFrame]:
    """Loads and concatenates processed parquet files for a ticker and years."""
    # Future Work: Add error handling for reading potentially corrupt Parquet files.
    # Future Work: Select only necessary columns during load to save memory.
    all_dfs = []
    logger.info(f"Loading processed data for {ticker}, years {min(years)}-{max(years)}...")
    for year in years:
        file_path = data_dir / ticker / f"{ticker}_{year}_risk.parquet"
        if file_path.exists():
            try:
                df_year = pd.read_parquet(file_path)
                # Validate essential columns
                essential_cols = ['ticker', 'year', 'paragraph_idx', 'sentence_idx', 'sentence']
                if not all(col in df_year.columns for col in essential_cols):
                     logger.warning(f"File {file_path} missing essential columns {essential_cols}. Skipping.")
                     continue
                df_year['year'] = df_year['year'].astype(int)
                df_year['ticker'] = ticker # Standardize ticker column
                all_dfs.append(df_year)
            except Exception as e:
                logger.error(f"Failed to load or process {file_path}: {e}")
        else:
            logger.warning(f"Processed file not found: {file_path}")

    if not all_dfs:
        logger.warning(f"No processed data loaded for {ticker}.")
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)
    # Add a unique ID
    combined_df['sentence_uid'] = combined_df['ticker'] + '_' + combined_df['year'].astype(str) + '_' + combined_df['paragraph_idx'].astype(str) + '_' + combined_df['sentence_idx'].astype(str)
    logger.info(f"Loaded {len(combined_df)} sentences for {ticker}.")
    return combined_df


def generate_embeddings(sentences: List[str], model_name: str, batch_size: int = 64, device: str = 'cpu') -> Optional[np.ndarray]:
    """Generates sentence embeddings using SentenceTransformer."""
    # Future Work: Implement caching mechanism (e.g., using joblib or hashing) to avoid re-generating embeddings for unchanged sentences/files.
    # Future Work: Explore fine-tuning the embedding model on financial text corpus for potentially better domain adaptation.
    if not NLP_LIBRARIES_AVAILABLE or not SentenceTransformer:
        logger.error("sentence-transformers library not available.")
        return None
    if not sentences:
        logger.warning("No sentences provided for embedding generation.")
        return np.array([])

    try:
        logger.info(f"Loading sentence embedding model: {model_name} onto device: {device}...")
        model = SentenceTransformer(model_name, device=device)
        logger.info(f"Generating {len(sentences)} embeddings (batch size: {batch_size})...")
        embeddings = model.encode(sentences, show_progress_bar=True, batch_size=batch_size, device=device)
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        if embeddings.size == 0 or np.all(embeddings == 0):
             logger.warning("Embedding generation resulted in empty or all-zero array.")
             return None
        return embeddings.astype(np.float32) # Ensure consistent type (float32 often sufficient)
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
        return None


def run_bertopic_clustering(docs: List[str], embeddings: np.ndarray, model_save_dir: Path) -> Tuple[Optional[Any], Optional[np.ndarray]]:
    """Runs BERTopic clustering on documents and pre-computed embeddings."""
    # Future Work: Explore hierarchical topic modeling using BERTopic.
    # Future Work: Experiment with different dimensionality reduction (UMAP params) and clustering algorithms (HDBSCAN params).
    # Future Work: Implement topic quality evaluation metrics (e.g., coherence scores like NPMI/C_V).
    # Future Work: Consider using guided topic modeling if prior knowledge about risk categories exists.
    if not NLP_LIBRARIES_AVAILABLE or not BERTopic or not UMAP or not HDBSCAN:
        logger.error("bertopic library (or dependencies) not available.")
        return None, None
    if embeddings is None or embeddings.size == 0 or embeddings.ndim != 2:
        logger.error("Valid 2D embeddings are required for BERTopic.")
        return None, None
    if len(docs) != embeddings.shape[0]:
        logger.error(f"Mismatch between number of docs ({len(docs)}) and embeddings ({embeddings.shape[0]}).")
        return None, None

    try:
        logger.info("Initializing BERTopic components...")
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42, low_memory=True)
        hdbscan_model = HDBSCAN(min_cluster_size=BERTOPIC_MIN_TOPIC_SIZE, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

        logger.info(f"Running BERTopic (nr_topics={BERTOPIC_NR_TOPICS}, min_topic_size={BERTOPIC_MIN_TOPIC_SIZE})...")
        topic_model = BERTopic(
            embedding_model="passthrough",
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            nr_topics=BERTOPIC_NR_TOPICS,
            calculate_probabilities=BERTOPIC_CALCULATE_PROBABILITIES,
            verbose=True
            # Future Work: Pass `vectorizer_model` if custom stop words or n-grams are needed for topic representation.
        )
        topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)

        num_topics_found = len(topic_model.get_topic_info()) - 1
        logger.info(f"BERTopic processing complete. Found {num_topics_found} topics (excluding outliers).")

        # Save model
        model_save_path = model_save_dir / f"bertopic_model_{time.strftime('%Y%m%d_%H%M%S')}"
        topic_model.save(str(model_save_path), serialization="safetensors", save_embedding_model=False)
        logger.info(f"Saved BERTopic model to {model_save_path}")

        return topic_model, np.array(topics)
    except Exception as e:
        logger.error(f"Failed to run BERTopic: {e}", exc_info=True)
        return None, None


def calculate_novelty(current_embeddings: np.ndarray, previous_embeddings: np.ndarray) -> Optional[np.ndarray]:
    """Calculates novelty score (1 - max_cosine_similarity) against previous period."""
    # Future Work: Explore alternative novelty measures (e.g., distance to cluster centroids, average similarity instead of max).
    # Future Work: Consider comparing against a longer history (e.g., previous N years) instead of just Y-1.
    if not NLP_LIBRARIES_AVAILABLE or not cosine_similarity:
        logger.error("scikit-learn not available for cosine_similarity.")
        return None
    if current_embeddings is None or previous_embeddings is None or \
       current_embeddings.size == 0 or previous_embeddings.size == 0 or \
       current_embeddings.ndim != 2 or previous_embeddings.ndim != 2:
        logger.warning("Invalid or empty embeddings provided for novelty calculation.")
        return None
    if current_embeddings.shape[1] != previous_embeddings.shape[1]:
        logger.error(f"Embedding dimensions mismatch: {current_embeddings.shape[1]} vs {previous_embeddings.shape[1]}")
        return None

    try:
        logger.debug(f"Calculating novelty: Comparing {current_embeddings.shape[0]} current vs {previous_embeddings.shape[0]} previous sentences...")
        sim_matrix = cosine_similarity(current_embeddings, previous_embeddings)
        max_similarities = np.max(sim_matrix, axis=1)
        novelty_scores = 1.0 - max_similarities
        novelty_scores = np.clip(novelty_scores, 0.0, 1.0).astype(np.float32) # Ensure float32 and bounds
        logger.debug("Novelty scores calculated.")
        return novelty_scores
    except Exception as e:
        logger.error(f"Failed to calculate novelty: {e}", exc_info=True)
        return None


def aggregate_features_to_filing(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Aggregates sentence-level features to the filing level (ticker, year)."""
    # Future Work: Engineer more complex features (e.g., interaction terms between sentiment and topic).
    # Future Work: Calculate changes (deltas) in aggregated features year-over-year directly in this step.
    # Future Work: Explore different aggregation methods (e.g., weighted averages based on sentence position or emphasis).
    if df is None or df.empty:
        logger.warning("Cannot aggregate features from empty DataFrame.")
        return None
    logger.info("Aggregating features to filing (ticker-year) level...")

    required_cols = ['ticker', 'year', 'sentiment_score', 'word_count', 'contains_risk', 'contains_uncertainty']
    optional_cols = ['novelty_score', 'topic_id']
    available_cols = [col for col in required_cols + optional_cols if col in df.columns]

    if not all(col in available_cols for col in required_cols):
         logger.error(f"Missing essential columns for aggregation. Need at least: {required_cols}, Have: {df.columns.tolist()}")
         return None

    try:
        agg_funcs = {
            'sentence': 'count', # -> total_sentences
            'word_count': ['sum', 'mean'], # -> total_words, avg_sentence_length
            'sentiment_score': ['mean', 'std', 'min', 'max', lambda x: (x < -0.1).mean(), lambda x: (x > 0.1).mean()],
            'contains_risk': 'mean', # -> risk_mention_pct
            'contains_uncertainty': 'mean', # -> uncertainty_mention_pct
        }
        if 'novelty_score' in available_cols:
            # Aggregate only non-NaN novelty scores
            agg_funcs['novelty_score'] = ['mean', 'std', 'max', lambda x: (x > 0.5).mean() if pd.notna(x).any() else 0.0]
        if 'topic_id' in available_cols:
             # Aggregate only non-error/-outlier topics for nunique? Or include -1?
             agg_funcs['topic_id'] = [lambda x: x[x != -1 & x != -2].nunique()]

        # Perform initial aggregation
        aggregated_df = df.groupby(['ticker', 'year']).agg(agg_funcs).copy()

        # --- Flatten MultiIndex Columns ---
        new_cols = []
        for col in aggregated_df.columns:
            prefix = col[0]
            suffix = col[1]
            if isinstance(suffix, str) and '<lambda' in suffix:
                 if prefix == 'sentiment_score': new_name = f"pct_sentiment_{'neg' if 'x < -0.1' in suffix else 'pos'}"
                 elif prefix == 'novelty_score': new_name = f"pct_novelty_gt_0_5"
                 elif prefix == 'topic_id': new_name = "num_distinct_topics"
                 else: new_name = f"{prefix}_lambda"
            else: new_name = f"{prefix}_{suffix}"
            new_cols.append(new_name)
        aggregated_df.columns = new_cols
        # --- End Flatten ---

        # Rename columns
        rename_map = {
            'sentence_count': 'total_sentences',
            'word_count_sum': 'total_words',
            'word_count_mean': 'avg_sentence_length',
            'contains_risk_mean': 'risk_mention_pct',
            'contains_uncertainty_mean': 'uncertainty_mention_pct',
            # 'topic_id_<lambda>': 'num_distinct_topics' # Handled during flattening now
        }
        aggregated_df = aggregated_df.rename(columns=rename_map)

        # Scale percentages
        for col in ['risk_mention_pct', 'uncertainty_mention_pct', 'pct_sentiment_neg', 'pct_sentiment_pos', 'pct_novelty_gt_0_5']:
             if col in aggregated_df.columns: aggregated_df[col] = (aggregated_df[col] * 100).round(2)

        # --- Add Topic Distribution Features ---
        if 'topic_id' in df.columns and (df['topic_id'] != -2).any(): # Check topics were generated
            logger.info("Calculating topic distributions...")
            # Normalize topic counts to get proportions
            topic_proportions = pd.crosstab(index=[df['ticker'], df['year']],
                                            columns=df['topic_id'],
                                            normalize='index')
            # Rename columns
            topic_proportions.columns = [f"topic_{'outlier' if col == -1 else col}_prop" for col in topic_proportions.columns]
            # Multiply by 100 and round
            topic_proportions = (topic_proportions * 100).round(2)
            # Merge topic proportions
            aggregated_df = aggregated_df.join(topic_proportions, on=['ticker', 'year'])
            logger.info("Topic distributions added.")
        # --- End Topic Distribution ---

        aggregated_df = aggregated_df.reset_index()
        # Fill NaNs resulting from aggregations (e.g., std dev on single value)
        aggregated_df = aggregated_df.fillna(0)
        logger.info(f"Aggregation complete. Feature shape: {aggregated_df.shape}")
        return aggregated_df

    except Exception as e:
        logger.error(f"Failed during feature aggregation: {e}", exc_info=True)
        return None

# =============================================================================
# Main Orchestration Function
# =============================================================================

def main():
    """Main processing function for NLP feature engineering."""
    start_time = time.time()
    logger.info("--- Starting NLP Feature Engineering Pipeline ---")

    if not NLP_LIBRARIES_AVAILABLE:
         logger.error("Core NLP libraries not found. Cannot perform main feature engineering steps. Exiting.")
         return

    all_ticker_data = []
    # --- Step 1: Load Data ---
    logger.info("Step 1: Loading processed data...")
    for ticker in COMPANIES_TO_PROCESS:
        df = load_processed_data(ticker, YEARS_TO_PROCESS, PROCESSED_DIR)
        if df is not None and not df.empty:
            all_ticker_data.append(df)
    if not all_ticker_data: logger.error("No data loaded. Exiting."); return
    master_df = pd.concat(all_ticker_data, ignore_index=True)
    logger.info(f"Combined data loaded. Total sentences: {len(master_df)}. Shape: {master_df.shape}")

    # --- Step 2: Generate Embeddings ---
    logger.info("Step 2: Generating sentence embeddings...")
    if 'sentence' not in master_df.columns or master_df['sentence'].isnull().any():
         logger.warning("Sentence column missing or contains nulls. Filling with empty string.")
         master_df['sentence'] = master_df['sentence'].fillna('')
    sentences_list = master_df['sentence'].tolist()
    embeddings = generate_embeddings(sentences_list, SENTENCE_MODEL_NAME, EMBEDDING_BATCH_SIZE, DEVICE)
    # Future Work: Add option to load pre-saved embeddings here.
    if embeddings is None:
        logger.error("Embedding generation failed. Subsequent steps requiring embeddings will be skipped.")
        master_df['topic_id'] = -2; master_df['novelty_score'] = np.nan
    else: logger.info("Embeddings generated successfully.")

    # --- Step 3: Calculate Novelty ---
    logger.info("Step 3: Calculating novelty scores...")
    master_df['novelty_score'] = np.nan # Initialize
    if embeddings is not None:
        # Create mapping for faster index lookup if DataFrame index is not sequential 0..N-1
        master_df = master_df.reset_index(drop=True) # Ensure sequential index matches embeddings order
        # Iterate through tickers and years
        for ticker in master_df['ticker'].unique():
            ticker_years = sorted(master_df[master_df['ticker'] == ticker]['year'].unique())
            for i, current_year in enumerate(ticker_years):
                if i == 0: continue
                previous_year = ticker_years[i-1]
                current_indices = master_df.index[(master_df['ticker'] == ticker) & (master_df['year'] == current_year)].tolist()
                prev_indices = master_df.index[(master_df['ticker'] == ticker) & (master_df['year'] == previous_year)].tolist()
                if not current_indices or not prev_indices: continue
                current_embs = embeddings[current_indices]; prev_embs = embeddings[prev_indices]
                novelty_scores_year = calculate_novelty(current_embs, prev_embs)
                if novelty_scores_year is not None: master_df.loc[current_indices, 'novelty_score'] = novelty_scores_year
                else: logger.warning(f"Novelty calculation failed for {ticker} year {current_year}.")
        logger.info("Novelty score calculation completed.")
    else: logger.warning("Skipping novelty calculation (no embeddings).")

    # --- Step 4: Run BERTopic Clustering ---
    logger.info("Step 4: Running BERTopic clustering...")
    master_df['topic_id'] = -2 # Default state
    topic_model = None
    if embeddings is not None:
        topic_model, topics = run_bertopic_clustering(sentences_list, embeddings, BERTOPIC_MODELS_DIR)
        if topics is not None: master_df['topic_id'] = topics; logger.info("Added topic IDs to DataFrame.")
        else: logger.warning("BERTopic clustering failed. topic_id set to -2.")
    else: logger.warning("Skipping BERTopic clustering (no embeddings).")

    # --- Step 5: Aggregate Features ---
    logger.info("Step 5: Aggregating features to filing level...")
    filing_features_df = aggregate_features_to_filing(master_df)

    # --- Step 6: Save Aggregated Features & Topic Info ---
    logger.info("Step 6: Saving results...")
    if filing_features_df is not None and not filing_features_df.empty:
        features_path = FEATURES_DIR / "filing_nlp_features.parquet"
        try:
            # Future Work: Consider schema enforcement for feature Parquet file.
            filing_features_df.to_parquet(features_path, index=False)
            logger.info(f"Successfully saved aggregated features to: {features_path}")
        except Exception as e: logger.error(f"Failed to save aggregated features: {e}", exc_info=True)
    else: logger.error("Feature aggregation failed. No features saved.")

    if topic_model is not None:
        try:
            topic_info = topic_model.get_topic_info()
            topic_info_path = FEATURES_DIR / "topic_info.csv"
            topic_info.to_csv(topic_info_path, index=False)
            logger.info(f"Saved topic info to {topic_info_path}")
            # Future Work: Save topic hierarchy, visualizations (intertopic distance map), etc.
        except Exception as e: logger.error(f"Failed to save topic info: {e}")

    end_time = time.time()
    logger.info(f"--- NLP Feature Engineering Pipeline Finished ---")
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()