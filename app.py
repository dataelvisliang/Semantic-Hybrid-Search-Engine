"""
Hybrid Semantic Search Engine - Multi-strategy search with diversity
Features: Semantic embeddings + BM25 + Char n-gram TF-IDF + MMR diversity
Configurable via config.yaml
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import yaml
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Tuple
import re


# Load configuration
@st.cache_data
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


# Initialize config first
try:
    CONFIG = load_config()
except FileNotFoundError:
    st.error("❌ config.yaml not found! Please create a config.yaml file in the app directory.")
    st.stop()

# Page config
st.set_page_config(
    page_title=CONFIG['app']['title'],
    page_icon=CONFIG['app']['page_icon'],
    layout="wide"
)

# Initialize session state
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = {}
if 'bm25_indexes' not in st.session_state:
    st.session_state.bm25_indexes = {}
if 'char_ngram_indexes' not in st.session_state:
    st.session_state.char_ngram_indexes = {}
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'reranker' not in st.session_state:
    st.session_state.reranker = None
if 'preview_seed' not in st.session_state:
    st.session_state.preview_seed = 0
if 'search_results' not in st.session_state:
    st.session_state.search_results = None


@st.cache_resource
def load_embedding_model(model_name: str):
    """Load the sentence transformer model"""
    return SentenceTransformer(model_name)


@st.cache_resource
def load_reranker_model():
    """Load the BGE reranker cross-encoder model"""
    return CrossEncoder('BAAI/bge-reranker-base')


@st.cache_data
def load_embeddings_and_metadata(config: Dict[str, Any]):
    """Load pre-computed embeddings and metadata based on config"""
    output_dir = Path(config['embeddings']['output_dir'])
    metadata_path = output_dir / 'metadata.pkl'

    if not metadata_path.exists():
        return None, None, None, None

    # Load all embedding files
    embeddings = {}
    bm25_indexes = {}
    char_ngram_indexes = {}

    for text_col in config['text_columns']:
        internal_name = text_col['name']

        # Load semantic embeddings
        emb_path = output_dir / f'{internal_name}_embeddings.npz'
        if not emb_path.exists():
            st.error(f"❌ Embedding file not found: {emb_path}")
            return None, None, None, None
        emb_data = np.load(emb_path)
        embeddings[internal_name] = emb_data['embeddings']

        # Load BM25 index
        bm25_path = output_dir / f'{internal_name}_bm25.pkl'
        if bm25_path.exists():
            with open(bm25_path, 'rb') as f:
                bm25_indexes[internal_name] = pickle.load(f)

        # Load char n-gram index
        char_ngram_path = output_dir / f'{internal_name}_char_ngram.pkl'
        if char_ngram_path.exists():
            with open(char_ngram_path, 'rb') as f:
                char_ngram_indexes[internal_name] = pickle.load(f)

    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    return embeddings, bm25_indexes, char_ngram_indexes, metadata


def rerank_with_bge(query: str, texts: list, reranker: CrossEncoder) -> list:
    """
    Use BGE reranker cross-encoder to score relevance
    """
    # Create query-text pairs
    pairs = [[query, text] for text in texts]

    # Get scores from the cross-encoder
    scores = reranker.predict(pairs)

    # Normalize scores to 0-1 range using sigmoid-like transformation
    normalized_scores = 1 / (1 + np.exp(-np.array(scores)))

    return normalized_scores.tolist()


def tokenize_for_bm25(text: str) -> List[str]:
    """Simple tokenizer for BM25."""
    tokens = re.findall(r'\w+', text.lower())
    return tokens


def semantic_search(query_embedding, embeddings, top_k=500):
    """
    Perform cosine similarity search
    """
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_scores = similarities[top_indices]

    return top_indices, top_scores


def bm25_search(query: str, bm25_index, top_k=500):
    """
    Perform BM25 search for keyword matching
    """
    query_tokens = tokenize_for_bm25(query)
    scores = bm25_index.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_scores = scores[top_indices]

    return top_indices, top_scores


def char_ngram_search(query: str, char_ngram_vectorizer, metadata_texts, top_k=500):
    """
    Perform character n-gram search for typo tolerance
    """
    # Transform query
    query_vec = char_ngram_vectorizer.transform([query])

    # Transform all texts
    corpus_vec = char_ngram_vectorizer.transform(metadata_texts)

    # Calculate cosine similarity
    similarities = cosine_similarity(query_vec, corpus_vec)[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_scores = similarities[top_indices]

    return top_indices, top_scores


def hybrid_search(
    query: str,
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    bm25_index,
    char_ngram_vectorizer,
    metadata_texts: List[str],
    top_k: int = 500,
    semantic_weight: float = 0.5,
    bm25_weight: float = 0.3,
    char_ngram_weight: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hybrid search combining semantic, BM25, and char n-gram scores
    """
    num_docs = len(embeddings)

    # Initialize combined scores
    combined_scores = np.zeros(num_docs)

    # 1. Semantic search
    sem_indices, sem_scores = semantic_search(query_embedding, embeddings, top_k=num_docs)
    sem_scores_full = np.zeros(num_docs)
    sem_scores_full[sem_indices] = sem_scores
    combined_scores += semantic_weight * sem_scores_full

    # 2. BM25 search
    if bm25_index is not None:
        bm25_indices, bm25_scores = bm25_search(query, bm25_index, top_k=num_docs)
        # Normalize BM25 scores to 0-1 range
        if bm25_scores.max() > 0:
            bm25_scores_normalized = bm25_scores / bm25_scores.max()
        else:
            bm25_scores_normalized = bm25_scores
        bm25_scores_full = np.zeros(num_docs)
        bm25_scores_full[bm25_indices] = bm25_scores_normalized
        combined_scores += bm25_weight * bm25_scores_full

    # 3. Char n-gram search
    if char_ngram_vectorizer is not None and metadata_texts is not None:
        char_indices, char_scores = char_ngram_search(query, char_ngram_vectorizer, metadata_texts, top_k=num_docs)
        char_scores_full = np.zeros(num_docs)
        char_scores_full[char_indices] = char_scores
        combined_scores += char_ngram_weight * char_scores_full

    # Get top-k results
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    top_scores = combined_scores[top_indices]

    return top_indices, top_scores


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    doc_indices: np.ndarray,
    relevance_scores: np.ndarray,
    top_k: int = 20,
    lambda_param: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply MMR to diversify results

    Args:
        query_embedding: Query embedding vector
        doc_embeddings: All document embeddings
        doc_indices: Indices of candidate documents
        relevance_scores: Relevance scores for candidates
        top_k: Number of diverse results to return
        lambda_param: Trade-off between relevance and diversity (0-1)
                     Higher = more relevance, Lower = more diversity

    Returns:
        Selected indices and their scores
    """
    selected_indices = []
    selected_scores = []
    remaining_indices = list(doc_indices)
    remaining_scores = relevance_scores.copy()

    # Get embeddings for candidate documents
    candidate_embeddings = doc_embeddings[doc_indices]

    for _ in range(min(top_k, len(remaining_indices))):
        if not remaining_indices:
            break

        mmr_scores = []
        for i, idx in enumerate(remaining_indices):
            # Relevance score
            relevance = remaining_scores[i]

            # Diversity score (max similarity to already selected documents)
            if selected_indices:
                selected_embeddings = doc_embeddings[selected_indices]
                current_embedding = doc_embeddings[idx].reshape(1, -1)
                similarities = cosine_similarity(current_embedding, selected_embeddings)[0]
                max_similarity = similarities.max()
            else:
                max_similarity = 0

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            mmr_scores.append(mmr_score)

        # Select document with highest MMR score
        best_idx = np.argmax(mmr_scores)
        selected_indices.append(remaining_indices[best_idx])
        selected_scores.append(remaining_scores[best_idx])

        # Remove selected document from candidates
        remaining_indices.pop(best_idx)
        remaining_scores = np.delete(remaining_scores, best_idx)

    return np.array(selected_indices), np.array(selected_scores)


def aggregate_by_period(df, date_column, period='month'):
    """
    Aggregate results by time period
    """
    df[date_column] = pd.to_datetime(df[date_column])

    if period == 'day':
        df['period'] = df[date_column].dt.strftime('%Y-%m-%d')
    elif period == 'month':
        df['period'] = df[date_column].dt.strftime('%Y-%m')
    else:  # year
        df['period'] = df[date_column].dt.strftime('%Y')

    # Count by period
    period_counts = df.groupby('period').size().reset_index(name='count')
    period_counts = period_counts.sort_values('period')

    return period_counts, df


def create_trend_chart(period_counts, period_type):
    """
    Create trend line/bar chart with improved styling
    """
    fig = go.Figure()

    # Add bar trace with gradient color
    fig.add_trace(go.Bar(
        x=period_counts['period'],
        y=period_counts['count'],
        name='Count',
        marker=dict(
            color=period_counts['count'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Count"),
            line=dict(color='rgba(255,255,255,0.3)', width=1.5)
        ),
        text=period_counts['count'],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text=f"📈 Mentions Over Time ({period_type.capitalize()})",
            font=dict(size=20, color='#2c3e50', family="Arial Black")
        ),
        xaxis=dict(
            title=dict(text=period_type.capitalize(), font=dict(size=14, color='#34495e')),
            gridcolor='rgba(189, 195, 199, 0.3)',
            showgrid=True
        ),
        yaxis=dict(
            title=dict(text="Count", font=dict(size=14, color='#34495e')),
            gridcolor='rgba(189, 195, 199, 0.3)',
            showgrid=True
        ),
        plot_bgcolor='rgba(236, 240, 241, 0.4)',
        paper_bgcolor='white',
        hovermode='x unified',
        margin=dict(t=80, b=60, l=60, r=40)
    )

    return fig


def create_score_distribution_chart(scores, score_display_name="Score"):
    """
    Create pie chart for score distribution
    Args:
        scores: Array of scores from the CSV file
        score_display_name: Display name for the score (from config)
    """
    # Count distribution (for integer ratings like 1-5)
    score_counts = pd.Series(scores).value_counts().sort_index()

    # Generate colors based on number of unique scores
    num_scores = len(score_counts)
    if num_scores <= 5:
        # For ratings 1-5, use color gradient from red to green
        colors = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60'][:num_scores]
    else:
        # For more scores, use a gradient
        colors = [f'rgb({int(231-231*i/num_scores)}, {int(76+180*i/num_scores)}, {int(60)})'
                  for i in range(num_scores)]

    fig = go.Figure(data=[go.Pie(
        labels=[str(label) for label in score_counts.index],
        values=score_counts.values,
        hole=0.4,
        marker=dict(
            colors=colors,
            line=dict(color='white', width=2)
        ),
        textinfo='label+percent',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])

    # Calculate average rating
    avg_score = np.mean(scores)

    fig.update_layout(
        title=dict(
            text=f"🥧 {score_display_name} Distribution",
            font=dict(size=20, color='#2c3e50', family="Arial Black")
        ),
        annotations=[dict(
            text=f'Total: {len(scores)}<br>Avg: {avg_score:.2f}',
            x=0.5, y=0.5,
            font=dict(size=16, color='#34495e', family="Arial"),
            showarrow=False
        )],
        paper_bgcolor='white',
        margin=dict(t=80, b=40, l=40, r=40)
    )

    return fig


# Main app
def main():
    st.title(f"🔍 {CONFIG['app']['title']}")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Check for embeddings and indexes
        embeddings, bm25_indexes, char_ngram_indexes, metadata = load_embeddings_and_metadata(CONFIG)

        if embeddings is None or metadata is None:
            st.error("⚠️ Embeddings not found! Please run generate_embeddings.py first.")
            st.info("Run: `python generate_embeddings.py`")
            st.stop()

        # Store in session state
        st.session_state.embeddings = embeddings
        st.session_state.bm25_indexes = bm25_indexes
        st.session_state.char_ngram_indexes = char_ngram_indexes
        st.session_state.metadata = metadata

        # Load models
        if st.session_state.model is None:
            with st.spinner("Loading embedding model..."):
                st.session_state.model = load_embedding_model(CONFIG['embeddings']['model'])

        if st.session_state.reranker is None:
            with st.spinner("Loading reranker model..."):
                st.session_state.reranker = load_reranker_model()

        st.success(f"✅ Loaded {len(metadata[CONFIG['metadata']['date_column']])} records")

    # Main content area
    st.markdown("##### 🔍 Search")

    # Search input
    search_placeholder = CONFIG['app'].get('search_placeholder', 'Enter your search query...')
    search_query = st.text_input(
        "Enter search phrase:",
        placeholder=search_placeholder,
        help="Enter keywords or phrases to search for"
    )

    # Search target selector (dynamic based on config)
    search_target_options = [col['name'] for col in CONFIG['text_columns']]
    search_target_display = {col['name']: col['display_name'] for col in CONFIG['text_columns']}

    default_target = CONFIG['app'].get('default_search_target', search_target_options[0])

    selected_target = st.radio(
        "Search in:",
        options=search_target_options,
        format_func=lambda x: search_target_display[x],
        index=search_target_options.index(default_target) if default_target in search_target_options else 0,
        horizontal=True,
        help="Choose which text field to search in"
    )

    st.markdown("##### Search Parameters")

    # Hybrid search mode selector
    use_hybrid = st.checkbox(
        "🚀 Enable Hybrid Search (Semantic + BM25 + Char N-gram)",
        value=True,
        help="Combine semantic, keyword (BM25), and typo-tolerant (char n-gram) search for best results"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        top_k_retrieval = st.slider(
            "🔍 Top-K for Hybrid Retrieval" if use_hybrid else "🔍 Top-K for Cosine Similarity",
            min_value=100,
            max_value=20000,
            value=CONFIG['app'].get('default_top_k_retrieval', 1000),
            step=100,
            help="Number of candidates to retrieve. Higher = more comprehensive but slower."
        )
    with col2:
        top_k_rerank = st.slider(
            "🎯 Top-K for Reranking",
            min_value=50,
            max_value=5000,
            value=CONFIG['app'].get('default_top_k_rerank', 500),
            step=50,
            help="Number of top candidates to rerank with BGE. Must be ≤ retrieval top-k. Higher = better quality but slower."
        )

    # Hybrid search weight controls
    if use_hybrid:
        with st.expander("⚙️ Advanced: Hybrid Search Weights", expanded=False):
            st.markdown("Adjust the contribution of each search method:")
            col_w1, col_w2, col_w3 = st.columns(3)
            with col_w1:
                semantic_weight = st.slider(
                    "Semantic",
                    min_value=0.0,
                    max_value=1.0,
                    value=CONFIG['app'].get('semantic_weight', 0.5),
                    step=0.1,
                    help="Weight for semantic (embedding) similarity"
                )
            with col_w2:
                bm25_weight = st.slider(
                    "BM25",
                    min_value=0.0,
                    max_value=1.0,
                    value=CONFIG['app'].get('bm25_weight', 0.3),
                    step=0.1,
                    help="Weight for keyword matching (BM25)"
                )
            with col_w3:
                char_ngram_weight = st.slider(
                    "Char N-gram",
                    min_value=0.0,
                    max_value=1.0,
                    value=CONFIG['app'].get('char_ngram_weight', 0.2),
                    step=0.1,
                    help="Weight for typo-tolerant matching"
                )

    # MMR diversity controls
    use_mmr = st.checkbox(
        "🎨 Enable MMR Diversity",
        value=CONFIG['app'].get('use_mmr', False),
        help="Use Maximal Marginal Relevance to diversify results and reduce redundancy"
    )

    if use_mmr:
        col_mmr1, col_mmr2 = st.columns([1, 1])
        with col_mmr1:
            mmr_lambda = st.slider(
                "MMR Lambda (Relevance vs Diversity)",
                min_value=0.0,
                max_value=1.0,
                value=CONFIG['app'].get('mmr_lambda', 0.5),
                step=0.1,
                help="Higher = more relevance, Lower = more diversity"
            )
        with col_mmr2:
            mmr_top_k = st.slider(
                "MMR Top-K Results",
                min_value=10,
                max_value=100,
                value=CONFIG['app'].get('mmr_top_k', 20),
                step=5,
                help="Number of diverse results to show"
            )

    # Ensure rerank top_k doesn't exceed retrieval top_k
    if top_k_rerank > top_k_retrieval:
        st.warning(f"⚠️ Rerank Top-K ({top_k_rerank}) cannot exceed Retrieval Top-K ({top_k_retrieval}). Adjusting to {top_k_retrieval}.")
        top_k_rerank = top_k_retrieval

    st.markdown("##### Filters")
    col3, col4 = st.columns([1, 1])

    with col3:
        # Date range filter
        date_col = CONFIG['metadata']['date_column']
        all_dates = pd.to_datetime(st.session_state.metadata[date_col])
        min_date = all_dates.min().date()
        max_date = all_dates.max().date()

        date_range = st.date_input(
            "📅 Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Filter results by date range"
        )

    with col4:
        # Score threshold
        score_threshold = st.slider(
            "Minimum Relevance Score",
            min_value=0.20,
            max_value=1.00,
            value=CONFIG['app'].get('default_score_threshold', 0.60),
            step=0.05,
            help="Only show results with relevance score greater than this threshold"
        )

    st.markdown("---")

    # Search button
    if st.button("🔍 Search", type="primary", use_container_width=True):
        if not search_query:
            st.warning("Please enter a search query")
            st.stop()

        # Step 1: Generate query embedding
        with st.spinner("Generating query embedding..."):
            query_embedding = st.session_state.model.encode(
                [search_query],
                normalize_embeddings=True
            )[0]

        # Step 2: Hybrid or Semantic search
        selected_embeddings = st.session_state.embeddings[selected_target]

        if use_hybrid:
            st.info(f"🚀 Hybrid search in {search_target_display[selected_target]} (Semantic + BM25 + Char N-gram)...")

            # Get BM25 and char n-gram indexes
            bm25_index = st.session_state.bm25_indexes.get(selected_target, None)
            char_ngram_index = st.session_state.char_ngram_indexes.get(selected_target, None)
            metadata_texts = st.session_state.metadata[selected_target].tolist() if char_ngram_index else None

            with st.spinner(f"Finding top {top_k_retrieval} candidates with hybrid search..."):
                top_indices, hybrid_scores = hybrid_search(
                    query=search_query,
                    query_embedding=query_embedding,
                    embeddings=selected_embeddings,
                    bm25_index=bm25_index,
                    char_ngram_vectorizer=char_ngram_index,
                    metadata_texts=metadata_texts,
                    top_k=top_k_retrieval,
                    semantic_weight=semantic_weight if 'semantic_weight' in locals() else 0.5,
                    bm25_weight=bm25_weight if 'bm25_weight' in locals() else 0.3,
                    char_ngram_weight=char_ngram_weight if 'char_ngram_weight' in locals() else 0.2
                )
            cos_scores = hybrid_scores  # Use hybrid scores as cos_scores for compatibility
        else:
            st.info(f"🔍 Semantic search in {search_target_display[selected_target]}...")

            with st.spinner(f"Finding top {top_k_retrieval} similar items..."):
                top_indices, cos_scores = semantic_search(
                    query_embedding,
                    selected_embeddings,
                    top_k=top_k_retrieval
                )

        # Step 3: Select top candidates for reranking
        rerank_indices = top_indices[:top_k_rerank]

        st.info(f"🎯 Reranking top {len(rerank_indices)} candidates with BGE cross-encoder...")

        # Step 4: Rerank with BGE
        with st.spinner(f"Reranking {len(rerank_indices)} results with BGE cross-encoder..."):
            candidate_texts = [
                st.session_state.metadata[selected_target][idx]
                for idx in rerank_indices
            ]

            rerank_scores = rerank_with_bge(
                search_query,
                candidate_texts,
                st.session_state.reranker
            )

            rerank_scores = np.array(rerank_scores)

        # Step 5: Apply MMR for diversity (if enabled)
        if use_mmr:
            st.info(f"🎨 Applying MMR diversity to select top {mmr_top_k if 'mmr_top_k' in locals() else 20} diverse results...")

            with st.spinner("Diversifying results with MMR..."):
                mmr_indices, mmr_scores = maximal_marginal_relevance(
                    query_embedding=query_embedding,
                    doc_embeddings=selected_embeddings,
                    doc_indices=rerank_indices,
                    relevance_scores=rerank_scores,
                    top_k=mmr_top_k if 'mmr_top_k' in locals() else 20,
                    lambda_param=mmr_lambda if 'mmr_lambda' in locals() else 0.5
                )

            # Use MMR results
            filtered_indices = mmr_indices
            filtered_scores = mmr_scores

            # Get corresponding initial scores
            filtered_cos_scores_dict = dict(zip(top_indices, cos_scores))
            filtered_cos_scores = np.array([filtered_cos_scores_dict.get(idx, 0) for idx in filtered_indices])

        else:
            # Step 5 (original): Filter by threshold
            mask = rerank_scores >= score_threshold

            filtered_indices = rerank_indices[mask]
            filtered_scores = rerank_scores[mask]

            # Get corresponding cosine similarity scores for filtered results
            filtered_cos_scores = cos_scores[:top_k_rerank][mask]

        if len(filtered_indices) == 0:
            st.warning(f"No results found with relevance score > {score_threshold:.2f}. Try lowering the score threshold or a different query.")
            st.stop()

        st.success(f"✅ Found {len(filtered_indices)} highly relevant results" + (f" (score > {score_threshold:.2f})" if not use_mmr else ""))

        # Step 6: Prepare results dataframe
        date_col = CONFIG['metadata']['date_column']
        score_col = CONFIG['metadata']['score_column']

        results_data = {
            'index': filtered_indices,
            'cosine_similarity': filtered_cos_scores,
            'relevance_score': filtered_scores,
            date_col: st.session_state.metadata[date_col][filtered_indices],
            score_col: st.session_state.metadata[score_col][filtered_indices]
        }

        # Add all text columns
        for text_col in CONFIG['text_columns']:
            internal_name = text_col['name']
            display_name = text_col['display_name']
            results_data[display_name] = st.session_state.metadata[internal_name][filtered_indices]

        results_df = pd.DataFrame(results_data)

        # Step 7: Date range filtering
        if len(date_range) == 2:
            start_date, end_date = date_range
            results_df[date_col] = pd.to_datetime(results_df[date_col])
            results_df = results_df[
                (results_df[date_col].dt.date >= start_date) &
                (results_df[date_col].dt.date <= end_date)
            ]

        if len(results_df) == 0:
            st.warning("No results found in the selected date range")
            st.session_state.search_results = None
        else:
            # Store results in session state
            st.session_state.search_results = {
                'results_df': results_df,
                'selected_target': selected_target,
                'search_target_display': search_target_display
            }

    # Display results (outside the button block so they persist)
    if st.session_state.search_results is not None:
        results_df = st.session_state.search_results['results_df']
        selected_target = st.session_state.search_results['selected_target']
        search_target_display = st.session_state.search_results['search_target_display']

        # Step 8: Preview results
        st.markdown("---")

        # Column selector for preview
        available_columns = []
        for text_col in CONFIG['text_columns']:
            available_columns.append(search_target_display[text_col['name']])

        # Add similarity and relevance score columns
        available_columns.append('cosine_similarity')
        available_columns.append('relevance_score')

        # Add date and score columns from config
        date_col = CONFIG['metadata']['date_column']
        score_col = CONFIG['metadata']['score_column']
        available_columns.extend([date_col, score_col])

        selected_columns = st.multiselect(
            "Select columns to display:",
            options=available_columns,
            default=[search_target_display[selected_target], date_col, score_col][:2],
            help="Choose which columns to show in the preview table"
        )

        st.subheader(f"📋 Preview Results (Top {min(20, len(results_df))} of {len(results_df)})")

        if selected_columns:
            preview_df = results_df[selected_columns].head(20).copy()

            # Configure columns using actual column names from config
            date_col = CONFIG['metadata']['date_column']
            score_col = CONFIG['metadata']['score_column']

            column_config = {}
            if date_col in selected_columns:
                column_config[date_col] = st.column_config.DateColumn(date_col, format="YYYY-MM-DD", width="small")
            if score_col in selected_columns:
                column_config[score_col] = st.column_config.NumberColumn(score_col, format="%.2f", width="small")
            if 'cosine_similarity' in selected_columns:
                column_config['cosine_similarity'] = st.column_config.NumberColumn("Cosine Similarity", format="%.4f", width="small")
            if 'relevance_score' in selected_columns:
                column_config['relevance_score'] = st.column_config.NumberColumn("Relevance Score", format="%.4f", width="small")

            st.dataframe(
                preview_df,
                use_container_width=True,
                hide_index=True,
                column_config=column_config
            )
        else:
            st.info("Please select at least one column to display")

        # Step 9: Visualization options
        st.markdown("---")
        st.subheader("📊 Visualizations")

        viz_col1, viz_col2 = st.columns([1, 1])

        with viz_col1:
            period_type = st.selectbox(
                "Time Granularity",
                options=['month', 'year', 'day'],
                help="Choose the time period for trend aggregation"
            )

        with viz_col2:
            chart_type = st.selectbox(
                "Chart Type",
                options=['trend', 'distribution', 'both'],
                help="Choose visualization type"
            )

        # Generate visualizations
        period_counts = None
        results_with_period = None

        if chart_type in ['trend', 'both']:
            date_col = CONFIG['metadata']['date_column']
            period_counts, results_with_period = aggregate_by_period(results_df.copy(), date_col, period=period_type)

            st.plotly_chart(
                create_trend_chart(period_counts, period_type),
                use_container_width=True
            )

        if chart_type in ['distribution', 'both']:
            score_col = CONFIG['metadata']['score_column']
            score_display_name = CONFIG['metadata'].get('score_display_name', score_col)
            st.plotly_chart(
                create_score_distribution_chart(results_df[score_col].values, score_display_name),
                use_container_width=True
            )

        # Step 10: Detailed results by period (only show if trend chart was generated)
        if chart_type in ['trend', 'both'] and period_counts is not None:
            st.markdown("---")
            st.subheader(f"📝 Top Results by {period_type.capitalize()}")

            date_col = CONFIG['metadata']['date_column']
            score_col = CONFIG['metadata']['score_column']

            for period in period_counts['period'].values[::-1]:
                period_data = results_with_period[results_with_period['period'] == period]
                period_sorted = period_data.nlargest(10, 'relevance_score')

                with st.expander(f"📅 {period} ({len(period_data)} results)"):
                    for idx, row in period_sorted.iterrows():
                        col_a, col_b = st.columns([4, 1])

                        with col_a:
                            # Display the selected search target text
                            text_content = row[search_target_display[selected_target]]
                            st.markdown(f"**{text_content}**")
                            st.caption(f"{date_col}: {row[date_col]}")

                        with col_b:
                            score_color = 'green' if row['relevance_score'] >= 0.8 else 'orange' if row['relevance_score'] >= 0.6 else 'red'
                            st.markdown(f"Score: :{score_color}[**{row['relevance_score']:.3f}**]")
                            st.caption(f"{score_col}: {row[score_col]}")

                        st.markdown("---")


if __name__ == "__main__":
    main()
