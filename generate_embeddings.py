"""
Generate embeddings for any dataset based on config.yaml.
This script should be run once to create the embeddings files.
Includes: Semantic embeddings, BM25 index, and char n-gram TF-IDF index.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle
import yaml
from typing import Dict, List, Any
import sys
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import re


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """Load dataset from CSV file(s)."""
    dataset_config = config['dataset']

    # Check if single file or multiple files
    if 'file_path' in dataset_config:
        file_path = dataset_config['file_path']
        print(f"Loading dataset from: {file_path}")
        df = pd.read_csv(file_path)
    elif 'file_paths' in dataset_config:
        file_paths = dataset_config['file_paths']
        print(f"Loading dataset from {len(file_paths)} files...")
        dfs = [pd.read_csv(fp) for fp in file_paths]
        df = pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError("Config must specify either 'file_path' or 'file_paths'")

    print(f"Total records: {len(df)}")
    return df


def validate_config(config: Dict[str, Any], df: pd.DataFrame) -> None:
    """Validate that config columns exist in dataframe."""
    errors = []

    # Check text columns
    for text_col in config['text_columns']:
        col_name = text_col['column']
        if col_name not in df.columns:
            errors.append(f"Text column '{col_name}' not found in dataset")

    # Check metadata columns
    metadata = config['metadata']
    if metadata['date_column'] not in df.columns:
        errors.append(f"Date column '{metadata['date_column']}' not found in dataset")

    if metadata['score_column'] not in df.columns:
        errors.append(f"Score column '{metadata['score_column']}' not found in dataset")

    if 'id_column' in metadata and metadata['id_column'] not in df.columns:
        errors.append(f"ID column '{metadata['id_column']}' not found in dataset")

    # Check display columns
    if 'display_columns' in metadata:
        for col in metadata['display_columns']:
            if col not in df.columns:
                errors.append(f"Display column '{col}' not found in dataset")

    if errors:
        print("\n[ERROR] Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        print(f"\nAvailable columns in dataset: {list(df.columns)}")
        sys.exit(1)

    print("[OK] Configuration validated successfully")


def tokenize_for_bm25(text: str) -> List[str]:
    """Simple tokenizer for BM25."""
    # Convert to lowercase and split on non-alphanumeric characters
    tokens = re.findall(r'\w+', text.lower())
    return tokens


def generate_embeddings_for_column(
    df: pd.DataFrame,
    column_name: str,
    model: SentenceTransformer,
    batch_size: int = 1000,
    encoding_batch_size: int = 32
) -> np.ndarray:
    """Generate embeddings for a specific text column."""
    print(f"\nGenerating embeddings for column '{column_name}'...")

    # Fill NaN values with empty string
    texts = df[column_name].fillna('').tolist()

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(
            batch_texts,
            show_progress_bar=True,
            batch_size=encoding_batch_size,
            normalize_embeddings=True  # Normalized for cosine similarity
        )
        all_embeddings.append(batch_embeddings)
        print(f"Processed {min(i+batch_size, len(texts))}/{len(texts)} items")

    embeddings = np.vstack(all_embeddings)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings size: {embeddings.nbytes / (1024*1024):.2f} MB")

    return embeddings


def generate_bm25_index(
    df: pd.DataFrame,
    column_name: str
) -> BM25Okapi:
    """Generate BM25 index for a specific text column."""
    print(f"\nGenerating BM25 index for column '{column_name}'...")

    # Fill NaN values with empty string
    texts = df[column_name].fillna('').tolist()

    # Tokenize all documents
    tokenized_corpus = [tokenize_for_bm25(text) for text in texts]

    # Create BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    print(f"BM25 index created for {len(tokenized_corpus)} documents")
    return bm25


def generate_char_ngram_index(
    df: pd.DataFrame,
    column_name: str,
    ngram_range: tuple = (2, 4)
) -> TfidfVectorizer:
    """Generate character n-gram TF-IDF index for typo tolerance."""
    print(f"\nGenerating char n-gram TF-IDF index for column '{column_name}'...")

    # Fill NaN values with empty string
    texts = df[column_name].fillna('').tolist()

    # Create character n-gram vectorizer
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=ngram_range,
        lowercase=True,
        max_features=50000  # Limit features to save memory
    )

    # Fit the vectorizer
    vectorizer.fit(texts)

    print(f"Char n-gram index created with {len(vectorizer.get_feature_names_out())} features")
    print(f"N-gram range: {ngram_range}")
    return vectorizer


def main():
    # Load configuration
    print("Loading configuration from config.yaml...")
    config = load_config()

    # Load dataset
    df = load_dataset(config)

    # Validate configuration
    validate_config(config, df)

    # Load embedding model
    model_name = config['embeddings']['model']
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Create output directory
    output_dir = Path(config['embeddings']['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate embeddings for each text column
    embedding_files = {}
    bm25_indexes = {}
    char_ngram_indexes = {}
    batch_size = config['embeddings'].get('batch_size', 32)

    for text_col in config['text_columns']:
        col_name = text_col['column']
        internal_name = text_col['name']

        # 1. Generate semantic embeddings
        embeddings = generate_embeddings_for_column(
            df=df,
            column_name=col_name,
            model=model,
            batch_size=1000,
            encoding_batch_size=batch_size
        )

        # Save embeddings
        output_file = output_dir / f'{internal_name}_embeddings.npz'
        np.savez_compressed(
            output_file,
            embeddings=embeddings.astype(np.float32)
        )

        file_size = output_file.stat().st_size / (1024*1024)
        embedding_files[internal_name] = {
            'file': output_file,
            'size': file_size
        }
        print(f"Saved: {output_file} ({file_size:.2f} MB)")

        # 2. Generate BM25 index
        bm25 = generate_bm25_index(df, col_name)
        bm25_indexes[internal_name] = bm25

        # Save BM25 index
        bm25_file = output_dir / f'{internal_name}_bm25.pkl'
        with open(bm25_file, 'wb') as f:
            pickle.dump(bm25, f, protocol=4)
        bm25_size = bm25_file.stat().st_size / (1024*1024)
        print(f"Saved: {bm25_file} ({bm25_size:.2f} MB)")

        # 3. Generate char n-gram TF-IDF index
        char_ngram = generate_char_ngram_index(df, col_name)
        char_ngram_indexes[internal_name] = char_ngram

        # Save char n-gram index
        char_ngram_file = output_dir / f'{internal_name}_char_ngram.pkl'
        with open(char_ngram_file, 'wb') as f:
            pickle.dump(char_ngram, f, protocol=4)
        char_ngram_size = char_ngram_file.stat().st_size / (1024*1024)
        print(f"Saved: {char_ngram_file} ({char_ngram_size:.2f} MB)")

    # Prepare metadata
    print("\nPreparing metadata...")
    metadata = {}

    # Add text columns
    for text_col in config['text_columns']:
        col_name = text_col['column']
        internal_name = text_col['name']
        metadata[internal_name] = df[col_name].fillna('').values

    # Add date column (use the actual column name as the key)
    date_col = config['metadata']['date_column']
    metadata[date_col] = df[date_col].values

    # Add score column (use the actual column name as the key)
    score_col = config['metadata']['score_column']
    metadata[score_col] = df[score_col].values

    # Add ID column if specified (use the actual column name as the key)
    if 'id_column' in config['metadata']:
        id_col = config['metadata']['id_column']
        metadata[id_col] = df[id_col].values

    # Add display columns if specified (use the actual column names as keys)
    if 'display_columns' in config['metadata']:
        for col in config['metadata']['display_columns']:
            metadata[col] = df[col].values

    # Save metadata
    metadata_file = output_dir / 'metadata.pkl'
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f, protocol=4)

    metadata_size = metadata_file.stat().st_size / (1024*1024)
    print(f"Metadata saved: {metadata_file} ({metadata_size:.2f} MB)")

    # Summary
    print(f"\n{'='*70}")
    print("[OK] All indexes generated successfully!")
    print(f"{'='*70}")
    print(f"\nGenerated files:")
    print(f"\n1. Semantic Embeddings:")
    for name, info in embedding_files.items():
        print(f"   - {name}: {info['file']} ({info['size']:.2f} MB)")

    print(f"\n2. BM25 Indexes:")
    for name in bm25_indexes.keys():
        bm25_file = output_dir / f'{name}_bm25.pkl'
        bm25_size = bm25_file.stat().st_size / (1024*1024)
        print(f"   - {name}: {bm25_file} ({bm25_size:.2f} MB)")

    print(f"\n3. Char N-gram TF-IDF Indexes:")
    for name in char_ngram_indexes.keys():
        char_ngram_file = output_dir / f'{name}_char_ngram.pkl'
        char_ngram_size = char_ngram_file.stat().st_size / (1024*1024)
        print(f"   - {name}: {char_ngram_file} ({char_ngram_size:.2f} MB)")

    print(f"\n4. Metadata: {metadata_file} ({metadata_size:.2f} MB)")

    total_size = sum(info['size'] for info in embedding_files.values()) + metadata_size
    print(f"\nTotal size: {total_size:.2f} MB")
    print(f"\nNext step: Run 'streamlit run app.py' to start the hybrid search app!")


if __name__ == "__main__":
    main()
