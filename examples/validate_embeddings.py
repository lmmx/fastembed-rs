#!/usr/bin/env python3
"""
Validation script to compare Python ColBERT embeddings with Rust.

Run this after running the Rust validate_embeddings example and compare outputs.

Usage: python validate_embeddings.py
"""

import numpy as np
from fastembed import LateInteractionTextEmbedding
from fastembed.postprocess import Muvera


def maxsim(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """Compute ColBERT MaxSim score."""
    # query_emb: (q_tokens, dim), doc_emb: (d_tokens, dim)
    # For each query token, find max similarity with any doc token
    similarities = query_emb @ doc_emb.T  # (q_tokens, d_tokens)
    return float(similarities.max(axis=1).sum())


def main():
    # Fixed test inputs - MUST match Rust script exactly
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
    ]
    queries = [
        "What animal jumps?",
        "What is ML?",
    ]

    print("=== Python ColBERT + MUVERA Validation ===\n")

    # Initialize model
    model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
    print(f"Model dimension: {model.embedding_size}")

    # Embed documents
    print("\n--- Document Embeddings ---")
    doc_embeddings = list(model.embed(documents))
    for i, emb in enumerate(doc_embeddings):
        print(f"Doc {i} shape: {emb.shape}")
        if emb.size > 0:
            print(f"  First token, first 5 values: {emb[0, :5].tolist()}")
            norm = np.linalg.norm(emb[0])
            print(f"  First token L2 norm: {norm:.6f}")
            print(f"  First token sum: {emb[0].sum():.6f}")

    # Embed queries
    print("\n--- Query Embeddings ---")
    query_embeddings = list(model.query_embed(queries))
    for i, emb in enumerate(query_embeddings):
        print(f"Query {i} shape: {emb.shape}")
        if emb.size > 0:
            print(f"  First token, first 5 values: {emb[0, :5].tolist()}")
            norm = np.linalg.norm(emb[0])
            print(f"  First token L2 norm: {norm:.6f}")
            print(f"  First token sum: {emb[0].sum():.6f}")

    # Test MaxSim scores (brute force ColBERT)
    print("\n--- MaxSim Scores (Query x Doc) ---")
    for qi, q_emb in enumerate(query_embeddings):
        for di, d_emb in enumerate(doc_embeddings):
            score = maxsim(q_emb, d_emb)
            print(f"Query {qi} x Doc {di}: {score:.4f}")

    # Test MUVERA
    print("\n--- MUVERA FDEs ---")
    muvera = Muvera(
        dim=model.embedding_size,
        k_sim=5,
        dim_proj=16,
        r_reps=20,
        random_seed=42,
    )
    print(f"MUVERA embedding size: {muvera.embedding_size}")

    doc_fdes = [muvera.process_document(emb) for emb in doc_embeddings]
    query_fdes = [muvera.process_query(emb) for emb in query_embeddings]

    for i, fde in enumerate(doc_fdes):
        print(f"Doc {i} FDE first 5: {fde[:5].tolist()}")
        print(f"  FDE L2 norm: {np.linalg.norm(fde):.6f}")

    for i, fde in enumerate(query_fdes):
        print(f"Query {i} FDE first 5: {fde[:5].tolist()}")
        print(f"  FDE L2 norm: {np.linalg.norm(fde):.6f}")

    # MUVERA dot product scores
    print("\n--- MUVERA Dot Product Scores (Query x Doc) ---")
    for qi, q_fde in enumerate(query_fdes):
        for di, d_fde in enumerate(doc_fdes):
            score = float(np.dot(q_fde, d_fde))
            print(f"Query {qi} x Doc {di}: {score:.4f}")


if __name__ == "__main__":
    main()
