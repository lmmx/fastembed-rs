//! Validation script to compare Rust ColBERT embeddings with Python
//!
//! Run this, then run validate_embeddings.py and compare outputs.
//!
//! Usage: cargo run --example validate_embeddings --features muvera

use anyhow::Result;
use fastembed::{LateInteractionInitOptions, LateInteractionModel, LateInteractionTextEmbedding, Muvera};

fn main() -> Result<()> {
    // Fixed test inputs - MUST match Python script exactly
    let documents = vec![
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
    ];
    let queries = vec![
        "What animal jumps?",
        "What is ML?",
    ];

    println!("=== Rust ColBERT + MUVERA Validation ===\n");

    // Initialize model
    let mut model = LateInteractionTextEmbedding::try_new(
        LateInteractionInitOptions::new(LateInteractionModel::ColBERTV2)
    )?;

    println!("Model dimension: {}", model.dim());

    // Embed documents
    println!("\n--- Document Embeddings ---");
    let doc_embeddings = model.embed(&documents, None)?;
    for (i, emb) in doc_embeddings.iter().enumerate() {
        println!("Doc {} shape: ({}, {})", i, emb.len(), if emb.is_empty() { 0 } else { emb[0].len() });
        if !emb.is_empty() && !emb[0].is_empty() {
            // Print first token's first 5 values
            println!("  First token, first 5 values: {:?}", &emb[0][..5.min(emb[0].len())]);
            // Print L2 norm of first token (should be ~1.0)
            let norm: f32 = emb[0].iter().map(|x| x * x).sum::<f32>().sqrt();
            println!("  First token L2 norm: {:.6}", norm);
            // Print sum of first token
            let sum: f32 = emb[0].iter().sum();
            println!("  First token sum: {:.6}", sum);
        }
    }

    // Embed queries
    println!("\n--- Query Embeddings ---");
    let query_embeddings = model.query_embed(&queries, None)?;
    for (i, emb) in query_embeddings.iter().enumerate() {
        println!("Query {} shape: ({}, {})", i, emb.len(), if emb.is_empty() { 0 } else { emb[0].len() });
        if !emb.is_empty() && !emb[0].is_empty() {
            println!("  First token, first 5 values: {:?}", &emb[0][..5.min(emb[0].len())]);
            let norm: f32 = emb[0].iter().map(|x| x * x).sum::<f32>().sqrt();
            println!("  First token L2 norm: {:.6}", norm);
            let sum: f32 = emb[0].iter().sum();
            println!("  First token sum: {:.6}", sum);
        }
    }

    // Test MaxSim scores (brute force ColBERT)
    println!("\n--- MaxSim Scores (Query x Doc) ---");
    for (qi, q_emb) in query_embeddings.iter().enumerate() {
        for (di, d_emb) in doc_embeddings.iter().enumerate() {
            let score = maxsim(q_emb, d_emb);
            println!("Query {} x Doc {}: {:.4}", qi, di, score);
        }
    }

    // Test MUVERA
    println!("\n--- MUVERA FDEs ---");
    let muvera = Muvera::from_late_interaction_model(
        &model,
        Some(5),   // k_sim
        Some(16),  // dim_proj
        Some(20),  // r_reps
        Some(42),  // seed
    )?;
    println!("MUVERA embedding size: {}", muvera.embedding_size());

    let doc_fdes: Vec<Vec<f32>> = doc_embeddings.iter()
        .map(|emb| muvera.process_document(emb))
        .collect();

    let query_fdes: Vec<Vec<f32>> = query_embeddings.iter()
        .map(|emb| muvera.process_query(emb))
        .collect();

    for (i, fde) in doc_fdes.iter().enumerate() {
        println!("Doc {} FDE first 5: {:?}", i, &fde[..5]);
        println!("  FDE L2 norm: {:.6}", fde.iter().map(|x| x * x).sum::<f32>().sqrt());
    }

    for (i, fde) in query_fdes.iter().enumerate() {
        println!("Query {} FDE first 5: {:?}", i, &fde[..5]);
        println!("  FDE L2 norm: {:.6}", fde.iter().map(|x| x * x).sum::<f32>().sqrt());
    }

    // MUVERA dot product scores
    println!("\n--- MUVERA Dot Product Scores (Query x Doc) ---");
    for (qi, q_fde) in query_fdes.iter().enumerate() {
        for (di, d_fde) in doc_fdes.iter().enumerate() {
            let score: f32 = q_fde.iter().zip(d_fde.iter()).map(|(a, b)| a * b).sum();
            println!("Query {} x Doc {}: {:.4}", qi, di, score);
        }
    }

    Ok(())
}

/// Compute ColBERT MaxSim score
fn maxsim(query_emb: &[Vec<f32>], doc_emb: &[Vec<f32>]) -> f32 {
    query_emb
        .iter()
        .map(|q| {
            doc_emb
                .iter()
                .map(|d| q.iter().zip(d.iter()).map(|(a, b)| a * b).sum::<f32>())
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}
