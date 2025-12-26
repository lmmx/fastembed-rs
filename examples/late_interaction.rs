//! Example demonstrating late interaction models (ColBERT) and MUVERA post-processing
//!
//! Late interaction models generate token-level embeddings for more fine-grained matching.
//! MUVERA transforms these multi-vector embeddings into fixed-dimensional representations
//! for efficient vector search.

#[cfg(feature = "hf-hub")]
fn main() -> anyhow::Result<()> {
    use fastembed::{
        LateInteractionInitOptions, LateInteractionModel, LateInteractionTextEmbedding, Muvera,
    };

    println!("=== Late Interaction Models Example ===\n");

    // Initialize a ColBERT model
    println!("Loading ColBERT v2.0 model...");
    let model = LateInteractionTextEmbedding::try_new(
        LateInteractionInitOptions::new(LateInteractionModel::ColBERTv2)
            .with_show_download_progress(true),
    )?;

    println!("Model loaded successfully!\n");

    // Sample documents
    let documents = vec![
        "ColBERT is a late interaction model for efficient retrieval.",
        "BERT uses pooled embeddings for semantic search.",
        "Late interaction allows token-level matching for better accuracy.",
    ];

    // Generate multi-vector embeddings for documents
    println!("Generating document embeddings...");
    let doc_embeddings = model.passage_embed(documents.clone(), None)?;

    println!("Generated {} document embeddings", doc_embeddings.len());
    for (i, emb) in doc_embeddings.iter().enumerate() {
        println!(
            "  Document {}: {} tokens × {} dims",
            i + 1,
            emb.len(),
            emb[0].len()
        );
    }

    // Generate query embeddings
    println!("\nGenerating query embeddings...");
    let queries = vec!["What is late interaction?"];
    let query_embeddings = model.query_embed(queries, None)?;

    for (i, emb) in query_embeddings.iter().enumerate() {
        println!(
            "  Query {}: {} tokens × {} dims",
            i + 1,
            emb.len(),
            emb[0].len()
        );
    }

    // MUVERA Post-Processing
    println!("\n=== MUVERA Post-Processing ===\n");

    // Create MUVERA processor with recommended parameters
    // k_sim=6 (64 buckets), dim_proj=32, r_reps=20
    // Output dimension: 20 * 64 * 32 = 40,960
    let muvera = Muvera::from_model(&model, 6, 32, 20);

    println!(
        "Created MUVERA processor (output dimension: {})",
        muvera.get_output_dimension()
    );

    // Transform multi-vector embeddings to fixed-dimensional encodings
    println!("\nProcessing documents with MUVERA...");
    let fde_embeddings: Vec<_> = doc_embeddings
        .iter()
        .map(|emb| muvera.process_document(emb))
        .collect::<Result<_, _>>()?;

    for (i, fde) in fde_embeddings.iter().enumerate() {
        println!("  Document {} FDE: {} dimensions", i + 1, fde.len());
    }

    // Process query
    let query_fde = muvera.process_query(&query_embeddings[0])?;
    println!("  Query FDE: {} dimensions", query_fde.len());

    println!("\n=== Workflow Summary ===");
    println!("1. Generate multi-vector embeddings (token-level)");
    println!("2. Apply MUVERA to get fixed-dimensional encodings (FDE)");
    println!("3. Use FDEs for fast initial retrieval");
    println!("4. Rerank top-k results using original multi-vector embeddings");
    println!("\nBenefits:");
    println!("  - ~8x faster than full multi-vector search");
    println!("  - Maintains high accuracy with hybrid approach");
    println!("  - Compatible with traditional vector databases");

    Ok(())
}

#[cfg(not(feature = "hf-hub"))]
fn main() {
    println!("This example requires the 'hf-hub' feature to be enabled.");
    println!("Run with: cargo run --example late_interaction --features hf-hub");
}
