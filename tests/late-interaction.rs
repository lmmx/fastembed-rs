#![cfg(feature = "hf-hub")]

use fastembed::{
    LateInteractionInitOptions, LateInteractionModel, LateInteractionTextEmbedding,
};
#[cfg(feature = "muvera")]
use fastembed::Muvera;

// Canonical values for "Hello World" with colbert-ir/colbertv2.0
// First 5 columns of first 5 tokens
const CANONICAL_DOC_VALUES_COLBERT: [[f32; 5]; 5] = [
    [0.0759, 0.0841, -0.0299, 0.0374, 0.0254],
    [0.0005, -0.0163, -0.0127, 0.2165, 0.1517],
    [-0.0257, -0.0575, 0.0135, 0.2202, 0.1896],
    [0.0846, 0.0122, 0.0032, -0.0109, -0.1041],
    [0.0477, 0.1078, -0.0314, 0.016, 0.0156],
];

const CANONICAL_QUERY_VALUES_COLBERT: [[f32; 5]; 5] = [
    [0.0824, 0.0872, -0.0324, 0.0418, 0.024],
    [-0.0007, -0.0154, -0.0113, 0.2277, 0.1528],
    [-0.0251, -0.0565, 0.0136, 0.2236, 0.1838],
    [0.0848, 0.0056, 0.0041, -0.0036, -0.1032],
    [0.0574, 0.1072, -0.0332, 0.0233, 0.0209],
];

const CANONICAL_DOC_VALUES_ANSWERAI: [[f32; 5]; 5] = [
    [-0.07281, 0.04632, -0.04711, 0.00762, -0.07374],
    [-0.04464, 0.04426, -0.074, 0.01801, -0.05233],
    [0.09936, -0.05123, -0.04925, -0.05276, -0.08944],
    [0.01644, 0.0203, -0.03789, 0.03165, -0.06501],
    [-0.07281, 0.04633, -0.04711, 0.00762, -0.07374],
];

fn assert_close(actual: &[f32], expected: &[f32], atol: f32) {
    assert_eq!(actual.len(), expected.len(), "Length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < atol,
            "Mismatch at index {}: actual={}, expected={}, diff={}",
            i, a, e, (a - e).abs()
        );
    }
}

#[test]
fn test_late_interaction_colbert_canonical_values() {
    let mut model = LateInteractionTextEmbedding::try_new(LateInteractionInitOptions::new(
        LateInteractionModel::ColBERTV2,
    ))
    .unwrap();

    let documents = vec!["Hello World"];
    let embeddings = model.embed(&documents, None).unwrap();

    assert_eq!(embeddings.len(), 1);
    let embedding = &embeddings[0];
    
    // Check we have at least 5 tokens
    assert!(embedding.len() >= 5, "Expected at least 5 tokens, got {}", embedding.len());
    
    // Check dimension
    assert_eq!(embedding[0].len(), 128);

    // Verify canonical values (first 5 columns of first 5 tokens)
    for (token_idx, expected_row) in CANONICAL_DOC_VALUES_COLBERT.iter().enumerate() {
        let actual: Vec<f32> = embedding[token_idx][..5].to_vec();
        assert_close(&actual, expected_row, 2e-3);
    }
}

#[test]
fn test_late_interaction_colbert_query_canonical_values() {
    let mut model = LateInteractionTextEmbedding::try_new(LateInteractionInitOptions::new(
        LateInteractionModel::ColBERTV2,
    ))
    .unwrap();

    let queries = vec!["Hello World"];
    let embeddings = model.query_embed(&queries, None).unwrap();

    assert_eq!(embeddings.len(), 1);
    let embedding = &embeddings[0];
    
    // Query should be padded to min_query_length (32)
    assert_eq!(embedding.len(), 32, "Query should be padded to 32 tokens");
    
    // Check dimension
    assert_eq!(embedding[0].len(), 128);

    // Verify canonical values (first 5 columns of first 5 tokens)
    for (token_idx, expected_row) in CANONICAL_QUERY_VALUES_COLBERT.iter().enumerate() {
        let actual: Vec<f32> = embedding[token_idx][..5].to_vec();
        assert_close(&actual, expected_row, 2e-3);
    }
}

#[test]
fn test_late_interaction_answerai_canonical_values() {
    let mut model = LateInteractionTextEmbedding::try_new(LateInteractionInitOptions::new(
        LateInteractionModel::AnswerAIColBERTSmallV1,
    ))
    .unwrap();

    let documents = vec!["Hello World"];
    let embeddings = model.embed(&documents, None).unwrap();

    assert_eq!(embeddings.len(), 1);
    let embedding = &embeddings[0];
    
    // Check dimension (96 for answerai)
    assert_eq!(embedding[0].len(), 96);

    // Verify canonical values
    for (token_idx, expected_row) in CANONICAL_DOC_VALUES_ANSWERAI.iter().enumerate() {
        let actual: Vec<f32> = embedding[token_idx][..5].to_vec();
        assert_close(&actual, expected_row, 2e-3);
    }
}

#[test]
fn test_batch_consistency() {
    let mut model = LateInteractionTextEmbedding::try_new(LateInteractionInitOptions::new(
        LateInteractionModel::AnswerAIColBERTSmallV1,
    ))
    .unwrap();

    let documents = vec![
        "short document",
        "A bit longer document, which should not affect the size",
    ];
    
    // Embed with batch_size=1
    let result_batch_1 = model.embed(&documents, Some(1)).unwrap();
    
    // Embed with batch_size=2
    let result_batch_2 = model.embed(&documents, Some(2)).unwrap();

    // First document should have same number of tokens regardless of batch size
    assert_eq!(
        result_batch_1[0].len(), 
        result_batch_2[0].len(),
        "Batch size should not affect token count"
    );
    
    // Values should be very close
    for (t1, t2) in result_batch_1[0].iter().zip(result_batch_2[0].iter()) {
        assert_close(t1, t2, 1e-3);
    }
}

#[test]
fn test_embedding_dimension() {
    let model = LateInteractionTextEmbedding::try_new(LateInteractionInitOptions::new(
        LateInteractionModel::ColBERTV2,
    ))
    .unwrap();
    
    assert_eq!(model.dim(), 128);
    
    let model = LateInteractionTextEmbedding::try_new(LateInteractionInitOptions::new(
        LateInteractionModel::AnswerAIColBERTSmallV1,
    ))
    .unwrap();
    
    assert_eq!(model.dim(), 96);
}

#[cfg(feature = "muvera")]
#[test]
fn test_muvera_with_colbert() {
    let mut model = LateInteractionTextEmbedding::try_new(LateInteractionInitOptions::new(
        LateInteractionModel::ColBERTV2,
    ))
    .unwrap();

    // Use from_late_interaction_model for convenience
    let muvera = Muvera::from_late_interaction_model(&model, Some(5), Some(16), Some(20), Some(42))
        .unwrap();

    let documents = vec!["This is a test document about neural networks."];
    let queries = vec!["What are neural networks?"];

    let doc_embeddings = model.embed(&documents, None).unwrap();
    let query_embeddings = model.query_embed(&queries, None).unwrap();

    // Convert to MUVERA fixed-dimensional encodings
    // NOTE: Fixed method names from encode_* to process_*
    let doc_fde = muvera.process_document(&doc_embeddings[0]);
    let query_fde = muvera.process_query(&query_embeddings[0]);

    // Expected size: r_reps * 2^k_sim * dim_proj = 20 * 32 * 16 = 10240
    assert_eq!(muvera.embedding_size(), 10240);
    assert_eq!(doc_fde.len(), muvera.embedding_size());
    assert_eq!(query_fde.len(), muvera.embedding_size());

    // FDEs should be usable for dot product similarity
    let similarity: f32 = doc_fde.iter().zip(query_fde.iter()).map(|(a, b)| a * b).sum();
    
    // Similarity should be positive for related query-document pair
    assert!(similarity > 0.0, "Expected positive similarity for related query-doc pair");
    println!("Similarity: {}", similarity);
}

#[cfg(feature = "muvera")]
#[test]
fn test_muvera_canonical_values() {
    let muvera = Muvera::new(128, Some(5), Some(16), Some(20), Some(42)).unwrap();

    let vectors: Vec<Vec<f32>> = (0..10)
        .map(|i| (0..128).map(|j| ((i * 128 + j) as f32).sin()).collect())
        .collect();

    let fde = muvera.process_document(&vectors);

    // Values from generate_muvera_canonical test
    const EXPECTED_FIRST_10: [f32; 10] = [
        2.0179653, 1.6323578, -1.5774617, -0.26919794, 3.2250175, -2.0104198, -2.2146697, -1.0453973, -1.2936, 2.5332289
    ];

    assert_close(&fde[..10], &EXPECTED_FIRST_10, 1e-5);
}

#[cfg(feature = "muvera")]
#[test]
fn test_muvera_deterministic() {
    let muvera1 = Muvera::new(128, Some(5), Some(16), Some(20), Some(42)).unwrap();
    let muvera2 = Muvera::new(128, Some(5), Some(16), Some(20), Some(42)).unwrap();

    let vectors: Vec<Vec<f32>> = (0..10)
        .map(|i| (0..128).map(|j| ((i * 128 + j) as f32).sin()).collect())
        .collect();

    let fde1 = muvera1.process_document(&vectors);
    let fde2 = muvera2.process_document(&vectors);

    assert_eq!(fde1, fde2, "MUVERA should be deterministic with same seed");
}

#[cfg(feature = "muvera")]
#[test]
fn test_muvera_different_seeds() {
    let muvera1 = Muvera::new(128, Some(5), Some(16), Some(20), Some(42)).unwrap();
    let muvera2 = Muvera::new(128, Some(5), Some(16), Some(20), Some(123)).unwrap();

    let vectors: Vec<Vec<f32>> = (0..10)
        .map(|i| (0..128).map(|j| ((i * 128 + j) as f32).sin()).collect())
        .collect();

    let fde1 = muvera1.process_document(&vectors);
    let fde2 = muvera2.process_document(&vectors);

    assert_ne!(fde1, fde2, "Different seeds should produce different results");
}

#[cfg(feature = "muvera")]
#[test]
fn test_muvera_doc_vs_query_processing() {
    let muvera = Muvera::new(128, Some(5), Some(16), Some(20), Some(42)).unwrap();

    let vectors: Vec<Vec<f32>> = (0..10)
        .map(|i| (0..128).map(|j| ((i * 128 + j) as f32).sin()).collect())
        .collect();

    let doc_fde = muvera.process_document(&vectors);
    let query_fde = muvera.process_query(&vectors);

    // Document and query processing should differ
    // (doc: normalize_by_count=true, fill_empty=true)
    // (query: normalize_by_count=false, fill_empty=false)
    assert_ne!(doc_fde, query_fde, "Document and query FDEs should differ");
}

#[cfg(feature = "muvera")]
#[test]
fn test_muvera_empty_cluster_handling() {
    // Test with very few vectors to ensure some clusters are empty
    let muvera = Muvera::new(8, Some(3), Some(4), Some(2), Some(42)).unwrap();

    // Only 2 vectors, but 2^3 = 8 clusters, so most will be empty
    let vectors: Vec<Vec<f32>> = vec![vec![1.0; 8], vec![-1.0; 8]];

    let doc_fde = muvera.process_document(&vectors);

    // Should not panic and should produce valid output
    assert_eq!(doc_fde.len(), muvera.embedding_size());
    assert!(doc_fde.iter().all(|&v| v.is_finite()), "FDE should have no NaN or Inf");
}

#[cfg(feature = "muvera")]
#[test]
fn test_muvera_dim_proj_validation() {
    // dim_proj > dim should fail
    let result = Muvera::new(128, Some(5), Some(256), Some(20), Some(42));
    assert!(result.is_err());
}
