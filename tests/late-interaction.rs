#[cfg(all(feature = "hf-hub", not(target_arch = "wasm32")))]
mod late_interaction_tests {
    use fastembed::{
        LateInteractionInitOptions, LateInteractionModel, LateInteractionTextEmbedding, Muvera,
    };

    #[test]
    #[ignore] // Ignore by default as it requires downloading models
    fn test_late_interaction_colbert() {
        let model = LateInteractionTextEmbedding::try_new(
            LateInteractionInitOptions::new(LateInteractionModel::ColBERTv2)
                .with_show_download_progress(true),
        )
        .expect("Failed to load model");

        let documents = vec![
            "Hello, world!",
            "This is a test document about information retrieval.",
            "ColBERT is a late interaction model that uses token-level embeddings.",
        ];

        // Generate document embeddings
        let doc_embeddings = model
            .passage_embed(documents, None)
            .expect("Failed to generate document embeddings");

        // Check that we got embeddings for all documents
        assert_eq!(doc_embeddings.len(), 3);

        // Check that embeddings are multi-vector (not pooled)
        for emb in &doc_embeddings {
            assert!(!emb.is_empty(), "Document embedding should not be empty");
            assert!(
                emb.len() > 1,
                "Late interaction embeddings should have multiple token vectors"
            );
            // ColBERT v2.0 has 128-dimensional embeddings
            assert_eq!(
                emb[0].len(),
                128,
                "Each token embedding should be 128-dimensional"
            );
        }

        // Test query embeddings
        let queries = vec!["What is ColBERT?", "information retrieval"];
        let query_embeddings = model
            .query_embed(queries, None)
            .expect("Failed to generate query embeddings");

        assert_eq!(query_embeddings.len(), 2);
        for emb in &query_embeddings {
            assert!(!emb.is_empty());
            assert_eq!(emb[0].len(), 128);
        }
    }

    #[test]
    #[ignore]
    fn test_muvera_postprocessing() {
        let model = LateInteractionTextEmbedding::try_new(
            LateInteractionInitOptions::new(LateInteractionModel::ColBERTv2)
                .with_show_download_progress(true),
        )
        .expect("Failed to load model");

        // Create MUVERA processor with recommended parameters
        let muvera = Muvera::from_model(&model, 6, 32, 20);

        // Expected output dimension: 20 * 64 * 32 = 40,960
        assert_eq!(muvera.get_output_dimension(), 40_960);

        let documents = vec!["This is a test document.", "Another sample text."];

        // Generate multi-vector embeddings
        let doc_embeddings = model
            .passage_embed(documents, None)
            .expect("Failed to generate embeddings");

        // Process with MUVERA to get fixed-dimensional encodings
        let fde_embeddings: Vec<_> = doc_embeddings
            .iter()
            .map(|emb| muvera.process_document(emb).expect("Failed to process"))
            .collect();

        // Check that we got fixed-dimensional encodings
        assert_eq!(fde_embeddings.len(), 2);
        for fde in &fde_embeddings {
            assert_eq!(fde.len(), 40_960, "FDE should have fixed dimension");
        }

        // Test query processing
        let query = vec!["test query"];
        let query_embeddings = model
            .query_embed(query, None)
            .expect("Failed to generate query embeddings");

        let query_fde = muvera
            .process_query(&query_embeddings[0])
            .expect("Failed to process query");
        assert_eq!(query_fde.len(), 40_960);
    }

    #[test]
    fn test_list_late_interaction_models() {
        let models = LateInteractionTextEmbedding::list_supported_models();
        assert!(!models.is_empty());

        // Check that we have the expected models
        let model_codes: Vec<_> = models.iter().map(|m| &m.model_code).collect();
        assert!(model_codes.contains(&&"colbert-ir/colbertv2.0".to_string()));
        assert!(model_codes.contains(&&"answerdotai/answerai-colbert-small-v1".to_string()));
        assert!(model_codes.contains(&&"jinaai/jina-colbert-v2".to_string()));
    }

    #[test]
    fn test_muvera_dimensions() {
        // Test different parameter combinations
        let test_cases = vec![
            (6, 32, 20, 40_960), // Recommended
            (5, 16, 20, 10_240), // Smaller
            (4, 8, 10, 1_280),   // Very small
        ];

        for (k_sim, dim_proj, r_reps, expected_dim) in test_cases {
            let muvera = Muvera::new(128, k_sim, dim_proj, r_reps, Some(42));
            assert_eq!(
                muvera.get_output_dimension(),
                expected_dim,
                "Dimension mismatch for k_sim={}, dim_proj={}, r_reps={}",
                k_sim,
                dim_proj,
                r_reps
            );
        }
    }
}
