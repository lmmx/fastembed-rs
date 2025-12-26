//! MUVERA (Multi-Vector Retrieval via Fixed Dimensional Encodings) post-processor
//!
//! MUVERA transforms variable-length multi-vector embeddings into fixed-dimensional
//! single-vector representations that can be used with traditional vector search.
//!
//! ## Algorithm
//!
//! 1. **Space Partitioning**: Use SimHash with k_sim random hyperplanes to partition
//!    token embeddings into 2^k_sim buckets
//! 2. **Dimensionality Reduction**: Within each bucket, aggregate vectors and apply
//!    random linear projection to dim_proj dimensions
//! 3. **Repetition**: Repeat steps 1-2 with r_reps independent random projections
//!    and concatenate all resulting vectors
//!
//! Final dimension: r_reps * 2^k_sim * dim_proj

use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::{LateInteractionTextEmbedding, MultiVectorEmbedding};

/// MUVERA post-processor for transforming multi-vector embeddings to fixed-dimensional encodings
pub struct Muvera {
    /// Number of SimHash projections (creates 2^k_sim buckets)
    k_sim: usize,
    /// Dimension after random linear projection
    dim_proj: usize,
    /// Number of repetitions with independent projections
    r_reps: usize,
    /// Original embedding dimension
    embedding_dim: usize,
    /// Random projection matrices for SimHash (r_reps x k_sim x embedding_dim)
    simhash_projections: Vec<Vec<Array1<f32>>>,
    /// Random projection matrices for dimensionality reduction (r_reps x 2^k_sim x dim_proj x embedding_dim)
    dim_reduction_projections: Vec<Vec<Array2<f32>>>,
}

impl Muvera {
    /// Create a MUVERA processor from a late interaction model
    ///
    /// # Arguments
    /// * `model` - The late interaction model
    /// * `k_sim` - Number of SimHash projections (typically 4-6)
    /// * `dim_proj` - Dimension after projection (typically 32)
    /// * `r_reps` - Number of repetitions (typically 20)
    ///
    /// # Recommended Parameters
    /// * k_sim=6, dim_proj=32, r_reps=20 -> final dimension: 40,960
    /// * k_sim=5, dim_proj=16, r_reps=20 -> final dimension: 10,240
    pub fn from_model(
        model: &LateInteractionTextEmbedding,
        k_sim: usize,
        dim_proj: usize,
        r_reps: usize,
    ) -> Self {
        Self::new(model.get_dimension(), k_sim, dim_proj, r_reps, None)
    }

    /// Create a MUVERA processor with custom parameters
    ///
    /// # Arguments
    /// * `embedding_dim` - Dimension of the token embeddings
    /// * `k_sim` - Number of SimHash projections
    /// * `dim_proj` - Dimension after projection
    /// * `r_reps` - Number of repetitions
    /// * `seed` - Optional seed for reproducible random projections
    pub fn new(
        embedding_dim: usize,
        k_sim: usize,
        dim_proj: usize,
        r_reps: usize,
        seed: Option<u64>,
    ) -> Self {
        let mut rng = if let Some(s) = seed {
            ChaCha8Rng::seed_from_u64(s)
        } else {
            ChaCha8Rng::from_entropy()
        };

        let num_buckets = 1 << k_sim; // 2^k_sim

        // Initialize random projection matrices
        let mut simhash_projections = Vec::with_capacity(r_reps);
        let mut dim_reduction_projections = Vec::with_capacity(r_reps);

        for _ in 0..r_reps {
            // SimHash projections: k_sim random hyperplanes
            let mut simhash_reps = Vec::with_capacity(k_sim);
            for _ in 0..k_sim {
                let proj: Vec<f32> = (0..embedding_dim)
                    .map(|_| rng.sample(rand::distributions::StandardNormal))
                    .collect();
                simhash_reps.push(Array1::from_vec(proj));
            }
            simhash_projections.push(simhash_reps);

            // Dimensionality reduction projections: one per bucket
            let mut dim_red_reps = Vec::with_capacity(num_buckets);
            for _ in 0..num_buckets {
                let mut proj_matrix = Array2::zeros((dim_proj, embedding_dim));
                for i in 0..dim_proj {
                    for j in 0..embedding_dim {
                        proj_matrix[[i, j]] = rng.sample(rand::distributions::StandardNormal);
                    }
                }
                // Normalize each row
                for mut row in proj_matrix.rows_mut() {
                    let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 1e-10 {
                        row /= norm;
                    }
                }
                dim_red_reps.push(proj_matrix);
            }
            dim_reduction_projections.push(dim_red_reps);
        }

        Self {
            k_sim,
            dim_proj,
            r_reps,
            embedding_dim,
            simhash_projections,
            dim_reduction_projections,
        }
    }

    /// Get the output dimension of the fixed-dimensional encoding
    pub fn get_output_dimension(&self) -> usize {
        self.r_reps * (1 << self.k_sim) * self.dim_proj
    }

    /// Process a multi-vector embedding to produce a fixed-dimensional encoding
    ///
    /// # Arguments
    /// * `embedding` - Multi-vector embedding (num_tokens x embedding_dim)
    ///
    /// # Returns
    /// Fixed-dimensional encoding of size (r_reps * 2^k_sim * dim_proj,)
    pub fn process(&self, embedding: &MultiVectorEmbedding) -> Result<Vec<f32>> {
        if embedding.is_empty() {
            return Err(anyhow::anyhow!("Empty embedding"));
        }

        let num_tokens = embedding.len();
        let token_dim = embedding[0].len();

        if token_dim != self.embedding_dim {
            return Err(anyhow::anyhow!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.embedding_dim,
                token_dim
            ));
        }

        let num_buckets = 1 << self.k_sim;
        let output_dim = self.get_output_dimension();
        let mut result = Vec::with_capacity(output_dim);

        // Process each repetition
        for rep in 0..self.r_reps {
            // Step 1: Compute SimHash bucket assignments for each token
            let mut bucket_assignments = Vec::with_capacity(num_tokens);

            for token_emb in embedding.iter() {
                let token_array = Array1::from_vec(token_emb.clone());
                let mut hash_code = 0usize;

                for (bit_idx, simhash_proj) in
                    self.simhash_projections[rep].iter().enumerate()
                {
                    let dot_product: f32 = token_array.iter().zip(simhash_proj.iter()).map(|(a, b)| a * b).sum();
                    if dot_product >= 0.0 {
                        hash_code |= 1 << bit_idx;
                    }
                }

                bucket_assignments.push(hash_code);
            }

            // Step 2: Aggregate tokens by bucket and apply dimensionality reduction
            for bucket_idx in 0..num_buckets {
                // Find all tokens in this bucket
                let bucket_tokens: Vec<&Vec<f32>> = embedding
                    .iter()
                    .zip(bucket_assignments.iter())
                    .filter(|(_, &assignment)| assignment == bucket_idx)
                    .map(|(token, _)| token)
                    .collect();

                let bucket_vector = if bucket_tokens.is_empty() {
                    // Empty bucket: use zero vector
                    Array1::zeros(self.embedding_dim)
                } else {
                    // Aggregate: compute mean of all tokens in bucket
                    let mut sum = Array1::zeros(self.embedding_dim);
                    for token in bucket_tokens.iter() {
                        for (i, &val) in token.iter().enumerate() {
                            sum[i] += val;
                        }
                    }
                    sum / bucket_tokens.len() as f32
                };

                // Apply dimensionality reduction
                let proj_matrix = &self.dim_reduction_projections[rep][bucket_idx];
                let reduced = proj_matrix.dot(&bucket_vector);

                // Append to result
                result.extend(reduced.iter());
            }
        }

        Ok(result)
    }

    /// Process a document embedding (alias for `process`)
    pub fn process_document(&self, embedding: &MultiVectorEmbedding) -> Result<Vec<f32>> {
        self.process(embedding)
    }

    /// Process a query embedding (alias for `process`)
    pub fn process_query(&self, embedding: &MultiVectorEmbedding) -> Result<Vec<f32>> {
        self.process(embedding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_muvera_dimension() {
        let muvera = Muvera::new(128, 6, 32, 20, Some(42));
        assert_eq!(muvera.get_output_dimension(), 20 * 64 * 32); // 40,960
    }

    #[test]
    fn test_muvera_process() {
        let embedding_dim = 128;
        let k_sim = 4; // 16 buckets
        let dim_proj = 8;
        let r_reps = 2;

        let muvera = Muvera::new(embedding_dim, k_sim, dim_proj, r_reps, Some(42));

        // Create a dummy multi-vector embedding (10 tokens x 128 dimensions)
        let mut embedding = Vec::new();
        for i in 0..10 {
            let token: Vec<f32> = (0..embedding_dim).map(|j| (i * j) as f32 * 0.01).collect();
            embedding.push(token);
        }

        let result = muvera.process(&embedding).unwrap();

        // Check output dimension: r_reps * 2^k_sim * dim_proj = 2 * 16 * 8 = 256
        assert_eq!(result.len(), 2 * 16 * 8);
    }

    #[test]
    fn test_muvera_empty_embedding() {
        let muvera = Muvera::new(128, 4, 8, 2, Some(42));
        let empty_embedding: MultiVectorEmbedding = Vec::new();
        assert!(muvera.process(&empty_embedding).is_err());
    }

    #[test]
    fn test_muvera_dimension_mismatch() {
        let muvera = Muvera::new(128, 4, 8, 2, Some(42));
        let wrong_dim_embedding = vec![vec![0.0; 64]]; // Wrong dimension
        assert!(muvera.process(&wrong_dim_embedding).is_err());
    }
}
