// ! Late interaction text embedding module, containing the main struct [LateInteractionTextEmbedding]
//! and its initialization options.
//!
//! Late interaction models like ColBERT generate embeddings for each token in the input text,
//! producing a matrix of embeddings rather than a single vector. This enables more fine-grained
//! matching using operations like MaxSim.

// Constants.
const DEFAULT_BATCH_SIZE: usize = 256;
const DEFAULT_MAX_LENGTH: usize = 512;
const DEFAULT_QUERY_MAX_LENGTH: usize = 32;

// Type alias for multi-vector embeddings
/// Multi-vector embedding - a matrix of token embeddings
pub type MultiVectorEmbedding = Vec<Vec<f32>>;

// Initialization options.
mod init;
pub use init::*;

// The implementation of the embedding models.
mod r#impl;

// MUVERA post-processor
pub mod muvera;
