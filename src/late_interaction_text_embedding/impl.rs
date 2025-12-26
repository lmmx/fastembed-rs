//! The definition of the main struct for late interaction text embeddings - [`LateInteractionTextEmbedding`].

#[cfg(feature = "hf-hub")]
use crate::common::load_tokenizer_hf_hub;
use crate::{
    common::load_tokenizer,
    models::{late_interaction::models_list, ModelTrait},
    LateInteractionModel, ModelInfo, OutputKey, QuantizationMode,
};
#[cfg(feature = "hf-hub")]
use anyhow::Context;
use anyhow::Result;
#[cfg(feature = "hf-hub")]
use hf_hub::api::sync::ApiRepo;
use ndarray::Array;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
#[cfg(feature = "hf-hub")]
use std::path::PathBuf;
use std::thread::available_parallelism;
use tokenizers::{AddedToken, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

#[cfg(feature = "hf-hub")]
use super::LateInteractionInitOptions;
use super::{
    InitOptionsUserDefined, LateInteractionTextEmbedding, MultiVectorEmbedding,
    UserDefinedLateInteractionModel, DEFAULT_BATCH_SIZE, DEFAULT_QUERY_MAX_LENGTH,
};

impl LateInteractionTextEmbedding {
    /// Try to generate a new LateInteractionTextEmbedding Instance
    ///
    /// Uses the highest level of Graph optimization
    ///
    /// Uses the total number of CPUs available as the number of intra-threads
    #[cfg(feature = "hf-hub")]
    pub fn try_new(options: LateInteractionInitOptions) -> Result<Self> {
        let LateInteractionInitOptions {
            max_length,
            model_name,
            execution_providers,
            cache_dir,
            show_download_progress,
        } = options;
        let threads = available_parallelism()?.get();

        let model_repo = LateInteractionTextEmbedding::retrieve_model(
            model_name.clone(),
            cache_dir.clone(),
            show_download_progress,
        )?;

        let model_info = LateInteractionTextEmbedding::get_model_info(&model_name)?;
        let model_file_name = &model_info.model_file;
        let model_file_reference = model_repo
            .get(model_file_name)
            .context(format!("Failed to retrieve {}", model_file_name))?;

        if !model_info.additional_files.is_empty() {
            for file in &model_info.additional_files {
                model_repo
                    .get(file)
                    .context(format!("Failed to retrieve {}", file))?;
            }
        }

        let session = Session::builder()?
            .with_execution_providers(execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(model_file_reference)?;

        // Adjust max_length to leave room for marker token
        let adjusted_max_length = if max_length > 1 {
            max_length - 1
        } else {
            max_length
        };

        let tokenizer = load_tokenizer_hf_hub(model_repo.clone(), adjusted_max_length)?;

        // For queries, use separate tokenizer with MASK padding
        let mut query_tokenizer = load_tokenizer_hf_hub(model_repo.clone(), adjusted_max_length)?;

        // Get mask token and configure query tokenizer
        let (mask_token, mask_token_id) = Self::get_mask_token_for_model(&model_name, &tokenizer)?;

        // Configure query tokenizer to pad with MASK tokens to MIN_QUERY_LENGTH
        query_tokenizer = query_tokenizer
            .with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::Fixed(DEFAULT_QUERY_MAX_LENGTH),
                pad_token: mask_token.clone(),
                pad_id: mask_token_id,
                ..Default::default()
            }))
            .clone();

        // Get marker token IDs
        let (query_marker_id, doc_marker_id) =
            Self::get_marker_token_ids_for_model(&model_name)?;

        // Build skip list from punctuation
        let skip_list = Self::build_skip_list(&tokenizer);

        // Get pad token ID
        let pad_token_id = tokenizer.get_padding().map(|p| p.pad_id).unwrap_or(0);

        Ok(Self::new(
            tokenizer,
            query_tokenizer,
            session,
            LateInteractionTextEmbedding::get_quantization_mode(&model_name),
            model_info.output_key.clone(),
            model_info.dim,
            mask_token_id,
            pad_token_id,
            skip_list,
            query_marker_id,
            doc_marker_id,
        ))
    }

    /// Create a LateInteractionTextEmbedding instance from model files provided by the user.
    ///
    /// This can be used for 'bring your own' embedding models
    pub fn try_new_from_user_defined(
        model: UserDefinedLateInteractionModel,
        options: InitOptionsUserDefined,
    ) -> Result<Self> {
        let InitOptionsUserDefined {
            execution_providers,
            max_length,
            query_max_length,
        } = options;

        let threads = available_parallelism()?.get();

        let session = Session::builder()?
            .with_execution_providers(execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_memory(&model.onnx_file)?;

        let adjusted_max_length = if max_length > 1 {
            max_length - 1
        } else {
            max_length
        };

        let tokenizer = load_tokenizer(model.tokenizer_files.clone(), adjusted_max_length)?;

        // Query tokenizer with MASK padding
        let mut query_tokenizer = load_tokenizer(model.tokenizer_files.clone(), adjusted_max_length)?;

        // Default MASK token for user-defined models
        let mask_token = "[MASK]".to_string();
        let mask_token_id = tokenizer
            .get_vocab(true)
            .get(&mask_token)
            .copied()
            .unwrap_or(0);

        query_tokenizer = query_tokenizer
            .with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::Fixed(query_max_length),
                pad_token: mask_token,
                pad_id: mask_token_id,
                ..Default::default()
            }))
            .clone();

        // Default marker tokens
        let query_marker_id = 1;
        let doc_marker_id = 2;

        let skip_list = Self::build_skip_list(&tokenizer);
        let pad_token_id = tokenizer.get_padding().map(|p| p.pad_id).unwrap_or(0);

        // Infer dimension from model output
        let dim = Self::infer_dimension(&session)?;

        Ok(Self::new(
            tokenizer,
            query_tokenizer,
            session,
            model.quantization,
            model.output_key,
            dim,
            mask_token_id,
            pad_token_id,
            skip_list,
            query_marker_id,
            doc_marker_id,
        ))
    }

    /// Private method to return an instance
    #[allow(clippy::too_many_arguments)]
    fn new(
        tokenizer: Tokenizer,
        query_tokenizer: Tokenizer,
        session: Session,
        quantization: QuantizationMode,
        output_key: Option<OutputKey>,
        dim: usize,
        mask_token_id: u32,
        pad_token_id: u32,
        skip_list: Vec<u32>,
        query_marker_token_id: u32,
        document_marker_token_id: u32,
    ) -> Self {
        let need_token_type_ids = session
            .inputs
            .iter()
            .any(|input| input.name == "token_type_ids");

        Self {
            tokenizer,
            query_tokenizer,
            session,
            need_token_type_ids,
            quantization,
            output_key,
            dim,
            mask_token_id,
            pad_token_id,
            skip_list,
            query_marker_token_id,
            document_marker_token_id,
        }
    }

    /// Build skip list from punctuation characters
    fn build_skip_list(tokenizer: &Tokenizer) -> Vec<u32> {
        let punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
        let mut skip_list = Vec::new();

        for ch in punctuation.chars() {
            let encoded = tokenizer.encode(ch.to_string(), false);
            if let Ok(enc) = encoded {
                if let Some(&id) = enc.get_ids().first() {
                    skip_list.push(id);
                }
            }
        }

        skip_list
    }

    /// Get mask token and ID for the model
    fn get_mask_token_for_model(
        model_name: &LateInteractionModel,
        tokenizer: &Tokenizer,
    ) -> Result<(String, u32)> {
        let mask_token = match model_name {
            LateInteractionModel::JinaColBERTv2 => "<mask>",
            _ => "[MASK]",
        }
        .to_string();

        let mask_token_id = tokenizer
            .get_vocab(true)
            .get(&mask_token)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Mask token {} not found in vocabulary", mask_token))?;

        Ok((mask_token, mask_token_id))
    }

    /// Get marker token IDs for query and document
    fn get_marker_token_ids_for_model(
        model_name: &LateInteractionModel,
    ) -> Result<(u32, u32)> {
        Ok(match model_name {
            LateInteractionModel::JinaColBERTv2 => (250002, 250003),
            _ => (1, 2), // ColBERT v2.0 and AnswerAI
        })
    }

    /// Return the LateInteractionTextEmbedding model's directory from cache or remote retrieval
    #[cfg(feature = "hf-hub")]
    fn retrieve_model(
        model: LateInteractionModel,
        cache_dir: PathBuf,
        show_download_progress: bool,
    ) -> anyhow::Result<ApiRepo> {
        use crate::common::pull_from_hf;

        pull_from_hf(model.to_string(), cache_dir, show_download_progress)
    }

    /// Get the quantization mode of the model.
    pub fn get_quantization_mode(_model_name: &LateInteractionModel) -> QuantizationMode {
        // Currently no quantized late interaction models
        QuantizationMode::None
    }

    /// Retrieve a list of supported models
    pub fn list_supported_models() -> Vec<ModelInfo<LateInteractionModel>> {
        models_list()
    }

    /// Get ModelInfo from LateInteractionModel
    pub fn get_model_info(
        model: &LateInteractionModel,
    ) -> Result<&ModelInfo<LateInteractionModel>> {
        LateInteractionModel::get_model_info(model).ok_or_else(|| {
            anyhow::Error::msg(format!(
                "Model {model:?} not found. Please check if the model is supported \
                by the current version."
            ))
        })
    }

    /// Get the embedding dimension
    pub fn get_dimension(&self) -> usize {
        self.dim
    }

    /// Infer dimension from the session
    fn infer_dimension(session: &Session) -> Result<usize> {
        // Try to infer from output shape
        if let Some(output) = session.outputs.first() {
            if let Some(shape) = &output.output_type.tensor_dimensions() {
                if shape.len() >= 3 {
                    // Expect shape like [batch_size, seq_len, dim]
                    if let Some(dim) = shape.last() {
                        return Ok(*dim as usize);
                    }
                }
            }
        }
        Err(anyhow::Error::msg(
            "Could not infer embedding dimension from model",
        ))
    }

    /// Generate embeddings for documents/passages
    ///
    /// This method is an alias for `passage_embed` and uses the document tokenizer
    /// with the full max_length setting.
    pub fn embed<S: AsRef<str> + Send + Sync>(
        &self,
        texts: impl AsRef<[S]>,
        batch_size: Option<usize>,
    ) -> Result<Vec<MultiVectorEmbedding>> {
        self.passage_embed(texts, batch_size)
    }

    /// Generate embeddings for passages using the passage tokenizer
    ///
    /// Late interaction models process passages with the full sequence length
    pub fn passage_embed<S: AsRef<str> + Send + Sync>(
        &self,
        texts: impl AsRef<[S]>,
        batch_size: Option<usize>,
    ) -> Result<Vec<MultiVectorEmbedding>> {
        self.embed_internal(texts.as_ref(), batch_size, false)
    }

    /// Generate embeddings for queries using the query tokenizer
    ///
    /// Late interaction models process queries differently, typically with shorter
    /// sequence lengths and different tokenization
    pub fn query_embed<S: AsRef<str> + Send + Sync>(
        &self,
        texts: impl AsRef<[S]>,
        batch_size: Option<usize>,
    ) -> Result<Vec<MultiVectorEmbedding>> {
        self.embed_internal(texts.as_ref(), batch_size, true)
    }

    /// Internal method to generate embeddings
    fn embed_internal<S: AsRef<str> + Send + Sync>(
        &self,
        texts: &[S],
        batch_size: Option<usize>,
        is_query: bool,
    ) -> Result<Vec<MultiVectorEmbedding>> {
        // Determine the batch size according to the quantization method used.
        let batch_size = match self.quantization {
            QuantizationMode::Dynamic => {
                if let Some(batch_size) = batch_size {
                    if batch_size < texts.len() {
                        return Err(anyhow::Error::msg(
                            "Dynamic quantization cannot be used with batching. \
                            This is due to the dynamic quantization process adjusting \
                            the data range to fit each batch, making the embeddings \
                            incompatible across batches. Try specifying a batch size \
                            of `None`, or use a model with static or no quantization.",
                        ));
                    } else {
                        texts.len()
                    }
                } else {
                    texts.len()
                }
            }
            _ => batch_size.unwrap_or(DEFAULT_BATCH_SIZE),
        };

        let tokenizer = if is_query {
            &self.query_tokenizer
        } else {
            &self.tokenizer
        };

        let marker_token_id = if is_query {
            self.query_marker_token_id
        } else {
            self.document_marker_token_id
        };

        let mut all_embeddings = Vec::with_capacity(texts.len());

        for batch in texts.chunks(batch_size) {
            // Encode the texts in the batch
            let inputs = batch.iter().map(|text| text.as_ref()).collect();
            let mut encodings = tokenizer.encode_batch(inputs, true).map_err(|e| {
                anyhow::Error::msg(e.to_string()).context("Failed to encode the batch.")
            })?;

            // Preprocess: Insert marker token at position 1
            for encoding in &mut encodings {
                let mut ids = encoding.get_ids().to_vec();
                let mut attention_mask = encoding.get_attention_mask().to_vec();

                // Insert marker token after CLS (position 1)
                ids.insert(1, marker_token_id);
                attention_mask.insert(1, 1);

                // Update the encoding (we'll need to recreate the arrays)
                // Note: This is a simplified approach - tokenizers crate doesn't easily support mutation
            }

            // Extract the encoding length and batch size (after inserting marker)
            let encoding_length = encodings
                .first()
                .ok_or_else(|| anyhow::anyhow!("Tokenizer returned empty encodings"))?
                .len()
                + 1; // +1 for marker token

            let current_batch_size = batch.len();
            let max_size = encoding_length * current_batch_size;

            // Preallocate arrays
            let mut ids_array = Vec::with_capacity(max_size);
            let mut mask_array = Vec::with_capacity(max_size);
            let mut type_ids_array = Vec::with_capacity(max_size);
            let mut original_ids = Vec::with_capacity(current_batch_size);

            for encoding in encodings.iter() {
                let mut ids = encoding.get_ids().to_vec();
                let mut mask = encoding.get_attention_mask().to_vec();
                let mut type_ids = encoding.get_type_ids().to_vec();

                // Store original IDs for post-processing
                original_ids.push(ids.clone());

                // Insert marker token at position 1
                ids.insert(1, marker_token_id);
                mask.insert(1, 1);
                type_ids.insert(1, 0);

                ids_array.extend(ids.iter().map(|x| *x as i64));
                mask_array.extend(mask.iter().map(|x| *x as i64));
                type_ids_array.extend(type_ids.iter().map(|x| *x as i64));
            }

            let inputs_ids_array =
                Array::from_shape_vec((current_batch_size, encoding_length), ids_array)?;
            let mut attention_mask_array =
                Array::from_shape_vec((current_batch_size, encoding_length), mask_array)?;
            let token_type_ids_array =
                Array::from_shape_vec((current_batch_size, encoding_length), type_ids_array)?;

            let mut session_inputs = ort::inputs![
                "input_ids" => Value::from_array(inputs_ids_array.clone())?,
                "attention_mask" => Value::from_array(attention_mask_array.clone())?,
            ];

            if self.need_token_type_ids {
                session_inputs.push((
                    "token_type_ids".into(),
                    Value::from_array(token_type_ids_array)?.into(),
                ));
            }

            let outputs = self
                .session
                .run(session_inputs)
                .map_err(anyhow::Error::new)?;

            // Extract the output tensor
            let output_tensor = if let Some(output_key) = &self.output_key {
                match output_key {
                    OutputKey::ByName(name) => outputs
                        .get(name)
                        .ok_or_else(|| anyhow::anyhow!("Output key {} not found", name))?,
                    OutputKey::ByOrder(idx) => outputs
                        .get(*idx)
                        .ok_or_else(|| anyhow::anyhow!("Output index {} not found", idx))?,
                    OutputKey::OnlyOne => {
                        if outputs.len() == 1 {
                            &outputs[0].1
                        } else {
                            return Err(anyhow::anyhow!("Expected exactly one output"));
                        }
                    }
                }
            } else {
                // Default: try to get last_hidden_state or the only output
                outputs
                    .get("last_hidden_state")
                    .or_else(|| {
                        if outputs.len() == 1 {
                            Some(&outputs[0].1)
                        } else {
                            None
                        }
                    })
                    .ok_or_else(|| anyhow::anyhow!("Could not find suitable output"))?
            };

            // Extract the array from the tensor
            let array = output_tensor.try_extract_tensor::<f32>()?;
            let mut array_view = array.view().to_owned();

            // Post-process based on whether this is a query or document
            if !is_query {
                // Document post-processing
                // 1. Zero out attention mask for skip_list tokens and pad tokens
                for (i, ids) in original_ids.iter().enumerate() {
                    for (j, &token_id) in ids.iter().enumerate() {
                        let j_adjusted = j + 1; // Account for marker token insertion
                        if j_adjusted < encoding_length {
                            if self.skip_list.contains(&token_id) || token_id == self.pad_token_id
                            {
                                attention_mask_array[[i, j_adjusted]] = 0;
                            }
                        }
                    }
                }

                // 2. Multiply embeddings by attention mask
                for i in 0..current_batch_size {
                    for j in 0..encoding_length {
                        let mask_val = attention_mask_array[[i, j]] as f32;
                        for k in 0..self.dim {
                            array_view[[i, j, k]] *= mask_val;
                        }
                    }
                }

                // 3. L2 normalize each token embedding
                for i in 0..current_batch_size {
                    for j in 0..encoding_length {
                        let mut norm = 0.0f32;
                        for k in 0..self.dim {
                            norm += array_view[[i, j, k]] * array_view[[i, j, k]];
                        }
                        norm = norm.sqrt().max(1e-12);

                        for k in 0..self.dim {
                            array_view[[i, j, k]] /= norm;
                        }
                    }
                }

                // 4. Return only tokens where attention_mask == 1
                for i in 0..current_batch_size {
                    let mut doc_embedding = Vec::new();
                    for j in 0..encoding_length {
                        if attention_mask_array[[i, j]] == 1 {
                            let mut token_embedding = Vec::with_capacity(self.dim);
                            for k in 0..self.dim {
                                token_embedding.push(array_view[[i, j, k]]);
                            }
                            doc_embedding.push(token_embedding);
                        }
                    }
                    all_embeddings.push(doc_embedding);
                }
            } else {
                // Query post-processing: return all embeddings as-is
                // For JinaColbert, attention mask is always 1
                if matches!(self.query_marker_token_id, 250002) {
                    // JinaColbert
                    attention_mask_array.fill(1);
                }

                for i in 0..current_batch_size {
                    let mut query_embedding = Vec::new();
                    for j in 0..encoding_length {
                        let mut token_embedding = Vec::with_capacity(self.dim);
                        for k in 0..self.dim {
                            token_embedding.push(array_view[[i, j, k]]);
                        }
                        query_embedding.push(token_embedding);
                    }
                    all_embeddings.push(query_embedding);
                }
            }
        }

        Ok(all_embeddings)
    }
}
