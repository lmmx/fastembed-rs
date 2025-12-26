//! Initialization options for the late interaction text embedding models.
//!

use crate::{
    common::TokenizerFiles,
    init::{HasMaxLength, InitOptionsWithLength},
    models::late_interaction::LateInteractionModel,
    OutputKey, QuantizationMode,
};
use ort::{execution_providers::ExecutionProviderDispatch, session::Session};
use tokenizers::Tokenizer;

use super::{DEFAULT_MAX_LENGTH, DEFAULT_QUERY_MAX_LENGTH};

impl HasMaxLength for LateInteractionModel {
    const MAX_LENGTH: usize = DEFAULT_MAX_LENGTH;
}

/// Options for initializing the LateInteractionTextEmbedding model
pub type LateInteractionInitOptions = InitOptionsWithLength<LateInteractionModel>;

/// Options for initializing UserDefinedLateInteractionModel
///
/// Model files are held by the UserDefinedLateInteractionModel struct
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct InitOptionsUserDefined {
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub max_length: usize,
    pub query_max_length: usize,
}

impl InitOptionsUserDefined {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    pub fn with_execution_providers(
        mut self,
        execution_providers: Vec<ExecutionProviderDispatch>,
    ) -> Self {
        self.execution_providers = execution_providers;
        self
    }

    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    pub fn with_query_max_length(mut self, query_max_length: usize) -> Self {
        self.query_max_length = query_max_length;
        self
    }
}

impl Default for InitOptionsUserDefined {
    fn default() -> Self {
        Self {
            execution_providers: Default::default(),
            max_length: DEFAULT_MAX_LENGTH,
            query_max_length: DEFAULT_QUERY_MAX_LENGTH,
        }
    }
}

/// Convert LateInteractionInitOptions to InitOptionsUserDefined
///
/// This is useful for when the user wants to use the same options for both the default and user-defined models
impl From<LateInteractionInitOptions> for InitOptionsUserDefined {
    fn from(options: LateInteractionInitOptions) -> Self {
        InitOptionsUserDefined {
            execution_providers: options.execution_providers,
            max_length: options.max_length,
            query_max_length: DEFAULT_QUERY_MAX_LENGTH,
        }
    }
}

/// Struct for "bring your own" late interaction embedding models
///
/// The onnx_file and tokenizer_files are expecting the files' bytes
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserDefinedLateInteractionModel {
    pub onnx_file: Vec<u8>,
    pub tokenizer_files: TokenizerFiles,
    pub quantization: QuantizationMode,
    pub output_key: Option<OutputKey>,
}

impl UserDefinedLateInteractionModel {
    pub fn new(onnx_file: Vec<u8>, tokenizer_files: TokenizerFiles) -> Self {
        Self {
            onnx_file,
            tokenizer_files,
            quantization: QuantizationMode::None,
            output_key: None,
        }
    }

    pub fn with_quantization(mut self, quantization: QuantizationMode) -> Self {
        self.quantization = quantization;
        self
    }
}

/// Rust representation of the LateInteractionTextEmbedding model
pub struct LateInteractionTextEmbedding {
    pub tokenizer: Tokenizer,
    pub query_tokenizer: Tokenizer,
    pub(crate) session: Session,
    pub(crate) need_token_type_ids: bool,
    pub(crate) quantization: QuantizationMode,
    pub(crate) output_key: Option<OutputKey>,
    pub(crate) dim: usize,
}
