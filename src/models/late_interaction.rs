use std::{collections::HashMap, convert::TryFrom, fmt::Display, str::FromStr, sync::OnceLock};

use super::{model_info::ModelInfo, ModelTrait};

/// Lazy static list of all available late interaction models.
static MODEL_MAP: OnceLock<HashMap<LateInteractionModel, ModelInfo<LateInteractionModel>>> =
    OnceLock::new();

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub enum LateInteractionModel {
    /// colbert-ir/colbertv2.0
    #[default]
    ColBERTv2,
    /// answerdotai/answerai-colbert-small-v1
    AnswerAIColBERTSmallV1,
    /// jinaai/jina-colbert-v2
    JinaColBERTv2,
}

/// Centralized function to initialize the models map.
fn init_models_map() -> HashMap<LateInteractionModel, ModelInfo<LateInteractionModel>> {
    let models_list = vec![
        ModelInfo {
            model: LateInteractionModel::ColBERTv2,
            dim: 128,
            description: String::from("ColBERT v2.0 - Late interaction model for efficient retrieval"),
            model_code: String::from("colbert-ir/colbertv2.0"),
            model_file: String::from("model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: LateInteractionModel::AnswerAIColBERTSmallV1,
            dim: 96,
            description: String::from(
                "Answer.AI ColBERT Small v1 - Smaller, faster ColBERT variant",
            ),
            model_code: String::from("answerdotai/answerai-colbert-small-v1"),
            model_file: String::from("vespa_colbert.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: LateInteractionModel::JinaColBERTv2,
            dim: 128,
            description: String::from("Jina AI ColBERT v2 - Multilingual late interaction model with 8192 context"),
            model_code: String::from("jinaai/jina-colbert-v2"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec!["onnx/model.onnx_data".to_string()],
            output_key: None,
        },
    ];

    models_list
        .into_iter()
        .fold(HashMap::new(), |mut map, model| {
            map.insert(model.model.clone(), model);
            map
        })
}

/// Get a map of all available late interaction models.
pub fn models_map() -> &'static HashMap<LateInteractionModel, ModelInfo<LateInteractionModel>> {
    MODEL_MAP.get_or_init(init_models_map)
}

/// Get a list of all available late interaction models.
///
/// This will assign new memory to the models list; where possible, use
/// [`models_map`] instead.
pub fn models_list() -> Vec<ModelInfo<LateInteractionModel>> {
    models_map().values().cloned().collect()
}

impl ModelTrait for LateInteractionModel {
    type Model = Self;

    /// Get model information by model code.
    fn get_model_info(model: &LateInteractionModel) -> Option<&ModelInfo<LateInteractionModel>> {
        models_map().get(model)
    }
}

impl Display for LateInteractionModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let model_info = LateInteractionModel::get_model_info(self).ok_or(std::fmt::Error)?;
        write!(f, "{}", model_info.model_code)
    }
}

impl FromStr for LateInteractionModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        models_list()
            .into_iter()
            .find(|m| m.model_code.eq_ignore_ascii_case(s))
            .map(|m| m.model)
            .ok_or_else(|| format!("Unknown late interaction model: {s}"))
    }
}

impl TryFrom<String> for LateInteractionModel {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}
