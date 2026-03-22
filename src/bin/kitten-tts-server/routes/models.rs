use axum::Json;
use serde::Serialize;

#[derive(Serialize)]
pub struct ModelInfo {
    id: &'static str,
    object: &'static str,
    owned_by: &'static str,
}

#[derive(Serialize)]
pub struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelInfo>,
}

pub async fn list_models() -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelInfo {
            id: "kitten-tts",
            object: "model",
            owned_by: "kittenml",
        }],
    })
}
