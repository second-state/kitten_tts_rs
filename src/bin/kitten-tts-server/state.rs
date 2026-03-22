use kitten_tts::model::KittenTTS;
use std::sync::{Arc, Mutex};

/// Shared application state — the loaded KittenTTS model behind a Mutex
/// so that it can be safely shared across async request handlers.
pub type AppState = Arc<Mutex<KittenTTS>>;
