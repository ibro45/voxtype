use serde::{Deserialize, Serialize};

/// Optional bounded VAD segmentation above the backend runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SpeechGateConfig {
    pub enabled: bool,
    pub model: String,
    pub threshold: f32,
    pub silence_ms: u64,
    pub max_speech_secs: u64,
}

impl Default for SpeechGateConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: "ggml-silero-vad.bin".into(),
            threshold: 0.5,
            silence_ms: 640,
            max_speech_secs: 30,
        }
    }
}
