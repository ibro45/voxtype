//! Audio capture module
//!
//! Provides audio recording capabilities using cpal, which works with
//! PipeWire, PulseAudio, and ALSA backends.

pub mod cpal_capture;
pub mod devices;
pub mod dual_capture;
pub mod echo_cancel;
#[cfg(feature = "onnx-common")]
pub mod enhance;
pub mod feedback;
pub mod levels;
pub mod media;
pub mod resampler;

pub use cpal_capture::stream_dropped_samples;
pub use dual_capture::{AudioSourceType, DualCapture, DualSamples, SourcedSample};

use crate::config::AudioConfig;
use crate::error::AudioError;
use tokio::sync::mpsc;

/// Whether capture also keeps a recording beside the bounded chunk channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureRetention {
    /// Batch/eager consumers retrieve the recording through stop/get_samples.
    Retain,
    /// Streaming consumers use only the chunk channel and discard stop's audio.
    ChannelOnly,
}

/// Trait for audio capture implementations
#[async_trait::async_trait]
#[allow(
    clippy::double_must_use,
    reason = "async_trait adds #[must_use] to methods returning must-use futures"
)]
pub trait AudioCapture: Send + Sync {
    /// Start capturing audio
    /// Returns a channel receiver for audio chunks (f32 samples, mono, 16kHz)
    async fn start(&mut self) -> Result<mpsc::Receiver<Vec<f32>>, AudioError>;

    /// Stop capturing and return all retained samples (empty in ChannelOnly mode).
    async fn stop(&mut self) -> Result<Vec<f32>, AudioError>;

    /// Get current samples without stopping (for continuous recording modes)
    /// This drains the internal buffer and returns samples collected since the last call.
    /// ChannelOnly capture always returns an empty Vec.
    /// Returns an empty Vec if not yet started or already stopped.
    async fn get_samples(&mut self) -> Vec<f32>;
}

/// Factory function to create audio capture
pub fn create_capture(config: &AudioConfig) -> Result<Box<dyn AudioCapture>, AudioError> {
    Ok(Box::new(cpal_capture::CpalCapture::new(config)?))
}

/// Create capture with an explicit recording-retention policy.
pub fn create_capture_with_retention(
    config: &AudioConfig,
    retention: CaptureRetention,
) -> Result<Box<dyn AudioCapture>, AudioError> {
    Ok(Box::new(cpal_capture::CpalCapture::new_with_retention(
        config, retention,
    )?))
}
