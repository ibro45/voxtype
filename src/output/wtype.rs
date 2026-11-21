//! wtype-based text output
//!
//! Uses wtype to simulate keyboard input on Wayland. This is the preferred
//! method on Wayland because:
//! - No daemon required (unlike ydotool)
//! - Better Unicode/CJK support
//!
//! Requires:
//! - wtype installed
//! - Running on Wayland (WAYLAND_DISPLAY set)

use super::TextOutput;
use crate::error::OutputError;
use std::process::Stdio;
use tokio::process::Command;

/// wtype-based text output
pub struct WtypeOutput {
    /// Whether to send Enter key after output
    auto_submit: bool,
    /// Text to append after transcription (before auto_submit)
    append_text: Option<String>,
    /// Delay between keystrokes in milliseconds
    type_delay_ms: u32,
    /// Delay before typing starts (ms), allows virtual keyboard to initialize
    pre_type_delay_ms: u32,
    /// Convert newlines to Shift+Enter (for apps where Enter submits)
    shift_enter_newlines: bool,
    /// Prefix output with Shift press/release (workaround for CJK first char drop)
    shift_prefix: bool,
}

impl WtypeOutput {
    /// Create a new wtype output
    pub fn new(
        auto_submit: bool,
        append_text: Option<String>,
        type_delay_ms: u32,
        pre_type_delay_ms: u32,
        shift_enter_newlines: bool,
        shift_prefix: bool,
    ) -> Self {
        Self {
            auto_submit,
            append_text,
            type_delay_ms,
            pre_type_delay_ms,
            shift_enter_newlines,
            shift_prefix,
        }
    }

    /// wtype arguments for typing text. The text itself is read from stdin
    /// (`-`), never passed as an argument: arguments are visible in
    /// /proc/<pid>/cmdline to every process of the same user, and dictated
    /// text is private.
    fn type_args(&self) -> Vec<String> {
        let mut args = Vec::new();
        // Pre-typing delay helps prevent first character drop
        if self.pre_type_delay_ms > 0 {
            args.extend(["-s".to_string(), self.pre_type_delay_ms.to_string()]);
        }
        if self.type_delay_ms > 0 {
            args.extend(["-d".to_string(), self.type_delay_ms.to_string()]);
        }
        // Shift prefix prevents the first CJK character drop in some apps
        if self.shift_prefix {
            args.extend(["-P", "Shift_L", "-p", "Shift_L"].map(String::from));
        }
        args.push("-".to_string());
        args
    }

    /// Type a string of text using wtype
    async fn type_text(&self, text: &str) -> Result<(), OutputError> {
        use tokio::io::AsyncWriteExt;

        if text.is_empty() {
            return Ok(());
        }

        let args = self.type_args();
        tracing::debug!(
            chars = text.chars().count(),
            "Running: wtype {}",
            args.join(" ")
        );

        let mut child = Command::new("wtype")
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    OutputError::WtypeNotFound
                } else {
                    OutputError::InjectionFailed(e.to_string())
                }
            })?;

        if let Some(mut stdin) = child.stdin.take() {
            let written = stdin.write_all(text.as_bytes()).await;
            // Closing stdin tells wtype the text is complete.
            drop(stdin);
            if let Err(e) = written {
                let _ = child.kill().await;
                return Err(OutputError::InjectionFailed(format!(
                    "wtype stdin write failed: {}",
                    e
                )));
            }
        }

        let output = child
            .wait_with_output()
            .await
            .map_err(|e| OutputError::InjectionFailed(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(OutputError::InjectionFailed(format!(
                "wtype failed: {}",
                stderr
            )));
        }

        Ok(())
    }

    /// Send Shift+Enter key combination using wtype
    async fn send_shift_enter(&self) -> Result<(), OutputError> {
        let output = Command::new("wtype")
            .args(["-M", "shift", "-k", "Return", "-m", "shift"])
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| {
                OutputError::InjectionFailed(format!("wtype Shift+Enter failed: {}", e))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::warn!("Failed to send Shift+Enter: {}", stderr);
        }

        Ok(())
    }

    /// Send Enter key using wtype
    async fn send_enter(&self) -> Result<(), OutputError> {
        let output = Command::new("wtype")
            .args(["-k", "Return"])
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| OutputError::InjectionFailed(format!("wtype Enter failed: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::warn!("Failed to send Enter key: {}", stderr);
        }

        Ok(())
    }

    /// Output text with newlines converted to Shift+Enter
    async fn output_with_shift_enter_newlines(&self, text: &str) -> Result<(), OutputError> {
        let segments: Vec<&str> = text.split('\n').collect();

        for (i, segment) in segments.iter().enumerate() {
            // Type the text segment
            if !segment.is_empty() {
                self.type_text(segment).await?;
            }

            // Send Shift+Enter between segments (not after the last one)
            if i < segments.len() - 1 {
                self.send_shift_enter().await?;
            }
        }

        Ok(())
    }
}

#[async_trait::async_trait]
impl TextOutput for WtypeOutput {
    async fn output(&self, text: &str) -> Result<(), OutputError> {
        if text.is_empty() {
            return Ok(());
        }

        // If shift_enter_newlines is enabled, process text with Shift+Enter for newlines
        if self.shift_enter_newlines && text.contains('\n') {
            self.output_with_shift_enter_newlines(text).await?;
        } else {
            self.type_text(text).await?;
        }

        // Append text if configured (e.g., a space to separate sentences)
        if let Some(ref append) = self.append_text {
            self.type_text(append).await?;
        }

        // Send Enter key if auto_submit is configured
        if self.auto_submit {
            self.send_enter().await?;
        }

        Ok(())
    }

    async fn is_available(&self) -> bool {
        // Just check if wtype exists in PATH
        // Don't check WAYLAND_DISPLAY - systemd services may not have it
        // wtype will fail naturally if Wayland isn't available
        Command::new("which")
            .arg("wtype")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .map(|s| s.success())
            .unwrap_or(false)
    }

    fn name(&self) -> &'static str {
        "wtype"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_is_never_an_argument() {
        let output = WtypeOutput::new(false, None, 1, 200, false, true);
        assert_eq!(
            output.type_args(),
            ["-s", "200", "-d", "1", "-P", "Shift_L", "-p", "Shift_L", "-"]
        );
        let plain = WtypeOutput::new(false, None, 0, 0, false, false);
        assert_eq!(plain.type_args(), ["-"]);
    }

    #[test]
    fn test_new() {
        let output = WtypeOutput::new(false, None, 0, 0, false, false);
        assert!(!output.auto_submit);
        assert_eq!(output.type_delay_ms, 0);
        assert_eq!(output.pre_type_delay_ms, 0);
        assert!(!output.shift_enter_newlines);
    }

    #[test]
    fn test_new_with_enter() {
        let output = WtypeOutput::new(true, None, 0, 0, false, false);
        assert!(output.auto_submit);
    }

    #[test]
    fn test_new_with_type_delay() {
        let output = WtypeOutput::new(false, None, 50, 0, false, false);
        assert!(!output.auto_submit);
        assert_eq!(output.type_delay_ms, 50);
        assert_eq!(output.pre_type_delay_ms, 0);
    }

    #[test]
    fn test_new_with_pre_type_delay() {
        let output = WtypeOutput::new(false, None, 0, 200, false, false);
        assert_eq!(output.type_delay_ms, 0);
        assert_eq!(output.pre_type_delay_ms, 200);
    }

    #[test]
    fn test_new_with_shift_enter_newlines() {
        let output = WtypeOutput::new(false, None, 0, 0, true, false);
        assert!(output.shift_enter_newlines);
    }

    #[test]
    fn test_new_with_shift_prefix() {
        let output = WtypeOutput::new(false, None, 0, 0, false, true);
        assert!(output.shift_prefix);
    }
}
