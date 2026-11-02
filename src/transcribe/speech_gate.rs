//! Bounded speech segmentation above the streaming backend contract.

use std::collections::VecDeque;

const RATE: usize = 16_000;
const HOP: usize = 2048;
const WINDOW: usize = 8192;
const PRE_ROLL: usize = 12288;
const MAX_PENDING: usize = 2 * RATE;

#[derive(Debug)]
enum Action {
    Start(Vec<f32>),
    Audio(Vec<f32>),
    End,
}

struct Segmenter {
    preroll: VecDeque<f32>,
    active: bool,
    quiet: usize,
    sent: usize,
    silence_samples: usize,
    max_samples: usize,
}

impl Segmenter {
    fn new(silence_samples: usize, max_samples: usize) -> Self {
        Self {
            preroll: VecDeque::with_capacity(PRE_ROLL),
            active: false,
            quiet: 0,
            sent: 0,
            silence_samples,
            max_samples,
        }
    }

    fn push(&mut self, audio: Vec<f32>, speech: bool) -> Vec<Action> {
        let mut actions = Vec::new();
        if !self.active {
            self.preroll.extend(&audio);
            while self.preroll.len() > PRE_ROLL {
                self.preroll.pop_front();
            }
            if !speech {
                return actions;
            }
            let initial: Vec<f32> = self.preroll.drain(..).collect();
            self.sent = initial.len();
            self.quiet = 0;
            self.active = true;
            actions.push(Action::Start(initial));
        } else {
            self.sent += audio.len();
            self.quiet = if speech { 0 } else { self.quiet + audio.len() };
            actions.push(Action::Audio(audio));
        }
        if self.quiet >= self.silence_samples || self.sent >= self.max_samples {
            self.active = false;
            self.preroll.clear();
            actions.push(Action::End);
        }
        actions
    }
}

use super::streaming::{StreamHandle, StreamingEvent};
use super::Transcriber;
use crate::config::SpeechGateConfig;
use crate::error::TranscribeError;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use whisper_rs::{WhisperVadContext, WhisperVadContextParams};

/// End the session when the microphone delivers nothing for this long. Long
/// enough to ride out Bluetooth reconnects and profile switches.
const STALL_SECS: u64 = 10;

fn failure(message: impl Into<String>) -> TranscribeError {
    TranscribeError::InferenceFailed(message.into())
}

trait SpeechDetector: Send {
    fn detect(&mut self, samples: &[f32]) -> Result<bool, TranscribeError>;
}

struct SileroDetector {
    context: WhisperVadContext,
    threshold: f32,
}
impl SpeechDetector for SileroDetector {
    fn detect(&mut self, samples: &[f32]) -> Result<bool, TranscribeError> {
        self.context
            .detect_speech(samples)
            .map_err(|e| failure(e.to_string()))?;
        Ok(self
            .context
            .probabilities()
            .windows(3)
            .any(|p| p.iter().all(|v| *v >= self.threshold)))
    }
}

/// Load the small detector. ASR weights remain in the supplied transcriber.
/// Defaults stay disabled; callers opt into this separate lifecycle explicitly.
pub fn start_gated_stream(
    transcriber: Arc<dyn Transcriber>,
    input: mpsc::Receiver<Vec<f32>>,
    config: SpeechGateConfig,
    batch: bool,
) -> Result<StreamHandle, TranscribeError> {
    if !(0.0..=1.0).contains(&config.threshold)
        || !config.threshold.is_finite()
        || !(128..=3000).contains(&config.silence_ms)
        || !(5..=60).contains(&config.max_speech_secs)
    {
        return Err(failure(
            "Invalid speech_gate threshold, silence_ms (128..3000), or max_speech_secs (5..60)",
        ));
    }
    whisper_rs::install_logging_hooks();
    let mut path = PathBuf::from(&config.model);
    if !path.is_absolute() {
        path = crate::config::Config::models_dir().join(path);
    }
    let mut params = WhisperVadContextParams::new();
    params.set_n_threads(1);
    params.set_use_gpu(false);
    let context = WhisperVadContext::new(
        path.to_str()
            .ok_or_else(|| failure("VAD model path is not UTF-8"))?,
        params,
    )
    .map_err(|e| failure(format!("Cannot load speech gate model: {e}")))?;
    let threshold = config.threshold;
    start_with_detector(
        transcriber,
        input,
        config,
        batch,
        Box::new(SileroDetector { context, threshold }),
    )
}

enum Command {
    Action(Action),
    Eof,
    Failed(String),
}

fn start_with_detector(
    transcriber: Arc<dyn Transcriber>,
    mut input: mpsc::Receiver<Vec<f32>>,
    config: SpeechGateConfig,
    batch: bool,
    mut detector: Box<dyn SpeechDetector>,
) -> Result<StreamHandle, TranscribeError> {
    let (events_tx, events) = mpsc::channel(64);
    let (cancel, mut cancellation) = oneshot::channel();
    let (commands_tx, mut commands) = mpsc::channel(32);
    let (detector_cancel, mut detector_cancellation) = oneshot::channel::<()>();
    let runtime = tokio::runtime::Handle::current();
    let input_count = Arc::new(AtomicU64::new(0));
    let input_counter = input_count.clone();
    let max_samples = config.max_speech_secs as usize * RATE;
    let silence_samples = config.silence_ms as usize * RATE / 1000;
    let detector_task = tokio::task::spawn_blocking(move || {
        let mut segmenter = Segmenter::new(silence_samples, max_samples);
        let mut window = VecDeque::from(vec![0.0; WINDOW]);
        let mut pending = Vec::with_capacity(HOP);
        loop {
            let next = runtime.block_on(async {
                tokio::select! {
                    _=&mut detector_cancellation=>None,
                    x=input.recv()=>x,
                }
            });
            let eof = next.is_none();
            let chunk = next.unwrap_or_default();
            input_counter.fetch_add(chunk.len() as u64, AtomicOrdering::Relaxed);
            // CPAL chunks are small. Refuse oversized input rather than retain
            // an arbitrarily large producer allocation inside this pipeline.
            if chunk.len() > RATE * 2 {
                let _ =
                    commands_tx.blocking_send(Command::Failed("Oversized capture chunk".into()));
                return;
            }
            let mut position = 0;
            loop {
                let take = (HOP - pending.len()).min(chunk.len() - position);
                pending.extend_from_slice(&chunk[position..position + take]);
                position += take;
                if pending.len() == HOP || (eof && !pending.is_empty()) {
                    window.extend(&pending);
                    while window.len() > WINDOW {
                        window.pop_front();
                    }
                    let samples: Vec<_> = window.iter().copied().collect();
                    let speech = match detector.detect(&samples) {
                        Ok(v) => v,
                        Err(e) => {
                            let _ = commands_tx.blocking_send(Command::Failed(e.to_string()));
                            return;
                        }
                    };
                    let audio = std::mem::replace(&mut pending, Vec::with_capacity(HOP));
                    for action in segmenter.push(audio, speech) {
                        if commands_tx.blocking_send(Command::Action(action)).is_err() {
                            return;
                        }
                    }
                }
                if position == chunk.len() {
                    break;
                }
            }
            if eof {
                if segmenter.active
                    && commands_tx
                        .blocking_send(Command::Action(Action::End))
                        .is_err()
                {
                    return;
                }
                let _ = commands_tx.blocking_send(Command::Eof);
                return;
            }
        }
    });
    let task = tokio::spawn(async move {
        struct Running {
            handle: StreamHandle,
            samples: Option<mpsc::Sender<Vec<f32>>>,
            ids: HashMap<u64, u64>,
        }
        let mut running: Option<Running> = None;
        let mut pending = VecDeque::new();
        let mut pending_samples = 0;
        let mut eof = false;
        let mut cancelled = false;
        let mut next_id = 0u64;
        let initial_drops = crate::audio::stream_dropped_samples();
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(1));
        let mut last_input_count = 0;
        let mut last_input_at = std::time::Instant::now();
        let pending_limit = if batch {
            max_samples + PRE_ROLL
        } else {
            MAX_PENDING
        };
        let outcome:Result<(),TranscribeError>=async {
            loop {
                while running.as_ref().is_none_or(|r|r.samples.is_some()) {
                    if matches!(pending.front(),Some(Action::Audio(_)))
                        && running.as_ref().and_then(|r|r.samples.as_ref()).is_some_and(|tx|tx.capacity()==0) { break; }
                    let Some(action)=pending.pop_front() else{break;};
                    match action {
                        Action::Start(audio)=>{
                            pending_samples-=audio.len();
                            if running.is_some(){return Err(failure("Overlapping speech segments"));}
                            let (tx,rx)=mpsc::channel(64);
                            let handle=if batch {start_batch(transcriber.clone(),rx,max_samples+PRE_ROLL)}
                                else {transcriber.as_streaming().ok_or_else(||failure("Backend has no streaming capability"))?.start_stream(rx)?};
                            tx.try_send(audio).map_err(|_|failure("Speech queue overflow on start"))?;
                            running=Some(Running{handle,samples:Some(tx),ids:HashMap::new()});
                        }
                        Action::Audio(audio)=>{
                            pending_samples-=audio.len();
                            running.as_ref().and_then(|r|r.samples.as_ref()).ok_or_else(||failure("Audio outside a speech segment"))?
                                .try_send(audio).map_err(|_|failure("Recognizer fell behind: bounded speech queue overflow"))?;
                        }
                        Action::End=>{if let Some(r)=running.as_mut(){r.samples.take();}}
                    }
                }
                if eof && running.is_none() && pending.is_empty(){return Ok(());}
                let ready_tx=running.as_ref().and_then(|r|r.samples.clone());
                let blocked=matches!(pending.front(),Some(Action::Audio(_))) && ready_tx.as_ref().is_some_and(|tx|tx.capacity()==0);
                tokio::select!{
                    _=&mut cancellation=>{cancelled=true;return Ok(());},
                    event=async{running.as_mut().unwrap().handle.events.recv().await},if running.is_some()=>{
                        match event {
                            Some(StreamingEvent::Ended)=>{
                                let r=running.take().unwrap();
                                r.handle.task.await.map_err(|e|failure(e.to_string()))??;
                                tokio::select! {
                                    result=events_tx.send(StreamingEvent::SegmentEnded)=>result.map_err(|_|failure("Output closed"))?,
                                    _=&mut cancellation=>{cancelled=true;return Ok(());},
                                }
                            }
                            None=>return Err(failure("Recognizer closed without Ended")),
                            Some(StreamingEvent::Error(e))=>return Err(e),
                            Some(mut event)=>{
                                let id=match &mut event {
                                    StreamingEvent::Partial{segment_id,..}|StreamingEvent::Final{segment_id,..}|StreamingEvent::Replace{segment_id,..}=>Some(segment_id),
                                    _=>None,
                                };
                                if let Some(id)=id {
                                    let mapped=running.as_mut().unwrap().ids.entry(*id).or_insert_with(||{let n=next_id;next_id+=1;n});
                                    *id = *mapped;
                                }
                                tokio::select! {
                                    result=events_tx.send(event)=>result.map_err(|_|failure("Output closed"))?,
                                    _=&mut cancellation=>{cancelled=true;return Ok(());},
                                }
                            }
                        }
                    }
                    _=async{ready_tx.unwrap().reserve_owned().await},if blocked=>{},
                    command=commands.recv(),if !eof=>{
                        match command {
                            Some(Command::Action(action))=>{
                                if let Action::Start(x)|Action::Audio(x)=&action {
                                    pending_samples+=x.len();
                                    if pending_samples>pending_limit{return Err(failure("Recognizer cannot keep up: bounded pending speech buffer exceeded"));}
                                }
                                pending.push_back(action);
                            }
                            Some(Command::Eof)=>eof=true,
                            Some(Command::Failed(e))=>return Err(failure(e)),
                            None=>return Err(failure("Speech detector stopped unexpectedly")),
                        }
                    }
                    _=tick.tick()=>{
                        if crate::audio::stream_dropped_samples()>initial_drops {return Err(failure("Capture overflow: audio was lost; stop and retry"));}
                        let input_samples=input_count.load(AtomicOrdering::Relaxed);
                        if input_samples!=last_input_count {last_input_count=input_samples;last_input_at=std::time::Instant::now();}
                        else if !eof && last_input_at.elapsed()>=std::time::Duration::from_secs(STALL_SECS) {return Err(failure(format!("Microphone delivered no audio for {STALL_SECS} seconds")));}
                    },
                }
            }
        }.await;
        // Drop event receivers before joining workers so blocked sends cannot
        // deadlock cancellation. In-flight native inference must finish first.
        drop(commands);
        let _ = detector_cancel.send(());
        let _ = detector_task.await;
        if let Some(mut r) = running.take() {
            // A capture-side failure must not erase the current accepted tail.
            // Drop input and drain it before surfacing the fault. User cancel
            // remains the one path that deliberately discards pending speech.
            r.samples.take();
            if !cancelled {
                let drain = async {
                    while let Some(mut event) = r.handle.events.recv().await {
                        match event {
                            StreamingEvent::Ended => break,
                            StreamingEvent::Error(_) => continue,
                            _ => {}
                        }
                        let id = match &mut event {
                            StreamingEvent::Partial { segment_id, .. }
                            | StreamingEvent::Final { segment_id, .. }
                            | StreamingEvent::Replace { segment_id, .. } => Some(segment_id),
                            _ => None,
                        };
                        if let Some(id) = id {
                            *id = *r.ids.entry(*id).or_insert_with(|| {
                                let n = next_id;
                                next_id += 1;
                                n
                            });
                        }
                        if events_tx.send(event).await.is_err() {
                            break;
                        }
                    }
                };
                tokio::select! {
                    _=&mut cancellation=>{cancelled=true;},
                    _=tokio::time::timeout(std::time::Duration::from_secs(10),drain)=>{},
                }
            }
            let _ = r.handle.cancel.send(());
            drop(r.handle.events);
            let _ = r.handle.task.await;
        }
        if !cancelled {
            if let Err(e) = outcome {
                let _ = events_tx.send(StreamingEvent::Error(e)).await;
            }
            let _ = events_tx.send(StreamingEvent::Ended).await;
        }
        Ok(())
    });
    Ok(StreamHandle {
        events,
        cancel,
        task,
    })
}

fn start_batch(
    transcriber: Arc<dyn Transcriber>,
    mut rx: mpsc::Receiver<Vec<f32>>,
    limit: usize,
) -> StreamHandle {
    let (tx, events) = mpsc::channel(4);
    let (cancel, mut cancellation) = oneshot::channel();
    let task = tokio::spawn(async move {
        let mut audio = Vec::new();
        loop {
            tokio::select! {
                _=&mut cancellation=>return Ok(()),
                chunk=rx.recv()=>match chunk {
                    Some(chunk)=>{if audio.len()+chunk.len()>limit{return Err(failure("Batch segment limit exceeded"));}audio.extend(chunk);},
                    None=>break,
                }
            }
        }
        let work = tokio::task::spawn_blocking(move || transcriber.transcribe(&audio));
        let result = work.await.map_err(|e| failure(e.to_string()))?;
        if cancellation.try_recv().is_ok() {
            return Ok(());
        }
        match result {
            Ok(text) => {
                let _ = tx
                    .send(StreamingEvent::Final {
                        text,
                        segment_id: 0,
                    })
                    .await;
            }
            Err(e) => {
                let _ = tx.send(StreamingEvent::Error(e)).await;
            }
        }
        let _ = tx.send(StreamingEvent::Ended).await;
        Ok(())
    });
    StreamHandle {
        events,
        cancel,
        task,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn capture_failure_drains_existing_final_before_error() {
        let backend = Arc::new(FakeBackend {
            starts: Arc::new(AtomicUsize::new(0)),
            revision: false,
            partial_count: 1,
        });
        let (tx, rx) = mpsc::channel(8);
        let mut h = start_with_detector(
            backend,
            rx,
            SpeechGateConfig::default(),
            false,
            Box::new(LevelDetector),
        )
        .unwrap();
        tx.send(vec![0.5; 3000]).await.unwrap();
        tx.send(vec![0.0; 40000]).await.unwrap();
        drop(tx);
        let mut order = Vec::new();
        while let Some(e) = h.events.recv().await {
            match e {
                StreamingEvent::Final { .. } => order.push("final"),
                StreamingEvent::Error(_) => order.push("error"),
                StreamingEvent::Ended => {
                    order.push("end");
                    break;
                }
                _ => {}
            }
        }
        h.task.await.unwrap().unwrap();
        assert_eq!(order, vec!["final", "error", "end"]);
    }

    #[tokio::test]
    async fn stalled_capture_ends_instead_of_listening_forever() {
        let backend = Arc::new(FakeBackend {
            starts: Arc::new(AtomicUsize::new(0)),
            revision: false,
            partial_count: 1,
        });
        let (_tx, rx) = mpsc::channel(8);
        let mut h = start_with_detector(
            backend,
            rx,
            SpeechGateConfig::default(),
            false,
            Box::new(LevelDetector),
        )
        .unwrap();
        let event = tokio::time::timeout(
            std::time::Duration::from_secs(STALL_SECS + 2),
            h.events.recv(),
        )
        .await
        .expect("capture stall was not detected");
        assert!(matches!(event, Some(StreamingEvent::Error(_))));
        h.task.await.unwrap().unwrap();
    }

    struct SlowBatch;
    impl Transcriber for SlowBatch {
        fn transcribe(&self, _: &[f32]) -> Result<String, TranscribeError> {
            std::thread::sleep(std::time::Duration::from_millis(2200));
            Ok(" complete".into())
        }
    }
    #[tokio::test]
    async fn resumed_batch_speech_is_bounded_by_segment_size_not_two_seconds() {
        let (tx, rx) = mpsc::channel(64);
        let mut h = start_with_detector(
            Arc::new(SlowBatch),
            rx,
            SpeechGateConfig::default(),
            true,
            Box::new(LevelDetector),
        )
        .unwrap();
        let producer = tokio::spawn(async move {
            for (speech, n) in [(true, 4), (false, 14), (true, 32), (false, 14)] {
                for _ in 0..n {
                    tx.send(vec![if speech { 0.5 } else { 0.0 }; HOP])
                        .await
                        .unwrap();
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                }
            }
        });
        let mut finals = 0;
        while let Some(e) = h.events.recv().await {
            match e {
                StreamingEvent::Final { .. } => finals += 1,
                StreamingEvent::Error(e) => panic!("{e}"),
                StreamingEvent::Ended => break,
                _ => {}
            }
        }
        producer.await.unwrap();
        h.task.await.unwrap().unwrap();
        assert_eq!(finals, 2);
    }

    struct LevelDetector;
    impl SpeechDetector for LevelDetector {
        fn detect(&mut self, samples: &[f32]) -> Result<bool, TranscribeError> {
            Ok(samples.iter().any(|v| v.abs() > 0.2))
        }
    }

    struct FakeBackend {
        starts: Arc<AtomicUsize>,
        revision: bool,
        partial_count: usize,
    }
    impl Transcriber for FakeBackend {
        fn transcribe(&self, samples: &[f32]) -> Result<String, TranscribeError> {
            assert!(samples.len() <= 61 * RATE);
            self.starts.fetch_add(1, Ordering::SeqCst);
            Ok(" batch".into())
        }
        fn as_streaming(&self) -> Option<&dyn super::super::streaming::StreamingTranscriber> {
            Some(self)
        }
    }
    impl super::super::streaming::StreamingTranscriber for FakeBackend {
        fn start_stream(
            &self,
            mut rx: mpsc::Receiver<Vec<f32>>,
        ) -> Result<StreamHandle, TranscribeError> {
            self.starts.fetch_add(1, Ordering::SeqCst);
            let (tx, events) = mpsc::channel(16);
            let (cancel, mut cancellation) = oneshot::channel();
            let revision = self.revision;
            let partial_count = self.partial_count;
            let task = tokio::spawn(async move {
                loop {
                    tokio::select! {
                        _=&mut cancellation=>return Ok(()),
                        x=rx.recv()=>if x.is_none(){break;},
                    }
                }
                for _ in 0..partial_count {
                    let _ = tx
                        .send(StreamingEvent::Partial {
                            text: " word".into(),
                            segment_id: 0,
                        })
                        .await;
                }
                let event = if revision {
                    StreamingEvent::Replace {
                        backspace: 5,
                        text: " revised".into(),
                        segment_id: 0,
                    }
                } else {
                    StreamingEvent::Final {
                        text: " tail".into(),
                        segment_id: 0,
                    }
                };
                let _ = tx.send(event).await;
                let _ = tx.send(StreamingEvent::Ended).await;
                Ok(())
            });
            Ok(StreamHandle {
                events,
                cancel,
                task,
            })
        }
    }

    #[tokio::test]
    async fn cancellation_joins_even_when_output_is_full() {
        let backend = Arc::new(FakeBackend {
            starts: Arc::new(AtomicUsize::new(0)),
            revision: false,
            partial_count: 128,
        });
        let (tx, rx) = mpsc::channel(8);
        let handle = start_with_detector(
            backend,
            rx,
            SpeechGateConfig::default(),
            false,
            Box::new(LevelDetector),
        )
        .unwrap();
        tx.send(vec![0.5; 3000]).await.unwrap();
        drop(tx);
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let _ = handle.cancel.send(());
        tokio::time::timeout(std::time::Duration::from_millis(500), handle.task)
            .await
            .expect("cancel stuck behind full output")
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn quiet_never_starts_backend_and_eof_joins_worker() {
        let starts = Arc::new(AtomicUsize::new(0));
        let backend = Arc::new(FakeBackend {
            starts: starts.clone(),
            revision: false,
            partial_count: 1,
        });
        let (tx, rx) = mpsc::channel(64);
        let mut handle = start_with_detector(
            backend,
            rx,
            SpeechGateConfig::default(),
            false,
            Box::new(LevelDetector),
        )
        .unwrap();
        for _ in 0..60 {
            tx.send(vec![0.0; HOP]).await.unwrap();
        }
        drop(tx);
        assert!(matches!(
            handle.events.recv().await,
            Some(StreamingEvent::Ended)
        ));
        handle.task.await.unwrap().unwrap();
        assert_eq!(starts.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn repeated_segments_preserve_final_and_revision_events_before_outer_end() {
        for revision in [false, true] {
            let starts = Arc::new(AtomicUsize::new(0));
            let backend = Arc::new(FakeBackend {
                starts: starts.clone(),
                revision,
                partial_count: 1,
            });
            let (tx, rx) = mpsc::channel(64);
            let mut handle = start_with_detector(
                backend,
                rx,
                SpeechGateConfig::default(),
                false,
                Box::new(LevelDetector),
            )
            .unwrap();
            let producer = tokio::spawn(async move {
                for _ in 0..3 {
                    for _ in 0..4 {
                        tx.send(vec![0.5; HOP]).await.unwrap();
                        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                    }
                    for _ in 0..14 {
                        tx.send(vec![0.0; HOP]).await.unwrap();
                        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                    }
                }
            });
            let mut finals = 0;
            let mut boundaries = 0;
            let mut ids = Vec::new();
            while let Some(event) = handle.events.recv().await {
                match event {
                    StreamingEvent::Final { segment_id, .. }
                    | StreamingEvent::Replace { segment_id, .. } => {
                        finals += 1;
                        ids.push(segment_id);
                    }
                    StreamingEvent::SegmentEnded => boundaries += 1,
                    StreamingEvent::Ended => break,
                    StreamingEvent::Error(e) => panic!("{e}"),
                    _ => {}
                }
            }
            producer.await.unwrap();
            handle.task.await.unwrap().unwrap();
            assert_eq!(starts.load(Ordering::SeqCst), 3);
            assert_eq!((finals, boundaries), (3, 3));
            assert!(ids.windows(2).all(|x| x[0] < x[1]));
        }
    }

    #[tokio::test]
    async fn batch_hold_decodes_only_accepted_speech_on_eof() {
        let starts = Arc::new(AtomicUsize::new(0));
        let backend = Arc::new(FakeBackend {
            starts: starts.clone(),
            revision: false,
            partial_count: 1,
        });
        let (tx, rx) = mpsc::channel(8);
        let mut handle = start_with_detector(
            backend,
            rx,
            SpeechGateConfig::default(),
            true,
            Box::new(LevelDetector),
        )
        .unwrap();
        tx.send(vec![0.5; 3000]).await.unwrap();
        drop(tx);
        let mut text = String::new();
        while let Some(event) = handle.events.recv().await {
            match event {
                StreamingEvent::Final { text: t, .. } => text.push_str(&t),
                StreamingEvent::Error(e) => panic!("{e}"),
                StreamingEvent::Ended => break,
                _ => {}
            }
        }
        handle.task.await.unwrap().unwrap();
        assert_eq!(text, " batch");
        assert_eq!(starts.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn silence_never_opens_a_decoder_and_preroll_is_bounded() {
        let mut gate = Segmenter::new(5 * HOP, 100 * HOP);
        for _ in 0..10_000 {
            assert!(gate.push(vec![0.0; HOP], false).is_empty());
            assert!(gate.preroll.len() <= PRE_ROLL);
        }
    }

    #[test]
    fn onset_includes_preroll_once_and_endpoint_rearms() {
        let mut gate = Segmenter::new(2 * HOP, 100 * HOP);
        gate.push(vec![1.0; HOP], false);
        let start = gate.push(vec![2.0; HOP], true);
        assert!(
            matches!(&start[..], [Action::Start(x)] if x.len()==2*HOP && x[0]==1.0 && x[HOP]==2.0)
        );
        assert!(matches!(
            &gate.push(vec![3.0; HOP], false)[..],
            [Action::Audio(_)]
        ));
        assert!(matches!(
            &gate.push(vec![4.0; HOP], false)[..],
            [Action::Audio(_), Action::End]
        ));
        assert!(
            matches!(&gate.push(vec![5.0;HOP],true)[..], [Action::Start(x)] if x.len()==HOP && x[0]==5.0)
        );
    }

    #[test]
    fn hard_limit_preserves_contiguous_audio_without_replaying_preroll() {
        let mut gate = Segmenter::new(2 * HOP, 2 * HOP);
        let mut accepted = Vec::new();
        let mut ends = 0;
        for n in 1..=8 {
            for action in gate.push(vec![n as f32; HOP], true) {
                match action {
                    Action::Start(x) | Action::Audio(x) => accepted.extend(x),
                    Action::End => ends += 1,
                }
            }
        }
        assert_eq!(ends, 4);
        for n in 1..=8 {
            assert!(accepted[(n - 1) * HOP..n * HOP]
                .iter()
                .all(|v| *v == n as f32));
        }
        assert_eq!(accepted.len(), 8 * HOP);
    }
}
