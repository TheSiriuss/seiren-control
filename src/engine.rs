//! The audio engine: one thread that captures the microphone, runs the voice chain on it and hands
//! the result to whoever wants it: the meters in the window, a take on disk, headphones, or a
//! virtual microphone other programs can pick.
//!
//! Capture and playback go through the PulseAudio protocol with `parec` / `pacat` (raw PCM over
//! pipes), which PipeWire serves too. The virtual microphone is a `pw-loopback` pair, or, where that
//! is missing, two PulseAudio modules. That keeps this file free of audio libraries, and the
//! microphone stays shared with every other program.

use crate::dsp::{Analyzer, Chain, DspParams, FrameMeters, FS, HOP, SPECTRUM_BANDS};
use crate::recorder::{RecordOpts, Recorder, Take};
use std::io::{self, Read, Write};
use std::os::unix::process::CommandExt;
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::mpsc::{channel, sync_channel, Receiver, RecvTimeoutError, Sender, SyncSender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const NAME_VIRTUAL_IN: &str = "seiren_clean_in";
const NAME_VIRTUAL_SRC: &str = "seiren_clean";
/// Frames (10 ms each) of the room tone collected when learning a noise profile.
const LEARN_FRAMES: usize = 200;

#[derive(Clone, Debug)]
pub enum SourceKind {
    /// The Seiren (or the default microphone) through PipeWire.
    Pulse,
    /// A made-up noisy voice, for demos and tests. `realtime = false` runs flat out.
    Synthetic { realtime: bool },
}

#[derive(Clone, Debug)]
pub struct Snapshot {
    pub running: bool,
    pub recording: bool,
    pub rec_secs: f32,
    /// Linear levels; the UI turns them into dB. Peaks fall back slowly.
    pub in_peak: f32,
    pub in_rms: f32,
    pub out_peak: f32,
    pub out_rms: f32,
    pub gate_open: bool,
    pub comp_gr_db: f32,
    pub limit_gr_db: f32,
    pub mouth_gr_db: f32,
    pub lufs: f32,
    pub spectrum: [f32; SPECTRUM_BANDS],
    pub monitor: bool,
    pub virtual_mic: bool,
    /// 0..1 while a noise profile is being learned, else None.
    pub learning: Option<f32>,
    pub dfn_loading: bool,
}

impl Default for Snapshot {
    fn default() -> Self {
        Snapshot {
            running: false,
            recording: false,
            rec_secs: 0.0,
            in_peak: 0.0,
            in_rms: 0.0,
            out_peak: 0.0,
            out_rms: 0.0,
            gate_open: true,
            comp_gr_db: 0.0,
            limit_gr_db: 0.0,
            mouth_gr_db: 0.0,
            lufs: -70.0,
            spectrum: [0.0; SPECTRUM_BANDS],
            monitor: false,
            virtual_mic: false,
            learning: None,
            dfn_loading: false,
        }
    }
}

pub enum EngCmd {
    SetParams(Box<DspParams>),
    /// Capture for the meters even when nothing is being recorded (the recording page is open).
    Run(bool),
    Record(RecordOpts),
    StopRecord,
    Monitor(bool),
    VirtualMic(bool),
    LearnNoise,
    Quit,
}

pub enum EngEvent {
    Info(String),
    Error(String),
    Take(Take),
    Profile(Vec<f32>),
    /// "monitor" or "virtual_mic" could not be started: the switch in the window must go back off.
    FeatureOff(&'static str),
}

pub type EngEmit = Arc<dyn Fn(EngEvent) + Send + Sync>;

pub struct Engine {
    pub tx: Sender<EngCmd>,
    pub snap: Arc<Mutex<Snapshot>>,
    handle: JoinHandle<()>,
}

impl Engine {
    pub fn spawn(kind: SourceKind, params: DspParams, emit: EngEmit) -> Engine {
        let (tx, rx) = channel();
        let snap = Arc::new(Mutex::new(Snapshot::default()));
        let s2 = snap.clone();
        let handle = thread::Builder::new().name("audio-engine".into()).spawn(move || Eng::new(kind, params, emit, s2).run(rx)).expect("engine thread");
        Engine { tx, snap, handle }
    }

    pub fn send(&self, c: EngCmd) {
        let _ = self.tx.send(c);
    }

    pub fn snapshot(&self) -> Snapshot {
        self.snap.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Stop everything and wait (an unfinished take is finished first).
    pub fn quit(self) {
        let _ = self.tx.send(EngCmd::Quit);
        let _ = self.handle.join();
    }
}

// ================================================================================================
// Child processes
// ================================================================================================

/// Spawn a child that dies with us (SIGTERM when this process goes away).
fn spawn_child(cmd: &mut Command) -> io::Result<Child> {
    unsafe {
        cmd.pre_exec(|| {
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
            Ok(())
        });
    }
    cmd.spawn()
}

fn pactl_names(kind: &str) -> Vec<String> {
    Command::new("pactl")
        .args(["list", "short", kind])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).lines().filter_map(|l| l.split('\t').nth(1).map(str::to_string)).collect())
        .unwrap_or_default()
}

/// The Seiren's PipeWire source, or None (then the system default microphone is used).
pub fn find_source() -> Option<String> {
    pactl_names("sources").into_iter().find(|n| n.contains("Razer_Seiren_V3") && !n.ends_with(".monitor"))
}

/// The Seiren's playback device (its headphone jack), or None (then the default output).
pub fn find_sink() -> Option<String> {
    pactl_names("sinks").into_iter().find(|n| n.contains("Razer_Seiren_V3"))
}

trait FrameSource: Send {
    fn read(&mut self, frame: &mut [f32]) -> io::Result<()>;
}

struct Parec {
    child: Child,
    out: ChildStdout,
    bytes: Vec<u8>,
}

impl Parec {
    fn open() -> io::Result<Parec> {
        let mut cmd = Command::new("parec");
        if let Some(src) = find_source() {
            cmd.arg(format!("--device={src}"));
        }
        cmd.args(["--format=float32le", "--rate=48000", "--channels=1", "--latency-msec=20", "--raw", "--client-name=seiren-control", "--stream-name=Seiren Control capture"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        let mut child = spawn_child(&mut cmd)?;
        let out = child.stdout.take().ok_or_else(|| io::Error::other("no stdout"))?;
        Ok(Parec { child, out, bytes: vec![0; HOP * 4] })
    }
}

impl FrameSource for Parec {
    fn read(&mut self, frame: &mut [f32]) -> io::Result<()> {
        self.out.read_exact(&mut self.bytes)?;
        for (s, b) in frame.iter_mut().zip(self.bytes.chunks_exact(4)) {
            *s = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        }
        Ok(())
    }
}

impl Drop for Parec {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A noisy voice-like signal: a gliding fundamental with harmonics in bursts, over white noise.
struct Synth {
    n: u64,
    phase: f64,
    seed: u32,
    next: Instant,
    realtime: bool,
}

impl FrameSource for Synth {
    fn read(&mut self, frame: &mut [f32]) -> io::Result<()> {
        if self.realtime {
            let now = Instant::now();
            if self.next > now {
                thread::sleep(self.next - now);
            }
            self.next += Duration::from_millis(10);
        }
        for s in frame.iter_mut() {
            let t = self.n as f64 / FS;
            self.n += 1;
            let env = if t % 2.4 < 1.4 { (std::f64::consts::PI * (t % 2.4) / 1.4).sin().powi(2) } else { 0.0 };
            let f0 = 120.0 + 30.0 * (t * 2.2).sin();
            self.phase += 2.0 * std::f64::consts::PI * f0 / FS;
            let voice: f64 = (1..=10).map(|h| (self.phase * h as f64).sin() / h as f64).sum();
            self.seed = self.seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (self.seed >> 8) as f64 / 8_388_608.0 - 1.0;
            *s = (0.22 * env * voice + 0.012 * noise) as f32;
        }
        Ok(())
    }
}

/// Raw float32 mono 48 kHz frames written into a child's stdin by a helper thread; if the child
/// cannot keep up frames are dropped rather than stalling the engine.
struct PipeOut {
    tx: SyncSender<Vec<u8>>,
    child: Child,
}

impl PipeOut {
    fn spawn(mut cmd: Command) -> io::Result<PipeOut> {
        cmd.stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::null());
        let mut child = spawn_child(&mut cmd)?;
        let mut stdin = child.stdin.take().ok_or_else(|| io::Error::other("no stdin"))?;
        let (tx, rx) = sync_channel::<Vec<u8>>(24);
        thread::spawn(move || {
            while let Ok(b) = rx.recv() {
                if stdin.write_all(&b).is_err() {
                    break;
                }
            }
        });
        Ok(PipeOut { tx, child })
    }

    fn send(&self, frame: &[f32]) {
        let mut b = Vec::with_capacity(frame.len() * 4);
        for s in frame {
            b.extend_from_slice(&s.to_le_bytes());
        }
        let _ = self.tx.try_send(b);
    }
}

impl Drop for PipeOut {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn pacat(device: Option<&str>, stream: &str, latency_ms: u32) -> Command {
    let mut c = Command::new("pacat");
    c.args(["--playback", "--format=float32le", "--rate=48000", "--channels=1", "--raw", "--client-name=seiren-control"]);
    c.arg(format!("--latency-msec={latency_ms}"));
    c.arg(format!("--stream-name={stream}"));
    if let Some(d) = device {
        c.arg(format!("--device={d}"));
    }
    c
}

/// `pactl <args>`: its trimmed stdout when it succeeded.
fn pactl_run(args: &[&str]) -> Option<String> {
    Command::new("pactl")
        .args(args)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

/// "Server Name" from `pactl info` (e.g. "PulseAudio (on PipeWire 1.4.2)"), None if no server answers.
pub fn server_name() -> Option<String> {
    pactl_run(&["info"])?.lines().find_map(|l| l.strip_prefix("Server Name:").map(|s| s.trim().to_string()))
}

/// How the virtual microphone exists: a `pw-loopback` process (PipeWire), or two PulseAudio modules
/// (PulseAudio itself, or PipeWire's pulse layer on a system without `pw-loopback`).
enum Backing {
    PwLoopback(Child),
    PulseModules(Vec<String>),
    /// left by an earlier run; not ours to remove
    Existing,
}

impl Backing {
    fn start() -> Result<Backing, String> {
        if pactl_names("sinks").iter().any(|n| n == NAME_VIRTUAL_IN) {
            return Ok(Backing::Existing);
        }
        let forced_pulse = std::env::var("SEIREN_VIRTUAL_MIC").map(|v| v == "pulse").unwrap_or(false);
        if !forced_pulse && crate::sys::which("pw-loopback").is_some() {
            return Self::pw_loopback();
        }
        Self::pulse_modules()
    }

    fn pw_loopback() -> Result<Backing, String> {
        let mut cmd = Command::new("pw-loopback");
        cmd.args([
            "-m",
            "[ FL ]",
            &format!("--capture-props=media.class=Audio/Sink node.name={NAME_VIRTUAL_IN} node.description=\"Seiren Clean (input)\""),
            &format!("--playback-props=media.class=Audio/Source node.name={NAME_VIRTUAL_SRC} node.description=\"Seiren Clean\""),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
        let mut child = spawn_child(&mut cmd).map_err(|e| format!("pw-loopback не запустился: {e}"))?;
        let t0 = Instant::now();
        while !pactl_names("sinks").iter().any(|n| n == NAME_VIRTUAL_IN) {
            if t0.elapsed() > Duration::from_secs(4) {
                let _ = child.kill();
                let _ = child.wait();
                return Err("виртуальный микрофон не появился в PipeWire".into());
            }
            thread::sleep(Duration::from_millis(100));
        }
        Ok(Backing::PwLoopback(child))
    }

    fn pulse_modules() -> Result<Backing, String> {
        if crate::sys::which("pactl").is_none() {
            return Err("для виртуального микрофона нужен pw-loopback (PipeWire) или pactl (PulseAudio)".into());
        }
        let sink = pactl_run(&[
            "load-module",
            "module-null-sink",
            &format!("sink_name={NAME_VIRTUAL_IN}"),
            "sink_properties=device.description='Seiren Clean (input)'",
            "channels=1",
            "channel_map=mono",
            "rate=48000",
        ])
        .ok_or("PulseAudio не загрузил module-null-sink")?;
        let src = pactl_run(&[
            "load-module",
            "module-remap-source",
            &format!("master={NAME_VIRTUAL_IN}.monitor"),
            &format!("source_name={NAME_VIRTUAL_SRC}"),
            "source_properties=device.description='Seiren Clean'",
            "channels=1",
            "channel_map=mono",
        ]);
        match src {
            Some(src) => Ok(Backing::PulseModules(vec![src, sink])),
            None => {
                let _ = pactl_run(&["unload-module", &sink]);
                Err("PulseAudio не загрузил module-remap-source".into())
            }
        }
    }
}

impl Drop for Backing {
    fn drop(&mut self) {
        match self {
            Backing::PwLoopback(c) => {
                let _ = c.kill();
                let _ = c.wait();
            }
            // the source first, then the sink it listens to
            Backing::PulseModules(ids) => {
                for id in ids.iter() {
                    let _ = pactl_run(&["unload-module", id]);
                }
            }
            Backing::Existing => {}
        }
    }
}

/// "Seiren Clean": a virtual sink/source pair. We play the processed sound into the sink and any
/// program can record the source as an ordinary microphone.
struct VirtualMic {
    // fields drop in this order: stop the player first, then take the virtual device away
    out: PipeOut,
    #[allow(dead_code)] // held for its Drop
    backing: Backing,
}

impl VirtualMic {
    fn start() -> Result<VirtualMic, String> {
        let backing = Backing::start()?;
        let out = PipeOut::spawn(pacat(Some(NAME_VIRTUAL_IN), "Seiren Clean", 40)).map_err(|e| format!("pacat не запустился: {e}"))?;
        Ok(VirtualMic { out, backing })
    }
}

/// Play a file for listening back (ffplay, or paplay for WAV).
pub fn play(path: &std::path::Path) -> io::Result<Child> {
    let mut c = Command::new("ffplay");
    c.args(["-nodisp", "-autoexit", "-loglevel", "quiet"]).arg(path).stdin(Stdio::null());
    match spawn_child(&mut c) {
        Ok(ch) => Ok(ch),
        Err(_) => {
            let mut p = Command::new("paplay");
            p.arg(path).stdin(Stdio::null());
            spawn_child(&mut p)
        }
    }
}

// ================================================================================================
// The engine thread
// ================================================================================================

struct Eng {
    kind: SourceKind,
    chain: Chain,
    analyzer: Analyzer,
    emit: EngEmit,
    snap: Arc<Mutex<Snapshot>>,
    want_run: bool,
    recorder: Option<Recorder>,
    monitor: Option<PipeOut>,
    virt: Option<VirtualMic>,
    learn_frames: Option<usize>,
    source: Option<Box<dyn FrameSource>>,
    retry_at: Instant,
    /// An error was shown for the current fault; shown once, cleared when sound flows again.
    problem_shown: bool,
    frame_no: u64,
    meters: Snapshot,
    inbuf: Vec<f32>,
    outbuf: Vec<f32>,
}

impl Eng {
    fn new(kind: SourceKind, params: DspParams, emit: EngEmit, snap: Arc<Mutex<Snapshot>>) -> Eng {
        Eng {
            kind,
            chain: Chain::new(params),
            analyzer: Analyzer::new(),
            emit,
            snap,
            want_run: false,
            recorder: None,
            monitor: None,
            virt: None,
            learn_frames: None,
            source: None,
            retry_at: Instant::now(),
            problem_shown: false,
            frame_no: 0,
            meters: Snapshot::default(),
            inbuf: vec![0.0; HOP],
            outbuf: vec![0.0; HOP],
        }
    }

    fn wanted(&self) -> bool {
        self.want_run || self.recorder.is_some() || self.monitor.is_some() || self.virt.is_some() || self.learn_frames.is_some()
    }

    fn info(&self, t: impl Into<String>) {
        (self.emit)(EngEvent::Info(t.into()));
    }

    fn error(&self, t: impl Into<String>) {
        (self.emit)(EngEvent::Error(t.into()));
    }

    /// Returns true on Quit.
    fn handle(&mut self, cmd: EngCmd) -> bool {
        match cmd {
            EngCmd::SetParams(p) => self.chain.set_params(*p),
            EngCmd::Run(on) => self.want_run = on,
            EngCmd::Record(opts) => {
                if self.recorder.is_none() {
                    let emit = self.emit.clone();
                    match Recorder::start(opts, Arc::new(move |r| match r {
                        Ok(t) => emit(EngEvent::Take(t)),
                        Err(e) => emit(EngEvent::Error(e)),
                    })) {
                        Ok(r) => self.recorder = Some(r),
                        Err(e) => self.error(e),
                    }
                }
            }
            EngCmd::StopRecord => {
                if let Some(r) = self.recorder.take() {
                    let _ = r.stop();
                }
            }
            EngCmd::Monitor(on) => {
                if on && self.monitor.is_none() {
                    match PipeOut::spawn(pacat(find_sink().as_deref(), "Seiren monitor", 40)) {
                        Ok(p) => self.monitor = Some(p),
                        Err(e) => {
                            self.error(format!("не удалось включить прослушивание: {e}"));
                            (self.emit)(EngEvent::FeatureOff("monitor"));
                        }
                    }
                } else if !on {
                    self.monitor = None;
                }
            }
            EngCmd::VirtualMic(on) => {
                if on && self.virt.is_none() {
                    match VirtualMic::start() {
                        Ok(v) => {
                            self.virt = Some(v);
                            self.info("Виртуальный микрофон «Seiren Clean» создан: выберите его в Discord, OBS, браузере.");
                        }
                        Err(e) => {
                            self.error(e);
                            (self.emit)(EngEvent::FeatureOff("virtual_mic"));
                        }
                    }
                } else if !on {
                    self.virt = None;
                }
            }
            EngCmd::LearnNoise => {
                self.chain.start_learning();
                self.learn_frames = Some(0);
            }
            EngCmd::Quit => {
                if let Some(r) = self.recorder.take() {
                    let _ = r.stop().join();
                }
                self.monitor = None;
                self.virt = None;
                self.source = None;
                return true;
            }
        }
        false
    }

    fn open_source(&mut self) {
        let r: io::Result<Box<dyn FrameSource>> = match &self.kind {
            SourceKind::Pulse => Parec::open().map(|p| Box::new(p) as Box<dyn FrameSource>),
            SourceKind::Synthetic { realtime } => {
                Ok(Box::new(Synth { n: 0, phase: 0.0, seed: 1, next: Instant::now(), realtime: *realtime }))
            }
        };
        match r {
            Ok(s) => self.source = Some(s),
            Err(e) => {
                if !self.problem_shown {
                    self.problem_shown = true;
                    self.error(format!("не удалось открыть микрофон: {e}"));
                }
                self.retry_at = Instant::now() + Duration::from_secs(2);
            }
        }
    }

    fn publish(&mut self) {
        self.meters.recording = self.recorder.is_some();
        self.meters.rec_secs = self.recorder.as_ref().map_or(0.0, Recorder::elapsed);
        self.meters.monitor = self.monitor.is_some();
        self.meters.virtual_mic = self.virt.is_some();
        self.meters.learning = self.learn_frames.map(|n| n as f32 / LEARN_FRAMES as f32);
        self.meters.dfn_loading = self.chain.params().denoise == crate::dsp::DenoiseMode::Dfn && !self.chain.den.dfn_ready();
        if let Ok(mut s) = self.snap.lock() {
            *s = self.meters.clone();
        }
    }

    fn take_meters(&mut self, m: &FrameMeters) {
        let s = &mut self.meters;
        s.running = true;
        s.in_peak = m.in_peak.max(s.in_peak * 0.94);
        s.out_peak = m.out_peak.max(s.out_peak * 0.94);
        s.in_rms += (m.in_rms - s.in_rms) * 0.35;
        s.out_rms += (m.out_rms - s.out_rms) * 0.35;
        s.gate_open = m.gate_open;
        s.comp_gr_db = m.comp_gr_db.max(s.comp_gr_db * 0.9);
        s.limit_gr_db = m.limit_gr_db.max(s.limit_gr_db * 0.9);
        s.mouth_gr_db = m.mouth_gr_db.max(s.mouth_gr_db * 0.9);
        s.lufs = m.momentary_lufs;
        s.spectrum = self.analyzer.bands;
    }

    fn frame(&mut self) {
        let m = self.chain.process(&self.inbuf, &mut self.outbuf);
        self.analyzer.push(&self.outbuf);
        if self.frame_no % 2 == 0 {
            self.analyzer.update();
        }
        self.frame_no += 1;
        self.take_meters(&m);
        if let Some(r) = self.recorder.as_mut() {
            r.push(&self.outbuf, &self.inbuf);
        }
        if let Some(p) = &self.monitor {
            p.send(&self.outbuf);
        }
        if let Some(v) = &self.virt {
            v.out.send(&self.outbuf);
        }
        if let Some(n) = self.learn_frames.as_mut() {
            *n += 1;
            if *n >= LEARN_FRAMES {
                self.learn_frames = None;
                let profile = self.chain.finish_learning();
                if profile.is_empty() {
                    self.error("не удалось запомнить шум");
                } else {
                    (self.emit)(EngEvent::Profile(profile));
                }
            }
        }
    }

    fn run(&mut self, rx: Receiver<EngCmd>) {
        loop {
            // commands: poll while capturing, wait while idle
            loop {
                let c = if self.source.is_some() {
                    match rx.try_recv() {
                        Ok(c) => Some(c),
                        Err(TryRecvError::Empty) => None,
                        Err(TryRecvError::Disconnected) => Some(EngCmd::Quit),
                    }
                } else {
                    match rx.recv_timeout(Duration::from_millis(100)) {
                        Ok(c) => Some(c),
                        Err(RecvTimeoutError::Timeout) => None,
                        Err(RecvTimeoutError::Disconnected) => Some(EngCmd::Quit),
                    }
                };
                match c {
                    Some(c) => {
                        if self.handle(c) {
                            self.publish();
                            return;
                        }
                    }
                    None => break,
                }
            }
            let wanted = self.wanted();
            if wanted && self.source.is_none() && Instant::now() >= self.retry_at {
                self.open_source();
            }
            if !wanted && self.source.is_some() {
                self.source = None;
                self.meters = Snapshot { monitor: false, ..Snapshot::default() };
            }
            if let Some(src) = self.source.as_mut() {
                match src.read(&mut self.inbuf) {
                    Ok(()) => {
                        self.problem_shown = false;
                        self.frame();
                    }
                    Err(_) => {
                        self.source = None;
                        self.meters.running = false;
                        self.retry_at = Instant::now() + Duration::from_secs(1);
                        if wanted && !self.problem_shown {
                            self.problem_shown = true;
                            self.error("захват с микрофона прервался, пробую снова");
                        }
                    }
                }
            }
            self.publish();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::BINS;

    type Events = Arc<Mutex<Vec<String>>>;

    fn start(kind: SourceKind, params: DspParams) -> (Engine, Events, Arc<Mutex<Vec<Take>>>, Arc<Mutex<Vec<Vec<f32>>>>) {
        let events: Events = Arc::new(Mutex::new(Vec::new()));
        let takes = Arc::new(Mutex::new(Vec::new()));
        let profiles = Arc::new(Mutex::new(Vec::new()));
        let (e2, t2, p2) = (events.clone(), takes.clone(), profiles.clone());
        let emit: EngEmit = Arc::new(move |ev| match ev {
            EngEvent::Info(s) => e2.lock().unwrap().push(format!("info: {s}")),
            EngEvent::Error(s) => e2.lock().unwrap().push(format!("error: {s}")),
            EngEvent::Take(t) => t2.lock().unwrap().push(t),
            EngEvent::Profile(p) => p2.lock().unwrap().push(p),
            EngEvent::FeatureOff(f) => e2.lock().unwrap().push(format!("off: {f}")),
        });
        (Engine::spawn(kind, params, emit), events, takes, profiles)
    }

    fn wait_for(what: &str, mut ok: impl FnMut() -> bool) {
        for _ in 0..400 {
            if ok() {
                return;
            }
            thread::sleep(Duration::from_millis(25));
        }
        panic!("timed out waiting for {what}");
    }

    fn fast() -> SourceKind {
        SourceKind::Synthetic { realtime: false }
    }

    #[test]
    fn runs_publishes_meters_and_a_spectrum_and_stops_when_not_wanted() {
        let (e, events, _, _) = start(fast(), DspParams::neutral());
        assert!(!e.snapshot().running, "idle until asked");
        e.send(EngCmd::Run(true));
        wait_for("meters", || e.snapshot().in_rms > 0.001);
        let s = e.snapshot();
        assert!(s.running && s.spectrum.iter().any(|&v| v > 0.1) && s.out_peak > 0.0);
        e.send(EngCmd::Run(false));
        wait_for("stop", || !e.snapshot().running);
        assert!(events.lock().unwrap().iter().all(|l| !l.starts_with("error")), "{:?}", events.lock().unwrap());
        e.quit();
    }

    #[test]
    fn parameter_changes_reach_the_running_chain() {
        let (e, _, _, _) = start(fast(), DspParams::neutral());
        e.send(EngCmd::Run(true));
        wait_for("level", || e.snapshot().out_rms > 0.005);
        let before = e.snapshot().out_rms;
        e.send(EngCmd::SetParams(Box::new(DspParams { out_gain_db: -20.0, ..DspParams::neutral() })));
        wait_for("quieter", || e.snapshot().out_rms < before * 0.4 || e.snapshot().out_rms < 0.002);
        e.quit();
    }

    #[test]
    fn records_a_take_of_the_processed_sound_and_reports_it() {
        let dir = std::env::temp_dir().join(format!("seiren-engine-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (e, events, takes, _) = start(fast(), DspParams::default());
        e.send(EngCmd::Record(RecordOpts { dir: dir.clone(), format: crate::recorder::Format::Wav, keep_raw: true, normalize: false }));
        wait_for("2 s recorded", || e.snapshot().rec_secs >= 2.0);
        assert!(e.snapshot().recording);
        e.send(EngCmd::StopRecord);
        wait_for("the take", || !takes.lock().unwrap().is_empty());
        let t = takes.lock().unwrap()[0].clone();
        assert!(t.secs >= 2.0 && t.secs < 30.0, "{}", t.secs);
        assert!(t.path.exists() && t.raw_path.as_ref().is_some_and(|p| p.exists()));
        let r = hound::WavReader::open(&t.path).unwrap();
        assert!((r.len() as f32 / FS as f32 - t.secs).abs() < 0.02);
        assert!(events.lock().unwrap().iter().all(|l| !l.starts_with("error")), "{:?}", events.lock().unwrap());
        e.quit();
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn learns_a_noise_profile_from_the_room() {
        let (e, _, _, profiles) = start(fast(), DspParams::neutral());
        e.send(EngCmd::LearnNoise);
        wait_for("profile", || !profiles.lock().unwrap().is_empty());
        assert_eq!(profiles.lock().unwrap()[0].len(), BINS);
        e.quit();
    }

    /// Needs the real microphone and PipeWire: `cargo test --release -- --ignored real_microphone --nocapture`
    #[test]
    #[ignore]
    fn real_microphone_capture_flows_through_the_chain() {
        let (e, events, _, _) = start(SourceKind::Pulse, DspParams::default());
        e.send(EngCmd::Run(true));
        wait_for("audio", || e.snapshot().running && e.snapshot().in_rms > 0.0);
        thread::sleep(Duration::from_secs(3));
        let s = e.snapshot();
        println!("room: in rms {:.1} dBFS peak {:.1} dBFS | out rms {:.1} dBFS | momentary {:.1} LUFS", crate::dsp::lin_to_db(s.in_rms), crate::dsp::lin_to_db(s.in_peak), crate::dsp::lin_to_db(s.out_rms), s.lufs);
        assert!(s.in_rms > 1e-5, "silent capture");
        assert!(events.lock().unwrap().iter().all(|l| !l.starts_with("error")), "{:?}", events.lock().unwrap());
        e.quit();
    }

    /// Records three seconds of the real microphone into a temp folder and checks the file; the
    /// take is deleted again. `cargo test --release -- --ignored real_microphone_take --nocapture`
    #[test]
    #[ignore]
    fn real_microphone_take_is_written_to_disk() {
        let dir = std::env::temp_dir().join(format!("seiren-real-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (e, events, takes, _) = start(SourceKind::Pulse, DspParams::neutral());
        e.send(EngCmd::Record(RecordOpts { dir: dir.clone(), format: crate::recorder::Format::Wav, keep_raw: true, normalize: false }));
        wait_for("3 s recorded", || e.snapshot().rec_secs >= 3.0);
        e.send(EngCmd::StopRecord);
        wait_for("the take", || !takes.lock().unwrap().is_empty());
        let t = takes.lock().unwrap()[0].clone();
        let raw = t.raw_path.clone().expect("raw copy");
        let r = hound::WavReader::open(&raw).unwrap();
        let samples: Vec<f32> = r.into_samples::<i32>().map(|s| s.unwrap() as f32 / 8_388_608.0).collect();
        let rms = (samples.iter().map(|v| v * v).sum::<f32>() / samples.len() as f32).sqrt();
        println!("real take: {:.2} s, raw rms {:.1} dBFS", samples.len() as f32 / FS as f32, crate::dsp::lin_to_db(rms));
        assert!((t.secs - 3.0).abs() < 0.6, "{}", t.secs);
        assert!(rms > 1e-5, "the recorded file is silent");
        assert!(events.lock().unwrap().iter().all(|l| !l.starts_with("error")), "{:?}", events.lock().unwrap());
        e.quit();
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Creates and removes the virtual microphone: `cargo test --release -- --ignored virtual_mic --nocapture`
    #[test]
    #[ignore]
    fn virtual_mic_carries_the_processed_sound_to_other_programs() {
        let (e, events, _, _) = start(SourceKind::Synthetic { realtime: true }, DspParams::neutral());
        e.send(EngCmd::VirtualMic(true));
        wait_for("the virtual source", || pactl_names("sources").iter().any(|n| n == NAME_VIRTUAL_SRC));
        thread::sleep(Duration::from_millis(500));
        // listen to it the way any other program would
        let mut c = Command::new("parec");
        c.args([&format!("--device={NAME_VIRTUAL_SRC}"), "--format=float32le", "--rate=48000", "--channels=1", "--raw"]).stdout(Stdio::piped()).stderr(Stdio::null());
        let mut child = spawn_child(&mut c).unwrap();
        let mut out = child.stdout.take().unwrap();
        let mut got = Vec::new();
        let t0 = Instant::now();
        let mut buf = vec![0u8; 19200];
        while t0.elapsed() < Duration::from_secs(4) {
            if out.read_exact(&mut buf).is_err() {
                break;
            }
            got.extend(buf.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])));
        }
        let _ = child.kill();
        let _ = child.wait();
        let peak = got.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        println!("virtual mic: {:.1} s of audio, peak {:.1} dBFS", got.len() as f32 / FS as f32, crate::dsp::lin_to_db(peak));
        assert!(got.len() as f32 / FS as f32 > 2.0, "too little audio came through");
        assert!(peak > 0.05, "the virtual microphone is silent (peak {peak})");
        e.send(EngCmd::VirtualMic(false));
        e.quit();
        thread::sleep(Duration::from_millis(500));
        assert!(!pactl_names("sources").iter().any(|n| n == NAME_VIRTUAL_SRC), "virtual microphone left behind");
        assert!(events.lock().unwrap().iter().all(|l| !l.starts_with("error")), "{:?}", events.lock().unwrap());
    }
}
