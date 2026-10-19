//! The one thread that talks to the microphone. The window never blocks on the device: it sends
//! commands here, and results come back as events.
//!
//! Slider drags produce dozens of commands a second; the worker coalesces everything that piled up
//! while it was busy into the newest state. It also runs the "soft breathing" effect: the firmware's
//! own breathing has a fixed speed, so for a slow, adjustable breath the app moves the brightness
//! of a static colour itself, a few dozen writes a second, "now only" (nothing reaches flash).

use crate::mic::{Backend, Mic, MicError, STORE_FLASH, STORE_NONE};
use crate::settings::Settings;
use std::sync::mpsc::{channel, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConnState {
    Ok,
    Missing,
    NoAccess,
    Error,
}

#[derive(Clone, Debug)]
pub enum Event {
    Conn { state: ConnState, firmware: Option<String>, text: String },
    /// The brightness the microphone has stored (read once, on the first run).
    Brightness(u8),
    Info(String),
    Error(String),
}

pub type Emit = Arc<dyn Fn(Event) + Send + Sync>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Effect, colours, speed and brightness.
    Full,
    /// Only the brightness moved.
    Brightness,
}

pub enum Cmd {
    Connect,
    /// A live preview: storage "now only".
    Light(Settings, Kind),
    Mute(Settings),
    /// Send everything to the device; `STORE_FLASH` keeps it in the microphone's memory.
    Apply(Settings, u8),
    Quit,
}

/// Brightness byte of the soft breath at `t` seconds: a raised cosine, squared so the dim half
/// lingers the way a real breath does.
pub fn soft_level(t: f32, period: f32, max: u8) -> u8 {
    let phase = 0.5 - 0.5 * (std::f32::consts::TAU * t / period.max(0.1)).cos();
    let floor = 2.0_f32.min(max as f32);
    (floor + (max as f32 - floor) * phase * phase).round().clamp(0.0, 255.0) as u8
}

struct Soft {
    start: Instant,
    period: f32,
    max: u8,
    last: Option<u8>,
}

/// Effect, brightness and, when it had to move, the mute colour.
pub fn send_light(mic: &mut Mic, s: &Settings, storage: u8) -> Result<(), MicError> {
    let (effect, colors, reverse, speed) = s.firmware_effect();
    mic.set_effect(storage, effect, &colors, reverse, speed)?;
    mic.set_brightness(storage, s.brightness)?;
    let (c, moved) = s.muted_wire_color();
    if moved {
        mic.set_mute_color(storage, c)?;
    }
    Ok(())
}

pub fn send_mute(mic: &mut Mic, s: &Settings, storage: u8) -> Result<(), MicError> {
    mic.set_mute_color(storage, s.muted_wire_color().0)
}

struct Worker {
    mic: Mic,
    emit: Emit,
    soft: Option<Soft>,
    state: Option<ConnState>,
    probe: Instant,
    demo: bool,
    read_brightness: bool,
}

impl Worker {
    fn set_state(&mut self, state: ConnState, firmware: Option<String>, text: String) {
        if self.state != Some(state) || firmware.is_some() {
            self.state = Some(state);
            (self.emit)(Event::Conn { state, firmware, text });
        }
    }

    fn connect(&mut self) {
        self.probe = Instant::now();
        let r = self.mic.open().and_then(|_| self.mic.firmware());
        match r {
            Ok(fw) => {
                if self.read_brightness {
                    self.read_brightness = false;
                    if let Ok(b) = self.mic.brightness(STORE_FLASH) {
                        (self.emit)(Event::Brightness(b));
                    }
                }
                self.set_state(ConnState::Ok, Some(fw), String::new());
            }
            Err(e) => self.lost(e),
        }
    }

    /// The device went away or misbehaved.
    fn lost(&mut self, e: MicError) {
        self.mic.close();
        self.soft = None;
        let (state, text) = match &e {
            MicError::NotFound | MicError::Disconnected => (ConnState::Missing, e.to_string()),
            MicError::NoAccess(_) => (ConnState::NoAccess, e.to_string()),
            _ => (ConnState::Error, e.to_string()),
        };
        self.set_state(state, None, text);
    }

    /// A failed command: a device that refuses one command stays connected, anything else is a loss.
    fn failed(&mut self, e: MicError) {
        match e {
            MicError::Device(_) | MicError::Protocol(_) => (self.emit)(Event::Error(e.to_string())),
            _ => self.lost(e),
        }
    }

    fn start_soft(&mut self, s: &Settings) {
        self.soft = Some(Soft { start: Instant::now(), period: s.soft_period, max: s.brightness, last: None });
    }

    fn light(&mut self, s: &Settings, kind: Kind) {
        if !self.mic.is_open() {
            return;
        }
        // Moving only the brightness while soft breathing just changes the top of the breath.
        if kind == Kind::Brightness && s.effect == "soft" {
            if let Some(sf) = &mut self.soft {
                sf.max = s.brightness;
                sf.period = s.soft_period;
                return;
            }
        }
        let r = if kind == Kind::Brightness && s.effect != "soft" {
            self.mic.set_brightness(STORE_NONE, s.brightness)
        } else {
            send_light(&mut self.mic, s, STORE_NONE)
        };
        match r {
            Ok(()) => {
                if s.effect == "soft" {
                    self.start_soft(s);
                } else {
                    self.soft = None;
                }
            }
            Err(e) => self.failed(e),
        }
    }

    fn apply(&mut self, s: &Settings, storage: u8) {
        if !self.mic.is_open() {
            return;
        }
        let r = send_light(&mut self.mic, s, storage).and_then(|_| send_mute(&mut self.mic, s, storage));
        match r {
            Ok(()) => {
                if s.effect == "soft" {
                    self.start_soft(s);
                } else {
                    self.soft = None;
                }
                let text = match (storage, s.effect.as_str()) {
                    (STORE_FLASH, "soft") => "Сохранено в память микрофона: статичный цвет (плавное дыхание живёт только пока открыто окно).",
                    (STORE_FLASH, _) => "Сохранено в память микрофона.",
                    _ => "Применено до переподключения микрофона.",
                };
                (self.emit)(Event::Info(text.into()));
            }
            Err(e) => self.failed(e),
        }
    }

    fn mute(&mut self, s: &Settings) {
        if self.mic.is_open() {
            if let Err(e) = send_mute(&mut self.mic, s, STORE_NONE) {
                self.failed(e);
            }
        }
    }

    fn soft_tick(&mut self) {
        let Some(sf) = &mut self.soft else { return };
        let level = soft_level(sf.start.elapsed().as_secs_f32(), sf.period, sf.max);
        if sf.last == Some(level) {
            return;
        }
        sf.last = Some(level);
        if let Err(e) = self.mic.set_brightness(STORE_NONE, level) {
            self.failed(e);
        }
    }

    fn housekeeping(&mut self) {
        if self.probe.elapsed() < Duration::from_secs(3) {
            return;
        }
        self.probe = Instant::now();
        if !self.mic.is_open() {
            self.connect();
        } else if !self.demo && crate::mic::find_node().is_none() {
            self.lost(MicError::Disconnected);
        }
    }

    fn quit(&mut self) {
        // do not leave the ring stuck at the dark end of a breath
        if let Some(sf) = self.soft.take() {
            let _ = self.mic.set_brightness(STORE_NONE, sf.max);
        }
        self.mic.close();
    }
}

pub fn spawn(backend: Backend, emit: Emit, read_brightness: bool) -> (Sender<Cmd>, JoinHandle<()>) {
    let (tx, rx) = channel::<Cmd>();
    let demo = matches!(backend, Backend::Demo(_));
    let handle = thread::spawn(move || {
        let mut w = Worker {
            mic: Mic::new(backend),
            emit,
            soft: None,
            state: None,
            probe: Instant::now(),
            demo,
            read_brightness,
        };
        w.connect();
        loop {
            let wait = if w.soft.is_some() && w.mic.is_open() { Duration::from_millis(25) } else { Duration::from_millis(500) };
            match rx.recv_timeout(wait) {
                Ok(first) => {
                    let mut batch = vec![first];
                    while let Ok(next) = rx.try_recv() {
                        batch.push(next);
                    }
                    // Keep the newest of each live command; a full update covers a brightness-only one.
                    let mut light: Option<(Settings, Kind)> = None;
                    let mut mute: Option<Settings> = None;
                    let mut ordered: Vec<Cmd> = Vec::new();
                    let mut quit = false;
                    for c in batch {
                        match c {
                            Cmd::Light(s, k) => {
                                let kind = if k == Kind::Full || matches!(light, Some((_, Kind::Full))) { Kind::Full } else { Kind::Brightness };
                                light = Some((s, kind));
                            }
                            Cmd::Mute(s) => mute = Some(s),
                            Cmd::Quit => quit = true,
                            other => {
                                // an explicit command: flush what came before it, in order
                                if let Some((s, k)) = light.take() {
                                    ordered.push(Cmd::Light(s, k));
                                }
                                if let Some(s) = mute.take() {
                                    ordered.push(Cmd::Mute(s));
                                }
                                ordered.push(other);
                            }
                        }
                    }
                    if let Some((s, k)) = light {
                        ordered.push(Cmd::Light(s, k));
                    }
                    if let Some(s) = mute {
                        ordered.push(Cmd::Mute(s));
                    }
                    for c in ordered {
                        match c {
                            Cmd::Connect => w.connect(),
                            Cmd::Light(s, k) => w.light(&s, k),
                            Cmd::Mute(s) => w.mute(&s),
                            Cmd::Apply(s, st) => w.apply(&s, st),
                            Cmd::Quit => {}
                        }
                    }
                    if quit {
                        w.quit();
                        return;
                    }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    w.quit();
                    return;
                }
            }
            w.soft_tick();
            w.housekeeping();
        }
    });
    (tx, handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mic::DemoLog;
    use std::sync::Mutex;

    fn harness(read_brightness: bool) -> (Sender<Cmd>, JoinHandle<()>, DemoLog, Arc<Mutex<Vec<Event>>>) {
        let log = DemoLog::default();
        let events = Arc::new(Mutex::new(Vec::new()));
        let ev = events.clone();
        let emit: Emit = Arc::new(move |e| ev.lock().unwrap().push(e));
        let (tx, h) = spawn(Backend::Demo(log.clone()), emit, read_brightness);
        (tx, h, log, events)
    }

    fn settle(ms: u64) {
        thread::sleep(Duration::from_millis(ms));
    }

    #[test]
    fn soft_level_breathes_slowly_between_the_floor_and_the_top() {
        let (period, max) = (10.0, 200);
        assert_eq!(soft_level(0.0, period, max), 2);
        assert_eq!(soft_level(5.0, period, max), 200);
        assert_eq!(soft_level(10.0, period, max), 2);
        assert!(soft_level(2.5, period, max) < 120, "the dim half lingers");
        // slow: at a 10 s period the level moves by at most a few steps per 50 ms
        let mut worst = 0i32;
        for i in 0..200 {
            let a = soft_level(i as f32 * 0.05, period, max) as i32;
            let b = soft_level((i + 1) as f32 * 0.05, period, max) as i32;
            worst = worst.max((a - b).abs());
        }
        assert!(worst <= 9, "steps too coarse: {worst}");
    }

    #[test]
    fn connects_and_reports_the_firmware() {
        let (tx, h, _log, events) = harness(true);
        settle(150);
        tx.send(Cmd::Quit).unwrap();
        h.join().unwrap();
        let ev = events.lock().unwrap();
        assert!(ev.iter().any(|e| matches!(e, Event::Brightness(0x66))));
        assert!(ev.iter().any(|e| matches!(e, Event::Conn { state: ConnState::Ok, firmware: Some(f), .. } if f == "1.3")));
    }

    #[test]
    fn a_burst_of_slider_moves_is_coalesced_into_the_newest_state() {
        let (tx, h, log, _ev) = harness(false);
        settle(150);
        log.clear();
        for b in 0..40u8 {
            tx.send(Cmd::Light(Settings { brightness: b * 6, ..Default::default() }, Kind::Brightness)).unwrap();
        }
        tx.send(Cmd::Quit).unwrap();
        h.join().unwrap();
        let cmds = log.commands();
        let writes: Vec<_> = cmds.iter().filter(|c| (c.0, c.1) == (0x0F, 0x04)).collect();
        assert!(writes.len() < 40, "not coalesced: {} writes", writes.len());
        assert_eq!(writes.last().unwrap().2, vec![0, 0, 39 * 6], "the newest value wins");
    }

    #[test]
    fn soft_breathing_streams_brightness_now_only_and_never_touches_flash() {
        let (tx, h, log, _ev) = harness(false);
        settle(150);
        log.clear();
        let s = Settings { effect: "soft".into(), soft_period: 2.0, brightness: 200, ..Default::default() };
        tx.send(Cmd::Light(s, Kind::Full)).unwrap();
        settle(900);
        tx.send(Cmd::Quit).unwrap();
        h.join().unwrap();
        let cmds = log.commands();
        // first the static colour, then a stream of brightness writes, all with storage 0
        assert_eq!((cmds[0].0, cmds[0].1, cmds[0].2[2]), (0x0F, 0x02, 1), "static colour first");
        let bright: Vec<_> = cmds.iter().filter(|c| (c.0, c.1) == (0x0F, 0x04)).collect();
        assert!(bright.len() > 8, "expected a stream, got {}", bright.len());
        assert!(cmds.iter().all(|c| c.2[0] == 0), "everything is storage 'now only'");
        let levels: Vec<u8> = bright.iter().map(|c| c.2[2]).collect();
        assert!(levels.iter().max().unwrap() > &100 && levels.iter().min().unwrap() < &40, "{levels:?}");
        assert_eq!(*levels.last().unwrap(), 200, "quitting restores the top brightness");
    }

    /// Drives the real microphone (live values only, nothing is saved) and puts the look back:
    /// `cargo test --release -- --ignored hardware --nocapture`
    #[test]
    #[ignore]
    fn hardware_wave_speeds_and_soft_breathing_on_the_real_microphone() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let ev = events.clone();
        let emit: Emit = Arc::new(move |e| ev.lock().unwrap().push(e));
        let (tx, h) = spawn(Backend::Hid, emit, false);
        settle(600);
        let connected = events.lock().unwrap().iter().any(|e| matches!(e, Event::Conn { state: ConnState::Ok, .. }));
        assert!(connected, "microphone not reachable: {:?}", events.lock().unwrap());
        let (saved, _) = crate::settings::load();
        for speed in [0x28u8, 0xC8, 0xFF] {
            tx.send(Cmd::Light(Settings { effect: "wave".into(), wave_speed: speed, ..saved.clone() }, Kind::Full)).unwrap();
            settle(500);
        }
        tx.send(Cmd::Light(Settings { effect: "soft".into(), soft_period: 4.0, color1: "22d3ee".into(), brightness: 255, ..saved.clone() }, Kind::Full)).unwrap();
        settle(8000);
        tx.send(Cmd::Light(saved.clone(), Kind::Full)).unwrap();
        settle(400);
        tx.send(Cmd::Quit).unwrap();
        h.join().unwrap();
        let ev = events.lock().unwrap();
        let errors: Vec<_> = ev.iter().filter(|e| matches!(e, Event::Error(_))).collect();
        assert!(errors.is_empty(), "the microphone refused something: {errors:?}");
        assert!(!ev.iter().any(|e| matches!(e, Event::Conn { state, .. } if *state != ConnState::Ok)), "connection dropped: {ev:?}");
    }

    #[test]
    fn save_writes_flash_and_soft_is_saved_as_a_static_colour() {
        let (tx, h, log, events) = harness(false);
        settle(150);
        log.clear();
        let s = Settings { effect: "soft".into(), color1: "00ff40".into(), brightness: 180, ..Default::default() };
        tx.send(Cmd::Apply(s, STORE_FLASH)).unwrap();
        settle(300);
        tx.send(Cmd::Quit).unwrap();
        h.join().unwrap();
        let cmds = log.commands();
        let flash: Vec<_> = cmds.iter().filter(|c| c.2[0] == 1).collect();
        assert_eq!(flash.len(), 3, "effect, brightness and mute colour are saved: {cmds:?}");
        assert_eq!(flash[0].2[2], 1, "saved as static");
        assert!(events.lock().unwrap().iter().any(|e| matches!(e, Event::Info(t) if t.contains("статичный"))));
    }
}
