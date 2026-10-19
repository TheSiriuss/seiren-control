//! The recording page: wires the `Rec` global of the window to the audio engine.
//!
//! The sliders in the window are always 0..1; this file owns what that means (a table of
//! parameters with ranges, scales and units), keeps the window's models in step with the
//! parameters, and turns engine events and meters into what the window shows. Everything here
//! runs on the UI thread: engine events arrive through a queue that `tick` drains.

use crate::dsp::{self, DenoiseMode, DspParams, EQ_BANDS};
use crate::engine::{self, EngCmd, EngEmit, EngEvent, Engine, SourceKind};
use crate::recorder::{self, Format, RecordOpts, Take};
use crate::settings;
use crate::{AppWindow, FileRow, Param, Rec};
use serde::{Deserialize, Serialize};
use slint::{ComponentHandle, Model, ModelRc, VecModel};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Child;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ================================================================================================
// What is kept on disk
// ================================================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioSettings {
    pub params: DspParams,
    pub format: Format,
    pub keep_raw: bool,
    pub normalize: bool,
    /// Where takes go; empty = ~/Музыка/Seiren (the XDG music folder).
    pub dir: String,
}

impl Default for AudioSettings {
    fn default() -> Self {
        AudioSettings { params: DspParams::default(), format: Format::Wav, keep_raw: true, normalize: false, dir: String::new() }
    }
}

fn settings_path() -> PathBuf {
    settings::dir().join("audio.json")
}

pub fn load_settings() -> AudioSettings {
    let mut s: AudioSettings = std::fs::read_to_string(settings_path()).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    s.params.sanitize();
    s
}

fn save_settings(s: &AudioSettings) {
    let p = settings_path();
    if let Some(d) = p.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    if let Ok(t) = serde_json::to_string_pretty(s) {
        let _ = std::fs::write(p, t);
    }
}

/// ~/Музыка/Seiren (or wherever the XDG music folder is).
pub fn default_dir() -> PathBuf {
    let music = std::process::Command::new("xdg-user-dir")
        .arg("MUSIC")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Music")))
        .unwrap_or_else(|| PathBuf::from("."));
    music.join("Seiren")
}

impl AudioSettings {
    fn dir(&self) -> PathBuf {
        if self.dir.trim().is_empty() { default_dir() } else { PathBuf::from(&self.dir) }
    }
}

// ================================================================================================
// The parameter table
// ================================================================================================

#[derive(Clone, Copy)]
enum Unit {
    Db,
    SignedDb,
    Hz,
    Ms,
    Ratio,
    Percent,
}

struct Spec {
    key: &'static str,
    label: &'static str,
    min: f32,
    max: f32,
    log: bool,
    unit: Unit,
    get: fn(&DspParams) -> f32,
    set: fn(&mut DspParams, f32),
}

impl Spec {
    fn norm(&self, v: f32) -> f32 {
        let n = if self.log { (v.max(self.min) / self.min).ln() / (self.max / self.min).ln() } else { (v - self.min) / (self.max - self.min) };
        n.clamp(0.0, 1.0)
    }

    fn denorm(&self, n: f32) -> f32 {
        let n = n.clamp(0.0, 1.0);
        let v = if self.log { self.min * (self.max / self.min).powf(n) } else { self.min + (self.max - self.min) * n };
        self.round(v)
    }

    fn round(&self, v: f32) -> f32 {
        match self.unit {
            Unit::Db | Unit::SignedDb => (v * 2.0).round() / 2.0,
            Unit::Hz if v >= 1000.0 => (v / 50.0).round() * 50.0,
            Unit::Hz => (v / 5.0).round() * 5.0,
            Unit::Ms if v < 10.0 => (v * 2.0).round() / 2.0,
            Unit::Ms => v.round(),
            Unit::Ratio => (v * 10.0).round() / 10.0,
            Unit::Percent => (v * 100.0).round() / 100.0,
        }
    }

    fn text(&self, v: f32) -> String {
        match self.unit {
            Unit::Db => format!("{v:.1} дБ").replace("-", "−"),
            Unit::SignedDb => format!("{v:+.1} дБ").replace("-", "−"),
            Unit::Hz if v >= 1000.0 => format!("{:.2} кГц", v / 1000.0),
            Unit::Hz => format!("{v:.0} Гц"),
            Unit::Ms => format!("{v:.0} мс"),
            Unit::Ratio => format!("{v:.1} : 1"),
            Unit::Percent => format!("{:.0}%", v * 100.0),
        }
    }

    fn row(&self, p: &DspParams) -> Param {
        let v = (self.get)(p);
        Param { key: self.key.into(), label: self.label.into(), value: self.norm(v), text: self.text(v).into() }
    }
}

macro_rules! spec {
    ($key:literal, $label:literal, $min:expr, $max:expr, $log:expr, $unit:expr, $field:ident) => {
        Spec { key: $key, label: $label, min: $min, max: $max, log: $log, unit: $unit, get: |p| p.$field, set: |p, v| p.$field = v }
    };
}

static SPECS: &[Spec] = &[
    spec!("denoise_strength", "Сила подавления", 0.0, 1.0, false, Unit::Percent, denoise_strength),
    spec!("hpf_hz", "Частота среза", 20.0, 400.0, true, Unit::Hz, hpf_hz),
    spec!("mouth_amount", "Насколько глушить", 0.0, 1.0, false, Unit::Percent, mouth_amount),
    spec!("mouth_sens", "Чувствительность", 0.0, 1.0, false, Unit::Percent, mouth_sens),
    spec!("gate_threshold_db", "Порог", -80.0, -10.0, false, Unit::Db, gate_threshold_db),
    spec!("gate_range_db", "Насколько глушить", 3.0, 60.0, false, Unit::Db, gate_range_db),
    spec!("gate_release_ms", "Возврат", 20.0, 1000.0, true, Unit::Ms, gate_release_ms),
    spec!("deess_freq_hz", "Частота «с»", 3000.0, 10000.0, true, Unit::Hz, deess_freq_hz),
    spec!("deess_amount", "Сила", 0.0, 1.0, false, Unit::Percent, deess_amount),
    spec!("comp_threshold_db", "Порог", -60.0, 0.0, false, Unit::Db, comp_threshold_db),
    spec!("comp_ratio", "Степень сжатия", 1.0, 20.0, true, Unit::Ratio, comp_ratio),
    spec!("comp_attack_ms", "Атака", 0.5, 100.0, true, Unit::Ms, comp_attack_ms),
    spec!("comp_release_ms", "Спад", 20.0, 1500.0, true, Unit::Ms, comp_release_ms),
    spec!("comp_knee_db", "Мягкость колена", 0.0, 24.0, false, Unit::Db, comp_knee_db),
    spec!("limit_ceiling_db", "Потолок", -12.0, 0.0, false, Unit::Db, limit_ceiling_db),
    spec!("in_gain_db", "Усиление на входе", -24.0, 36.0, false, Unit::SignedDb, in_gain_db),
    spec!("out_gain_db", "Громкость на выходе", -24.0, 24.0, false, Unit::SignedDb, out_gain_db),
];

const SECTIONS: &[(&str, &[&str])] = &[
    ("denoise", &["denoise_strength"]),
    ("hpf", &["hpf_hz"]),
    ("mouth", &["mouth_amount", "mouth_sens"]),
    ("gate", &["gate_threshold_db", "gate_range_db", "gate_release_ms"]),
    ("deess", &["deess_freq_hz", "deess_amount"]),
    ("comp", &["comp_threshold_db", "comp_ratio", "comp_attack_ms", "comp_release_ms", "comp_knee_db"]),
    ("limit", &["limit_ceiling_db"]),
    ("levels", &["in_gain_db", "out_gain_db"]),
];

fn spec(key: &str) -> Option<&'static Spec> {
    SPECS.iter().find(|s| s.key == key)
}

fn rows(p: &DspParams, keys: &[&str]) -> Vec<Param> {
    keys.iter().filter_map(|k| spec(k)).map(|s| s.row(p)).collect()
}

// ================================================================================================
// Meters and paths
// ================================================================================================

fn level_norm(lin: f32) -> f32 {
    ((dsp::lin_to_db(lin) + 60.0) / 60.0).clamp(0.0, 1.0)
}

fn db_text(lin: f32) -> String {
    let d = dsp::lin_to_db(lin);
    if d < -90.0 { "−∞".to_string() } else { format!("{d:.1} дБ").replace('-', "−") }
}

fn clock(secs: f32) -> String {
    let s = secs as u32;
    if s >= 3600 { format!("{}:{:02}:{:02}", s / 3600, s % 3600 / 60, s % 60) } else { format!("{:02}:{:02}", s / 60, s % 60) }
}

/// The equaliser's response as a line (and the area under it) in a 520 x 130 box: 20 Hz..20 kHz
/// across, +-15 dB up and down.
fn eq_paths(gains: &[f32; EQ_BANDS]) -> (String, String) {
    const N: usize = 130;
    let freqs: Vec<f64> = (0..N).map(|i| 20.0 * (1000.0f64).powf(i as f64 / (N - 1) as f64)).collect();
    let db = dsp::eq_response_db(gains, &freqs);
    let mut line = String::new();
    for (i, d) in db.iter().enumerate() {
        let (x, y) = (i as f32 * 520.0 / (N - 1) as f32, 65.0 - d.clamp(-15.0, 15.0) / 15.0 * 58.0);
        line.push_str(&format!("{} {x:.1} {y:.1} ", if i == 0 { "M" } else { "L" }));
    }
    let fill = format!("{line}L 520 65 L 0 65 Z");
    (line, fill)
}

fn file_rows(dir: &Path) -> Vec<FileRow> {
    let mut v: Vec<(std::time::SystemTime, FileRow)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let ext_ok = Path::new(&name).extension().and_then(|x| x.to_str()).is_some_and(|x| matches!(x, "wav" | "flac" | "opus" | "mp3"));
            if !name.starts_with("Seiren ") || !ext_ok {
                return None;
            }
            let md = e.metadata().ok()?;
            Some((
                md.modified().ok()?,
                FileRow { name: name.clone().into(), info: format!("{:.1} МБ", md.len() as f64 / 1_048_576.0).into(), path: e.path().to_string_lossy().into_owned().into() },
            ))
        })
        .collect();
    v.sort_by(|a, b| b.0.cmp(&a.0));
    v.into_iter().take(6).map(|(_, r)| r).collect()
}

fn tilde(p: &Path) -> String {
    match std::env::var("HOME") {
        Ok(h) if p.starts_with(&h) => format!("~{}", p.strip_prefix(&h).unwrap().display()),
        _ => p.display().to_string(),
    }
}

// ================================================================================================
// The page
// ================================================================================================

pub struct AudioApp {
    weak: slint::Weak<AppWindow>,
    state: Arc<Mutex<AudioSettings>>,
    persist: bool,
    demo: bool,
    engine: RefCell<Option<Engine>>,
    events: Arc<Mutex<VecDeque<EngEvent>>>,
    models: RefCell<HashMap<&'static str, Rc<VecModel<Param>>>>,
    spectrum: Rc<VecModel<f32>>,
    formats: Vec<Format>,
    last_take: RefCell<Option<PathBuf>>,
    player: RefCell<Option<Child>>,
    dirty: Cell<Option<Instant>>,
    started: Instant,
}

impl AudioApp {
    pub fn new(ui: &AppWindow, demo: bool) -> Rc<AudioApp> {
        let state = if demo { AudioSettings::default() } else { load_settings() };
        Self::with_settings(ui, demo, state)
    }

    pub fn with_settings(ui: &AppWindow, demo: bool, state: AudioSettings) -> Rc<AudioApp> {
        let mut formats = vec![Format::Wav];
        if !demo {
            formats = recorder::available_formats();
        }
        let app = Rc::new(AudioApp {
            weak: ui.as_weak(),
            state: Arc::new(Mutex::new(state)),
            persist: !demo,
            demo,
            engine: RefCell::new(None),
            events: Arc::new(Mutex::new(VecDeque::new())),
            models: RefCell::new(HashMap::new()),
            spectrum: Rc::new(VecModel::from(vec![0.0f32; dsp::SPECTRUM_BANDS])),
            formats,
            last_take: RefCell::new(None),
            player: RefCell::new(None),
            dirty: Cell::new(None),
            started: Instant::now(),
        });
        let rec = ui.global::<Rec>();
        rec.set_spectrum(app.spectrum.clone().into());
        rec.set_formats(ModelRc::new(VecModel::from(app.formats.iter().map(|f| f.label().into()).collect::<Vec<slint::SharedString>>())));
        app.refresh(ui);
        app.wire(ui);
        app
    }

    fn engine<R>(&self, f: impl FnOnce(&Engine) -> R) -> R {
        let mut slot = self.engine.borrow_mut();
        let e = slot.get_or_insert_with(|| {
            let events = self.events.clone();
            let emit: EngEmit = Arc::new(move |ev| events.lock().unwrap().push_back(ev));
            let kind = if self.demo { SourceKind::Synthetic { realtime: true } } else { SourceKind::Pulse };
            Engine::spawn(kind, self.state.lock().unwrap().params.clone(), emit)
        });
        f(e)
    }

    fn send(&self, c: EngCmd) {
        self.engine(|e| e.send(c));
    }

    /// Edit the parameters, tell the engine, and remember to save.
    fn edit(&self, f: impl FnOnce(&mut DspParams)) -> DspParams {
        let p = {
            let mut s = self.state.lock().unwrap();
            f(&mut s.params);
            s.params.sanitize();
            s.params.clone()
        };
        self.send(EngCmd::SetParams(Box::new(p.clone())));
        self.touched();
        p
    }

    fn touched(&self) {
        if self.persist {
            self.dirty.set(Some(Instant::now()));
        }
    }

    pub fn set_page(&self, page: i32) {
        if page == 1 {
            self.send(EngCmd::Run(true));
        } else if self.engine.borrow().is_some() {
            self.send(EngCmd::Run(false));
        }
    }

    /// Fill every control of the page from the settings.
    fn refresh(&self, ui: &AppWindow) {
        let rec = ui.global::<Rec>();
        let s = self.state.lock().unwrap().clone();
        let p = &s.params;
        rec.set_preset(p.preset.clone().into());
        rec.set_denoise_mode(match p.denoise {
            DenoiseMode::Off => 0,
            DenoiseMode::Rnnoise => 1,
            DenoiseMode::Dfn => 2,
            DenoiseMode::Spectral => 3,
        });
        rec.set_hpf_on(p.hpf_on);
        rec.set_mouth_on(p.mouth_on);
        rec.set_eq_on(p.eq_on);
        rec.set_gate_on(p.gate_on);
        rec.set_deess_on(p.deess_on);
        rec.set_comp_on(p.comp_on);
        rec.set_limit_on(p.limit_on);
        let mut models = self.models.borrow_mut();
        for (name, keys) in SECTIONS {
            let m = Rc::new(VecModel::from(rows(p, keys)));
            models.insert(name, m.clone());
            let mr: ModelRc<Param> = m.into();
            match *name {
                "denoise" => rec.set_p_denoise(mr),
                "hpf" => rec.set_p_hpf(mr),
                "mouth" => rec.set_p_mouth(mr),
                "gate" => rec.set_p_gate(mr),
                "deess" => rec.set_p_deess(mr),
                "comp" => rec.set_p_comp(mr),
                "limit" => rec.set_p_limit(mr),
                _ => rec.set_p_levels(mr),
            }
        }
        rec.set_eq_gains(ModelRc::new(VecModel::from(p.eq_gains.to_vec())));
        let (line, fill) = eq_paths(&p.eq_gains);
        rec.set_eq_curve(line.into());
        rec.set_eq_fill(fill.into());
        rec.set_has_profile(!p.noise_profile.is_empty());
        rec.set_format(self.formats.iter().position(|f| *f == s.format).unwrap_or(0) as i32);
        rec.set_keep_raw(s.keep_raw);
        rec.set_normalize(s.normalize);
        let dir = s.dir();
        rec.set_folder(tilde(&dir).into());
        rec.set_files(ModelRc::new(VecModel::from(file_rows(&dir))));
    }

    fn wire(self: &Rc<Self>, ui: &AppWindow) {
        let rec = ui.global::<Rec>();

        let a = self.clone();
        rec.on_param(move |key, v| {
            let Some(sp) = spec(&key) else { return };
            let val = sp.denorm(v);
            let p = a.edit(|p| {
                (sp.set)(p, val);
                p.preset.clear();
            });
            let Some(ui) = a.weak.upgrade() else { return };
            let rec = ui.global::<Rec>();
            rec.set_preset("".into());
            for (name, keys) in SECTIONS {
                if let Some(i) = keys.iter().position(|k| *k == &*key) {
                    if let Some(m) = a.models.borrow().get(name) {
                        m.set_row_data(i, sp.row(&p));
                    }
                }
            }
        });

        let a = self.clone();
        rec.on_toggle(move |key, on| {
            a.edit(|p| {
                match &*key {
                    "hpf" => p.hpf_on = on,
                    "mouth" => p.mouth_on = on,
                    "eq" => p.eq_on = on,
                    "gate" => p.gate_on = on,
                    "deess" => p.deess_on = on,
                    "comp" => p.comp_on = on,
                    "limit" => p.limit_on = on,
                    _ => {}
                }
                p.preset.clear();
            });
            if let Some(ui) = a.weak.upgrade() {
                ui.global::<Rec>().set_preset("".into());
            }
        });

        let a = self.clone();
        rec.on_set_denoise(move |i| {
            a.edit(|p| {
                p.denoise = match i {
                    1 => DenoiseMode::Rnnoise,
                    2 => DenoiseMode::Dfn,
                    3 => DenoiseMode::Spectral,
                    _ => DenoiseMode::Off,
                };
                p.preset.clear();
            });
            if let Some(ui) = a.weak.upgrade() {
                let rec = ui.global::<Rec>();
                rec.set_denoise_mode(i);
                rec.set_preset("".into());
            }
        });

        let a = self.clone();
        rec.on_eq_band(move |i, db| {
            let p = a.edit(|p| {
                if let Some(g) = p.eq_gains.get_mut(i as usize) {
                    *g = db;
                }
                p.preset.clear();
            });
            if let Some(ui) = a.weak.upgrade() {
                let rec = ui.global::<Rec>();
                let (line, fill) = eq_paths(&p.eq_gains);
                rec.set_eq_curve(line.into());
                rec.set_eq_fill(fill.into());
                rec.set_preset("".into());
            }
        });

        let a = self.clone();
        rec.on_eq_reset(move || {
            a.edit(|p| {
                p.eq_gains = [0.0; EQ_BANDS];
                p.preset.clear();
            });
            if let Some(ui) = a.weak.upgrade() {
                a.refresh(&ui);
            }
        });

        let a = self.clone();
        rec.on_preset_selected(move |name| {
            a.edit(|p| {
                p.apply_preset(&name);
            });
            if let Some(ui) = a.weak.upgrade() {
                a.refresh(&ui);
            }
        });

        let a = self.clone();
        rec.on_learn_noise(move || a.send(EngCmd::LearnNoise));

        let a = self.clone();
        rec.on_forget_noise(move || {
            a.edit(|p| p.noise_profile.clear());
            if let Some(ui) = a.weak.upgrade() {
                ui.global::<Rec>().set_has_profile(false);
            }
        });

        let a = self.clone();
        rec.on_set_format(move |i| {
            if let Some(f) = a.formats.get(i as usize) {
                a.state.lock().unwrap().format = *f;
                a.touched();
            }
            if let Some(ui) = a.weak.upgrade() {
                ui.global::<Rec>().set_format(i);
            }
        });

        let a = self.clone();
        rec.on_option(move |key, on| match &*key {
            "keep_raw" => {
                a.state.lock().unwrap().keep_raw = on;
                a.touched();
            }
            "normalize" => {
                a.state.lock().unwrap().normalize = on;
                a.touched();
            }
            "monitor" => a.send(EngCmd::Monitor(on)),
            "virtual_mic" => a.send(EngCmd::VirtualMic(on)),
            _ => {}
        });

        let a = self.clone();
        rec.on_record(move || {
            let s = a.state.lock().unwrap().clone();
            a.send(EngCmd::Record(RecordOpts { dir: s.dir(), format: s.format, keep_raw: s.keep_raw, normalize: s.normalize }));
        });

        let a = self.clone();
        rec.on_stop(move || a.send(EngCmd::StopRecord));

        let a = self.clone();
        rec.on_play(move |path| {
            let path = if path.is_empty() { a.last_take.borrow().clone() } else { Some(PathBuf::from(path.as_str())) };
            let Some(path) = path else { return };
            a.stop_player();
            match engine::play(&path) {
                Ok(c) => {
                    *a.player.borrow_mut() = Some(c);
                    if let Some(ui) = a.weak.upgrade() {
                        ui.global::<Rec>().set_playing(true);
                    }
                }
                Err(e) => {
                    if let Some(ui) = a.weak.upgrade() {
                        crate::toast(&ui, &format!("не удалось проиграть: {e}"));
                    }
                }
            }
        });

        let a = self.clone();
        rec.on_stop_play(move || {
            a.stop_player();
            if let Some(ui) = a.weak.upgrade() {
                ui.global::<Rec>().set_playing(false);
            }
        });

        let a = self.clone();
        rec.on_open_folder(move || {
            let dir = a.state.lock().unwrap().dir();
            let _ = std::fs::create_dir_all(&dir);
            let _ = std::process::Command::new("xdg-open").arg(dir).stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).spawn();
        });
    }

    fn stop_player(&self) {
        if let Some(mut c) = self.player.borrow_mut().take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }

    fn handle(&self, ui: &AppWindow, ev: EngEvent) {
        let rec = ui.global::<Rec>();
        match ev {
            EngEvent::Info(t) | EngEvent::Error(t) => crate::toast(ui, &t),
            EngEvent::FeatureOff(f) => match f {
                "monitor" => rec.set_monitor(false),
                _ => rec.set_virtual_mic(false),
            },
            EngEvent::Profile(profile) => {
                self.edit(|p| p.noise_profile = profile);
                rec.set_has_profile(true);
                crate::toast(ui, "Шум комнаты запомнен. Теперь он вычитается из звука.");
            }
            EngEvent::Take(t) => self.show_take(ui, t),
        }
    }

    fn show_take(&self, ui: &AppWindow, t: Take) {
        let rec = ui.global::<Rec>();
        let name = t.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let mut info = clock(t.secs);
        if let Some(l) = t.lufs {
            info.push_str(&format!(" · {l:.1} LUFS"));
        }
        if t.raw_path.is_some() {
            info.push_str(" · оригинал рядом");
        }
        rec.set_last_file(name.into());
        rec.set_last_info(info.into());
        rec.set_wave(recorder::wave_path(&t.peaks, 360.0, 48.0).into());
        *self.last_take.borrow_mut() = Some(t.path.clone());
        let dir = self.state.lock().unwrap().dir();
        rec.set_files(ModelRc::new(VecModel::from(file_rows(&dir))));
        crate::toast(ui, &match &t.note {
            Some(n) => format!("Запись сохранена, но: {n}"),
            None => format!("Запись сохранена: {}", t.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()),
        });
    }

    /// Called every frame of the window's timer.
    pub fn tick(&self, ui: &AppWindow) {
        let events: Vec<EngEvent> = self.events.lock().unwrap().drain(..).collect();
        for ev in events {
            self.handle(ui, ev);
        }
        let finished = self.player.borrow_mut().as_mut().is_some_and(|c| !matches!(c.try_wait(), Ok(None)));
        if finished {
            *self.player.borrow_mut() = None;
            ui.global::<Rec>().set_playing(false);
        }
        if let Some(t) = self.dirty.get() {
            if t.elapsed() > Duration::from_millis(700) {
                self.dirty.set(None);
                save_settings(&self.state.lock().unwrap());
            }
        }
        if ui.get_page() == 1 && self.engine.borrow().is_some() {
            self.meters(ui);
        }
    }

    fn meters(&self, ui: &AppWindow) {
        let s = self.engine(|e| e.snapshot());
        let rec = ui.global::<Rec>();
        rec.set_running(s.running);
        rec.set_recording(s.recording);
        rec.set_rec_time(clock(s.rec_secs).into());
        let t = self.started.elapsed().as_secs_f32();
        rec.set_pulse(if s.recording { 0.5 + 0.5 * (t * 4.0).sin() } else { 0.0 });
        rec.set_in_level(level_norm(s.in_rms));
        rec.set_in_peak(level_norm(s.in_peak));
        rec.set_out_level(level_norm(s.out_rms));
        rec.set_out_peak(level_norm(s.out_peak));
        rec.set_in_db(db_text(s.in_rms).into());
        rec.set_out_db(db_text(s.out_rms).into());
        rec.set_lufs(if s.lufs > -69.0 { format!("{:.1} LUFS", s.lufs).replace('-', "−") } else { "—".to_string() }.into());
        rec.set_comp_gr(format!("{:.1}", s.comp_gr_db).into());
        rec.set_limit_gr(format!("{:.1}", s.limit_gr_db).into());
        rec.set_mouth_gr(format!("{:.0}", s.mouth_gr_db).into());
        rec.set_gate_open(s.gate_open);
        rec.set_dfn_loading(s.dfn_loading);
        rec.set_learning(s.learning.unwrap_or(-1.0));
        for (i, v) in s.spectrum.iter().enumerate() {
            if (self.spectrum.row_data(i).unwrap_or(0.0) - v).abs() > 0.004 {
                self.spectrum.set_row_data(i, *v);
            }
        }
    }

    /// Write pending settings and stop the engine (an unfinished take is finished first).
    pub fn shutdown(&self) {
        self.stop_player();
        if self.dirty.get().is_some() && self.persist {
            save_settings(&self.state.lock().unwrap());
        }
        if let Some(e) = self.engine.borrow_mut().take() {
            e.quit();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_parameter_round_trips_through_its_slider() {
        let p = DspParams::default();
        for sp in SPECS {
            let v = (sp.get)(&p);
            let back = sp.denorm(sp.norm(v));
            let tol = (sp.max - sp.min) * 0.02;
            assert!((back - v).abs() <= tol.max(0.6), "{}: {v} -> {back}", sp.key);
            assert!((0.0..=1.0).contains(&sp.norm(v)), "{}", sp.key);
            assert!(!sp.text(v).is_empty());
        }
    }

    #[test]
    fn slider_ends_are_the_range_ends() {
        let s = spec("comp_ratio").unwrap();
        assert_eq!((s.denorm(0.0), s.denorm(1.0)), (1.0, 20.0));
        let h = spec("hpf_hz").unwrap();
        assert_eq!((h.denorm(0.0), h.denorm(1.0)), (20.0, 400.0));
        // a log slider puts the middle at the geometric mean
        assert!((h.denorm(0.5) - (20.0f32 * 400.0).sqrt()).abs() < 5.0);
    }

    #[test]
    fn every_section_key_has_a_spec_and_every_spec_a_section() {
        for (_, keys) in SECTIONS {
            for k in *keys {
                assert!(spec(k).is_some(), "{k}");
            }
        }
        for sp in SPECS {
            assert!(SECTIONS.iter().any(|(_, ks)| ks.contains(&sp.key)), "{} is in no section", sp.key);
        }
    }

    #[test]
    fn formats_and_clock_and_curve() {
        assert_eq!(clock(5.4), "00:05");
        assert_eq!(clock(3725.0), "1:02:05");
        assert_eq!(spec("in_gain_db").unwrap().text(3.0), "+3.0 дБ");
        assert_eq!(spec("gate_threshold_db").unwrap().text(-50.0), "−50.0 дБ");
        assert_eq!(spec("hpf_hz").unwrap().text(1500.0), "1.50 кГц");
        let (line, fill) = eq_paths(&[0.0; EQ_BANDS]);
        assert!(line.starts_with("M 0.0 65.0") && fill.ends_with("Z"), "a flat EQ is a line through the middle");
        let (boost, _) = eq_paths(&[0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0]);
        assert_ne!(line, boost);
    }

    #[test]
    fn level_scale() {
        assert_eq!(level_norm(1.0), 1.0);
        assert_eq!(level_norm(0.0), 0.0);
        assert!((level_norm(dsp::db_to_lin(-30.0)) - 0.5).abs() < 0.01);
        assert_eq!(db_text(0.0), "−∞");
    }

    /// The real window (headless), the real callbacks, a demo engine: what the buttons do.
    #[test]
    fn the_window_drives_the_engine_end_to_end() {
        i_slint_backend_testing::init_no_event_loop();
        let ui = AppWindow::new().unwrap();
        let dir = std::env::temp_dir().join(format!("seiren-ui-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let settings = AudioSettings { dir: dir.to_string_lossy().into_owned(), keep_raw: false, ..AudioSettings::default() };
        let app = AudioApp::with_settings(&ui, true, settings);
        ui.set_page(1);
        app.set_page(1);
        let rec = ui.global::<Rec>();
        let pump = |what: &str, mut ok: Box<dyn FnMut() -> bool>| {
            for _ in 0..400 {
                app.tick(&ui);
                if ok() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            panic!("timed out waiting for {what}");
        };

        // a preset fills every control
        rec.invoke_preset_selected("stream".into());
        assert_eq!(rec.get_preset().as_str(), "stream");
        assert_eq!(rec.get_denoise_mode(), 2);
        assert_eq!(rec.get_eq_gains().row_data(0), Some(-4.0));
        assert!(rec.get_gate_on() && rec.get_comp_on());
        assert_eq!(rec.get_p_denoise().row_data(0).unwrap().text.as_str(), "85%");

        // a slider turns the preset into "custom" and shows the real value
        rec.invoke_param("comp_ratio".into(), 1.0);
        assert_eq!(rec.get_preset().as_str(), "");
        let row = (0..rec.get_p_comp().row_count()).filter_map(|i| rec.get_p_comp().row_data(i)).find(|p| p.key == "comp_ratio").unwrap();
        assert_eq!(row.text.as_str(), "20.0 : 1");
        assert_eq!(app.state.lock().unwrap().params.comp_ratio, 20.0);

        // the equaliser moves its curve, toggles and modes reach the settings
        let flat = rec.get_eq_curve();
        rec.invoke_eq_band(5, 12.0);
        assert_ne!(rec.get_eq_curve(), flat);
        assert_eq!(app.state.lock().unwrap().params.eq_gains[5], 12.0);
        rec.invoke_set_denoise(3);
        assert_eq!(rec.get_denoise_mode(), 3);
        assert_eq!(app.state.lock().unwrap().params.denoise, DenoiseMode::Spectral);
        rec.invoke_toggle("gate".into(), false);
        assert!(!app.state.lock().unwrap().params.gate_on);
        rec.invoke_eq_reset();
        assert_eq!(app.state.lock().unwrap().params.eq_gains, [0.0; EQ_BANDS]);

        // the meters come alive
        pump("meters", Box::new(|| rec.get_running() && rec.get_in_level() > 0.0));

        // record, stop, and the take shows up
        rec.invoke_record();
        pump("a second of recording", Box::new(|| rec.get_recording() && rec.get_rec_time().as_str() >= "00:01"));
        rec.invoke_stop();
        pump("the finished take", Box::new(|| !rec.get_last_file().is_empty()));
        assert!(rec.get_last_file().as_str().starts_with("Seiren ") && rec.get_last_file().as_str().ends_with(".wav"));
        assert!(!rec.get_wave().is_empty() && rec.get_files().row_count() == 1);
        assert!(dir.join(rec.get_last_file().as_str()).exists());
        assert!(!rec.get_recording());

        // learning the room's noise
        rec.invoke_learn_noise();
        pump("a noise profile", Box::new(|| rec.get_has_profile()));
        assert_eq!(app.state.lock().unwrap().params.noise_profile.len(), dsp::BINS);
        rec.invoke_forget_noise();
        assert!(!rec.get_has_profile());

        app.shutdown();
        let _ = std::fs::remove_dir_all(dir);
    }
}
