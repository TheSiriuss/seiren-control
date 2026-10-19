//! The voice chain, 48 kHz mono, 10 ms frames:
//!
//!   input gain -> high-pass -> noise suppression -> mouth noise -> gate -> 8-band EQ -> de-esser
//!              -> compressor -> look-ahead limiter -> output gain
//!
//! Mouth noise (clicks, smacks, chewing, saliva) is its own block, `mouth`.
//!
//! Noise suppression is one of three algorithms:
//!   * RNNoise, a small recurrent network (Xiph; pure-Rust port `nnnoiseless`, BSD-3)
//!   * DeepFilterNet3, a much stronger network that also models fine harmonic structure
//!     (Schröter et al.; MIT / Apache-2.0; vendored `libDF`, run with tract)
//!   * spectral Wiener filtering against a learned (or tracked) noise profile, for stationary
//!     hum and fans; written here
//!
//! Everything is real time: no allocation per frame, the heavy parts are built once.

use crate::mouth::{Mouth, MOUTH_LATENCY};
use ndarray::Array2;
use realfft::{num_complex::Complex, ComplexToReal, RealFftPlanner, RealToComplex};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::sync::Arc;

pub const FS: f64 = 48_000.0;
/// Samples per frame: 10 ms, the frame of both neural denoisers.
pub const HOP: usize = 480;
pub const EQ_BANDS: usize = 8;
/// Band 0 is a low shelf, band 7 a high shelf, the six between are peaking filters.
pub const EQ_FREQS: [f64; EQ_BANDS] = [80.0, 160.0, 315.0, 630.0, 1250.0, 2500.0, 5000.0, 10000.0];
const NFFT: usize = 2 * HOP;
/// Bins of the spectral denoiser (and of a learned noise profile).
pub const BINS: usize = NFFT / 2 + 1;
pub const SPECTRUM_BANDS: usize = 64;

pub fn db_to_lin(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

pub fn lin_to_db(x: f32) -> f32 {
    20.0 * x.max(1e-9).log10()
}

// ================================================================================================
// Parameters
// ================================================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DenoiseMode {
    Off,
    Rnnoise,
    #[default]
    Dfn,
    Spectral,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DspParams {
    pub in_gain_db: f32,
    pub hpf_on: bool,
    pub hpf_hz: f32,
    pub denoise: DenoiseMode,
    /// 0..1, the same slider for every algorithm (see `Denoiser`).
    pub denoise_strength: f32,
    /// Learned noise power per bin for the spectral mode; empty = track the noise floor.
    pub noise_profile: Vec<f32>,
    /// Lip smacks, clicks, chewing and wet mouth sounds (see `mouth`); costs 70 ms of delay.
    /// (A file saved before this existed loads with it off, whatever the default preset says.)
    #[serde(default)]
    pub mouth_on: bool,
    /// 0..1: how far such a sound is turned down (6..30 dB).
    pub mouth_amount: f32,
    /// 0..1: how readily a sound is taken for mouth noise.
    pub mouth_sens: f32,
    pub gate_on: bool,
    pub gate_threshold_db: f32,
    pub gate_range_db: f32,
    pub gate_release_ms: f32,
    pub eq_on: bool,
    pub eq_gains: [f32; EQ_BANDS],
    pub deess_on: bool,
    pub deess_freq_hz: f32,
    pub deess_amount: f32,
    pub comp_on: bool,
    pub comp_threshold_db: f32,
    pub comp_ratio: f32,
    pub comp_attack_ms: f32,
    pub comp_release_ms: f32,
    pub comp_knee_db: f32,
    pub comp_makeup_db: f32,
    pub comp_auto_makeup: bool,
    pub limit_on: bool,
    pub limit_ceiling_db: f32,
    pub out_gain_db: f32,
    pub preset: String,
}

impl Default for DspParams {
    fn default() -> Self {
        let mut p = DspParams::neutral();
        p.apply_preset("podcast");
        p
    }
}

impl DspParams {
    /// Everything off or flat.
    pub fn neutral() -> Self {
        DspParams {
            in_gain_db: 0.0,
            hpf_on: false,
            hpf_hz: 80.0,
            denoise: DenoiseMode::Off,
            denoise_strength: 0.7,
            noise_profile: Vec::new(),
            mouth_on: false,
            mouth_amount: 0.6,
            mouth_sens: 0.5,
            gate_on: false,
            gate_threshold_db: -50.0,
            gate_range_db: 18.0,
            gate_release_ms: 150.0,
            eq_on: false,
            eq_gains: [0.0; EQ_BANDS],
            deess_on: false,
            deess_freq_hz: 6500.0,
            deess_amount: 0.45,
            comp_on: false,
            comp_threshold_db: -22.0,
            comp_ratio: 3.0,
            comp_attack_ms: 8.0,
            comp_release_ms: 120.0,
            comp_knee_db: 6.0,
            comp_makeup_db: 0.0,
            comp_auto_makeup: true,
            limit_on: false,
            limit_ceiling_db: -1.0,
            out_gain_db: 0.0,
            preset: String::new(),
        }
    }

    /// Replace everything except the learned noise profile with a named preset. False if unknown.
    pub fn apply_preset(&mut self, name: &str) -> bool {
        let profile = std::mem::take(&mut self.noise_profile);
        let mut p = DspParams::neutral();
        p.limit_on = true;
        p.preset = name.to_string();
        match name {
            "podcast" => {
                p.hpf_on = true;
                p.denoise = DenoiseMode::Dfn;
                p.denoise_strength = 0.7;
                p.mouth_on = true;
                p.gate_on = true;
                p.gate_threshold_db = -52.0;
                p.gate_range_db = 14.0;
                p.eq_on = true;
                p.eq_gains = [-3.0, 1.0, -2.0, 0.0, 1.0, 2.5, 2.0, 2.0];
                p.deess_on = true;
                p.deess_freq_hz = 6800.0;
                p.deess_amount = 0.5;
                p.comp_on = true;
                p.comp_threshold_db = -24.0;
                p.comp_ratio = 3.0;
            }
            "stream" => {
                p.hpf_on = true;
                p.hpf_hz = 90.0;
                p.denoise = DenoiseMode::Dfn;
                p.denoise_strength = 0.85;
                p.mouth_on = true;
                p.gate_on = true;
                p.gate_threshold_db = -48.0;
                p.gate_range_db = 24.0;
                p.eq_on = true;
                p.eq_gains = [-4.0, 0.0, -3.0, -1.0, 1.0, 3.0, 1.5, 1.0];
                p.deess_on = true;
                p.deess_amount = 0.45;
                p.comp_on = true;
                p.comp_threshold_db = -22.0;
                p.comp_ratio = 3.5;
                p.comp_attack_ms = 6.0;
            }
            "vocal" => {
                p.hpf_on = true;
                p.hpf_hz = 70.0;
                p.mouth_on = true;
                p.mouth_amount = 0.5;
                p.eq_on = true;
                p.eq_gains = [-2.0, 1.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0];
                p.deess_on = true;
                p.deess_freq_hz = 7000.0;
                p.deess_amount = 0.4;
                p.comp_on = true;
                p.comp_threshold_db = -20.0;
                p.comp_ratio = 2.5;
                p.comp_attack_ms = 15.0;
                p.comp_release_ms = 160.0;
            }
            "calls" => {
                p.hpf_on = true;
                p.hpf_hz = 100.0;
                p.denoise = DenoiseMode::Dfn;
                p.denoise_strength = 1.0;
                p.gate_on = true;
                p.gate_threshold_db = -45.0;
                p.gate_range_db = 30.0;
                p.eq_on = true;
                p.eq_gains = [-6.0, -3.0, -2.0, 0.0, 2.0, 4.0, 2.0, -3.0];
                p.comp_on = true;
                p.comp_threshold_db = -26.0;
                p.comp_ratio = 4.0;
            }
            "clean" => {
                p.hpf_on = true;
                p.hpf_hz = 70.0;
                p.denoise = DenoiseMode::Rnnoise;
                p.denoise_strength = 0.6;
            }
            "off" => {
                p.limit_on = false;
            }
            _ => {
                self.noise_profile = profile;
                return false;
            }
        }
        *self = p;
        self.noise_profile = profile;
        true
    }

    /// Clamp everything to what the processing can take, so a hand-edited file cannot break it.
    pub fn sanitize(&mut self) {
        let c = |v: &mut f32, lo: f32, hi: f32, d: f32| {
            *v = if v.is_finite() { v.clamp(lo, hi) } else { d };
        };
        c(&mut self.in_gain_db, -24.0, 36.0, 0.0);
        c(&mut self.hpf_hz, 20.0, 400.0, 80.0);
        c(&mut self.denoise_strength, 0.0, 1.0, 0.7);
        c(&mut self.mouth_amount, 0.0, 1.0, 0.6);
        c(&mut self.mouth_sens, 0.0, 1.0, 0.5);
        c(&mut self.gate_threshold_db, -80.0, -10.0, -50.0);
        c(&mut self.gate_range_db, 3.0, 60.0, 18.0);
        c(&mut self.gate_release_ms, 20.0, 1000.0, 150.0);
        for g in &mut self.eq_gains {
            c(g, -15.0, 15.0, 0.0);
        }
        c(&mut self.deess_freq_hz, 3000.0, 10000.0, 6500.0);
        c(&mut self.deess_amount, 0.0, 1.0, 0.45);
        c(&mut self.comp_threshold_db, -60.0, 0.0, -22.0);
        c(&mut self.comp_ratio, 1.0, 20.0, 3.0);
        c(&mut self.comp_attack_ms, 0.5, 100.0, 8.0);
        c(&mut self.comp_release_ms, 20.0, 1500.0, 120.0);
        c(&mut self.comp_knee_db, 0.0, 24.0, 6.0);
        c(&mut self.comp_makeup_db, 0.0, 24.0, 0.0);
        c(&mut self.limit_ceiling_db, -12.0, 0.0, -1.0);
        c(&mut self.out_gain_db, -24.0, 24.0, 0.0);
        if self.noise_profile.len() != BINS || self.noise_profile.iter().any(|v| !v.is_finite() || *v < 0.0) {
            self.noise_profile.clear();
        }
    }
}

// ================================================================================================
// Biquads (RBJ cookbook), f64 inside because low corner frequencies at 48 kHz need it
// ================================================================================================

#[derive(Clone, Copy, Debug)]
pub struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    pub const IDENTITY: Biquad = Biquad { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0, z1: 0.0, z2: 0.0 };

    fn raw(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Biquad {
        Biquad { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0, z1: 0.0, z2: 0.0 }
    }

    /// New coefficients, same state (no click when a slider moves).
    pub fn retune(&mut self, o: &Biquad) {
        self.b0 = o.b0;
        self.b1 = o.b1;
        self.b2 = o.b2;
        self.a1 = o.a1;
        self.a2 = o.a2;
    }

    pub fn highpass(f: f64, q: f64) -> Biquad {
        let w = 2.0 * PI * f / FS;
        let (c, a) = (w.cos(), w.sin() / (2.0 * q));
        Biquad::raw((1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0, 1.0 + a, -2.0 * c, 1.0 - a)
    }

    pub fn lowpass(f: f64, q: f64) -> Biquad {
        let w = 2.0 * PI * f / FS;
        let (c, a) = (w.cos(), w.sin() / (2.0 * q));
        Biquad::raw((1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0, 1.0 + a, -2.0 * c, 1.0 - a)
    }

    pub fn bandpass(f: f64, q: f64) -> Biquad {
        let w = 2.0 * PI * f / FS;
        let (c, a) = (w.cos(), w.sin() / (2.0 * q));
        Biquad::raw(a, 0.0, -a, 1.0 + a, -2.0 * c, 1.0 - a)
    }

    pub fn peaking(f: f64, q: f64, gain_db: f64) -> Biquad {
        let big_a = 10f64.powf(gain_db / 40.0);
        let w = 2.0 * PI * f / FS;
        let (c, a) = (w.cos(), w.sin() / (2.0 * q));
        Biquad::raw(1.0 + a * big_a, -2.0 * c, 1.0 - a * big_a, 1.0 + a / big_a, -2.0 * c, 1.0 - a / big_a)
    }

    pub fn low_shelf(f: f64, gain_db: f64) -> Biquad {
        let big_a = 10f64.powf(gain_db / 40.0);
        let w = 2.0 * PI * f / FS;
        let (c, s) = (w.cos(), w.sin());
        let alpha = s / 2.0 * (2.0f64).sqrt(); // shelf slope 1
        let t = 2.0 * big_a.sqrt() * alpha;
        Biquad::raw(
            big_a * ((big_a + 1.0) - (big_a - 1.0) * c + t),
            2.0 * big_a * ((big_a - 1.0) - (big_a + 1.0) * c),
            big_a * ((big_a + 1.0) - (big_a - 1.0) * c - t),
            (big_a + 1.0) + (big_a - 1.0) * c + t,
            -2.0 * ((big_a - 1.0) + (big_a + 1.0) * c),
            (big_a + 1.0) + (big_a - 1.0) * c - t,
        )
    }

    pub fn high_shelf(f: f64, gain_db: f64) -> Biquad {
        let big_a = 10f64.powf(gain_db / 40.0);
        let w = 2.0 * PI * f / FS;
        let (c, s) = (w.cos(), w.sin());
        let alpha = s / 2.0 * (2.0f64).sqrt();
        let t = 2.0 * big_a.sqrt() * alpha;
        Biquad::raw(
            big_a * ((big_a + 1.0) + (big_a - 1.0) * c + t),
            -2.0 * big_a * ((big_a - 1.0) + (big_a + 1.0) * c),
            big_a * ((big_a + 1.0) + (big_a - 1.0) * c - t),
            (big_a + 1.0) - (big_a - 1.0) * c + t,
            2.0 * ((big_a - 1.0) - (big_a + 1.0) * c),
            (big_a + 1.0) - (big_a - 1.0) * c - t,
        )
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let x = x as f64;
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y as f32
    }

    /// |H(e^jw)| at `f` Hz.
    pub fn magnitude(&self, f: f64) -> f64 {
        let w = 2.0 * PI * f / FS;
        let (c1, s1, c2, s2) = (w.cos(), w.sin(), (2.0 * w).cos(), (2.0 * w).sin());
        let num = ((self.b0 + self.b1 * c1 + self.b2 * c2).powi(2) + (self.b1 * s1 + self.b2 * s2).powi(2)).sqrt();
        let den = ((1.0 + self.a1 * c1 + self.a2 * c2).powi(2) + (self.a1 * s1 + self.a2 * s2).powi(2)).sqrt();
        num / den
    }
}

fn eq_filter(band: usize, gain_db: f32) -> Biquad {
    let (f, g) = (EQ_FREQS[band], gain_db as f64);
    match band {
        0 => Biquad::low_shelf(f, g),
        b if b == EQ_BANDS - 1 => Biquad::high_shelf(f, g),
        _ => Biquad::peaking(f, 1.2, g),
    }
}

/// The EQ's response in dB at each of `freqs` (for the curve in the window).
pub fn eq_response_db(gains: &[f32; EQ_BANDS], freqs: &[f64]) -> Vec<f32> {
    let filters: Vec<Biquad> = (0..EQ_BANDS).map(|b| eq_filter(b, gains[b])).collect();
    freqs.iter().map(|&f| filters.iter().map(|b| 20.0 * b.magnitude(f).log10()).sum::<f64>() as f32).collect()
}

// ================================================================================================
// Dynamics
// ================================================================================================

fn coef(ms: f32) -> f32 {
    1.0 - (-1.0 / (ms.max(0.01) * 0.001 * FS as f32)).exp()
}

/// A downward expander: below the threshold the signal is pulled down by `range` dB, with hold
/// and a hysteresis of 3 dB so it does not chatter on breaths.
struct Gate {
    env: f32,
    gain: f32,
    hold: usize,
    open: bool,
    thr_open: f32,
    thr_close: f32,
    floor: f32,
    att: f32,
    rel: f32,
    env_rel: f32,
}

impl Gate {
    fn new() -> Gate {
        Gate { env: 0.0, gain: 1.0, hold: 0, open: true, thr_open: 0.0, thr_close: 0.0, floor: 1.0, att: coef(0.5), rel: coef(150.0), env_rel: 1.0 - coef(25.0) }
    }

    fn tune(&mut self, p: &DspParams) {
        self.thr_open = db_to_lin(p.gate_threshold_db);
        self.thr_close = db_to_lin(p.gate_threshold_db - 3.0);
        self.floor = db_to_lin(-p.gate_range_db);
        self.rel = coef(p.gate_release_ms);
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let a = x.abs();
        self.env = if a > self.env { a } else { self.env * self.env_rel };
        if self.env > self.thr_open {
            self.open = true;
            self.hold = (0.06 * FS) as usize;
        } else if self.env < self.thr_close {
            if self.hold > 0 {
                self.hold -= 1;
            } else {
                self.open = false;
            }
        }
        let target = if self.open { 1.0 } else { self.floor };
        let k = if target > self.gain { self.att } else { self.rel };
        self.gain += (target - self.gain) * k;
        x * self.gain
    }
}

/// Feed-forward compressor, level detector in the dB domain, soft knee.
struct Compressor {
    env_db: f32,
    thr: f32,
    ratio: f32,
    knee: f32,
    makeup: f32,
    att: f32,
    rel: f32,
    /// the deepest gain reduction (dB, positive) seen since the last `take_gr`
    max_gr: f32,
}

impl Compressor {
    fn new() -> Compressor {
        Compressor { env_db: -90.0, thr: -22.0, ratio: 3.0, knee: 6.0, makeup: 0.0, att: coef(8.0), rel: coef(120.0), max_gr: 0.0 }
    }

    fn tune(&mut self, p: &DspParams) {
        self.thr = p.comp_threshold_db;
        self.ratio = p.comp_ratio;
        self.knee = p.comp_knee_db;
        self.att = coef(p.comp_attack_ms);
        self.rel = coef(p.comp_release_ms);
        self.makeup = if p.comp_auto_makeup { (-0.5 * p.comp_threshold_db * (1.0 - 1.0 / p.comp_ratio)).max(0.0) } else { p.comp_makeup_db };
    }

    /// Gain reduction in dB (positive) for a detector level.
    fn reduction(&self, level_db: f32) -> f32 {
        let slope = 1.0 - 1.0 / self.ratio;
        let over = level_db - self.thr;
        if 2.0 * over < -self.knee {
            0.0
        } else if 2.0 * over.abs() <= self.knee {
            slope * (over + self.knee / 2.0).powi(2) / (2.0 * self.knee.max(1e-3))
        } else {
            slope * over
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let level = lin_to_db(x.abs());
        let k = if level > self.env_db { self.att } else { self.rel };
        self.env_db += (level - self.env_db) * k;
        let gr = self.reduction(self.env_db);
        if gr > self.max_gr {
            self.max_gr = gr;
        }
        x * db_to_lin(self.makeup - gr)
    }

    fn take_gr(&mut self) -> f32 {
        std::mem::take(&mut self.max_gr)
    }
}

const LOOKAHEAD: usize = 96; // 2 ms

/// Look-ahead peak limiter: the signal is delayed by `LOOKAHEAD - 1` samples; the gain is the
/// moving average of a release-smoothed sliding minimum of what each sample needs, which is never
/// above what the delayed sample needs, so nothing gets past the ceiling.
struct Limiter {
    delay: [f32; LOOKAHEAD],
    need: [f32; LOOKAHEAD],
    smooth: [f32; LOOKAHEAD],
    pos: usize,
    r: f32,
    sum: f32,
    ceiling: f32,
    rel: f32,
    max_gr: f32,
}

impl Limiter {
    fn new() -> Limiter {
        Limiter { delay: [0.0; LOOKAHEAD], need: [1.0; LOOKAHEAD], smooth: [1.0; LOOKAHEAD], pos: 0, r: 1.0, sum: LOOKAHEAD as f32, ceiling: 0.89, rel: coef(60.0), max_gr: 0.0 }
    }

    fn tune(&mut self, p: &DspParams) {
        self.ceiling = db_to_lin(p.limit_ceiling_db);
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let i = self.pos;
        self.need[i] = (self.ceiling / x.abs().max(1e-9)).min(1.0);
        let out_x = self.delay[i];
        self.delay[i] = x;
        let m = self.need.iter().fold(1.0f32, |a, &b| a.min(b));
        self.r = if m < self.r { m } else { self.r + (m - self.r) * self.rel };
        self.sum += self.r - self.smooth[i];
        self.smooth[i] = self.r;
        self.pos = (i + 1) % LOOKAHEAD;
        let g = (self.sum / LOOKAHEAD as f32).min(1.0);
        if g < 1.0 {
            self.max_gr = self.max_gr.max(-lin_to_db(g));
        }
        let y = out_x * g;
        y.clamp(-self.ceiling, self.ceiling)
    }

    fn take_gr(&mut self) -> f32 {
        std::mem::take(&mut self.max_gr)
    }
}

/// Split-band de-esser: a band-pass sidechain around the sibilance frequency; when it gets loud
/// that band alone is turned down (`out = x - (1 - g) * band(x)`), so the rest is untouched.
struct DeEsser {
    side: Biquad,
    band: Biquad,
    env_db: f32,
    thr: f32,
    att: f32,
    rel: f32,
}

impl DeEsser {
    fn new() -> DeEsser {
        DeEsser { side: Biquad::bandpass(6500.0, 1.5), band: Biquad::bandpass(6500.0, 1.5), env_db: -90.0, thr: -30.0, att: coef(0.3), rel: coef(25.0) }
    }

    fn tune(&mut self, p: &DspParams) {
        let b = Biquad::bandpass(p.deess_freq_hz as f64, 1.5);
        self.side.retune(&b);
        self.band.retune(&b);
        self.thr = -20.0 - 25.0 * p.deess_amount;
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let s = self.side.process(x);
        let band = self.band.process(x);
        let level = lin_to_db(s.abs());
        let k = if level > self.env_db { self.att } else { self.rel };
        self.env_db += (level - self.env_db) * k;
        let red = ((self.env_db - self.thr).max(0.0) * (1.0 - 1.0 / 6.0)).min(14.0);
        x - (1.0 - db_to_lin(-red)) * band
    }
}

// ================================================================================================
// Noise suppression
// ================================================================================================

/// Spectral Wiener filter over 50 %-overlapped sqrt-Hann frames (960 / 480), so a frame in is
/// a frame out with 10 ms of delay. The noise power comes from a learned profile, or, without one,
/// from a slowly rising minimum-statistics floor.
struct Spectral {
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
    win: Vec<f32>,
    prev_in: Vec<f32>,
    tail: Vec<f32>,
    frame: Vec<f32>,
    spec: Vec<Complex<f32>>,
    scratch_f: Vec<Complex<f32>>,
    scratch_i: Vec<Complex<f32>>,
    smooth_p: Vec<f32>,
    floor_p: Vec<f32>,
    prev_gain: Vec<f32>,
    prev_snr: Vec<f32>,
    gain: Vec<f32>,
    started: bool,
}

impl Spectral {
    fn new() -> Spectral {
        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(NFFT);
        let inv = planner.plan_fft_inverse(NFFT);
        let win: Vec<f32> = (0..NFFT).map(|i| ((PI * (i as f64 + 0.5) / NFFT as f64).sin()) as f32).collect();
        Spectral {
            scratch_f: fwd.make_scratch_vec(),
            scratch_i: inv.make_scratch_vec(),
            fwd,
            inv,
            win,
            prev_in: vec![0.0; HOP],
            tail: vec![0.0; HOP],
            frame: vec![0.0; NFFT],
            spec: vec![Complex::new(0.0, 0.0); BINS],
            smooth_p: vec![0.0; BINS],
            floor_p: vec![0.0; BINS],
            prev_gain: vec![1.0; BINS],
            prev_snr: vec![0.0; BINS],
            gain: vec![1.0; BINS],
            started: false,
        }
    }

    /// Analysis of the frame ending with `inp`: fills `self.spec`.
    fn analyze(&mut self, inp: &[f32]) {
        self.frame[..HOP].copy_from_slice(&self.prev_in);
        self.frame[HOP..].copy_from_slice(inp);
        self.prev_in.copy_from_slice(inp);
        for (s, w) in self.frame.iter_mut().zip(&self.win) {
            *s *= *w;
        }
        let _ = self.fwd.process_with_scratch(&mut self.frame, &mut self.spec, &mut self.scratch_f);
    }

    fn process(&mut self, inp: &[f32], out: &mut [f32], strength: f32, profile: &[f32]) {
        self.analyze(inp);
        let use_profile = profile.len() == BINS;
        let beta = 1.0 + 2.5 * strength; // over-subtraction
        let floor_gain = db_to_lin(-(8.0 + 28.0 * strength));
        for k in 0..BINS {
            let p = self.spec[k].norm_sqr();
            let n = if use_profile {
                profile[k]
            } else {
                self.smooth_p[k] = if self.started { 0.8 * self.smooth_p[k] + 0.2 * p } else { p };
                self.floor_p[k] = if !self.started || self.smooth_p[k] < self.floor_p[k] { self.smooth_p[k] } else { self.floor_p[k] * 1.003 };
                1.6 * self.floor_p[k]
            };
            let nn = (beta * n).max(1e-12);
            let snr_post = p / nn;
            // decision-directed a-priori SNR, then the Wiener gain
            let xi = 0.92 * self.prev_gain[k].powi(2) * self.prev_snr[k] + 0.08 * (snr_post - 1.0).max(0.0);
            let g = (xi / (1.0 + xi)).max(floor_gain);
            self.prev_snr[k] = snr_post.min(1e4);
            self.gain[k] = g;
        }
        self.started = true;
        // smooth over frequency to tame musical noise
        let mut prev = self.gain[0];
        for k in 1..BINS - 1 {
            let cur = self.gain[k];
            self.gain[k] = 0.25 * prev + 0.5 * cur + 0.25 * self.gain[k + 1];
            prev = cur;
        }
        for k in 0..BINS {
            self.prev_gain[k] = self.gain[k];
            self.spec[k] *= self.gain[k];
        }
        self.spec[0].im = 0.0;
        self.spec[BINS - 1].im = 0.0;
        let _ = self.inv.process_with_scratch(&mut self.spec, &mut self.frame, &mut self.scratch_i);
        let scale = 1.0 / NFFT as f32;
        for i in 0..HOP {
            out[i] = self.tail[i] + self.frame[i] * self.win[i] * scale;
            self.tail[i] = self.frame[HOP + i] * self.win[HOP + i] * scale;
        }
    }
}

/// Mean power per bin of `samples` (frames of `HOP`), for a learned noise profile.
pub fn learn_profile(samples: &[f32]) -> Vec<f32> {
    let mut sp = Spectral::new();
    let mut sum = vec![0.0f64; BINS];
    let mut n = 0usize;
    for frame in samples.chunks_exact(HOP).skip(2) {
        sp.analyze(frame);
        for k in 0..BINS {
            sum[k] += sp.spec[k].norm_sqr() as f64;
        }
        n += 1;
    }
    if n == 0 {
        return Vec::new();
    }
    sum.iter().map(|s| (s / n as f64) as f32).collect()
}

/// DeepFilterNet3 with its buffers. tract's runtime is not `Send`, so it lives on its own thread
/// (see `DfnWorker`) and never leaves it.
struct Dfn {
    df: df::tract::DfTract,
    inp: Array2<f32>,
    enh: Array2<f32>,
    atten: f32,
}

impl Dfn {
    fn new() -> Option<Dfn> {
        let rp = df::tract::RuntimeParams::default_with_ch(1).with_atten_lim(40.0);
        let df = df::tract::DfTract::new(df::tract::DfParams::default(), &rp).ok()?;
        if df.hop_size != HOP || df.sr != FS as usize {
            return None;
        }
        Some(Dfn { df, inp: Array2::zeros((1, HOP)), enh: Array2::zeros((1, HOP)), atten: 40.0 })
    }

    fn process(&mut self, inp: &[f32], out: &mut [f32], atten_db: f32) {
        if (atten_db - self.atten).abs() > 0.5 {
            self.df.set_atten_lim(atten_db);
            self.atten = atten_db;
        }
        self.inp.as_slice_mut().unwrap().copy_from_slice(inp);
        if self.df.process(self.inp.view(), self.enh.view_mut()).is_ok() {
            out.copy_from_slice(self.enh.as_slice().unwrap());
        } else {
            out.copy_from_slice(inp);
        }
    }
}

/// Owns the DeepFilterNet thread: frames go in over a channel and come back processed (about
/// 0.4 ms each). Until the model has loaded, `process` passes the sound through unchanged.
struct DfnWorker {
    tx: std::sync::mpsc::Sender<([f32; HOP], f32)>,
    rx: std::sync::mpsc::Receiver<[f32; HOP]>,
    ready: Arc<std::sync::atomic::AtomicBool>,
}

impl DfnWorker {
    fn spawn() -> DfnWorker {
        let (tx, req_rx) = std::sync::mpsc::channel::<([f32; HOP], f32)>();
        let (res_tx, rx) = std::sync::mpsc::channel::<[f32; HOP]>();
        let ready = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = ready.clone();
        std::thread::spawn(move || {
            let Some(mut d) = Dfn::new() else { return };
            flag.store(true, std::sync::atomic::Ordering::Release);
            while let Ok((frame, atten)) = req_rx.recv() {
                let mut out = [0.0f32; HOP];
                d.process(&frame, &mut out, atten);
                if res_tx.send(out).is_err() {
                    break;
                }
            }
        });
        DfnWorker { tx, rx, ready }
    }

    fn is_ready(&self) -> bool {
        self.ready.load(std::sync::atomic::Ordering::Acquire)
    }

    fn process(&self, inp: &[f32], out: &mut [f32], atten_db: f32) {
        if self.is_ready() {
            let mut frame = [0.0f32; HOP];
            frame.copy_from_slice(inp);
            if self.tx.send((frame, atten_db)).is_ok() {
                if let Ok(o) = self.rx.recv() {
                    out.copy_from_slice(&o);
                    return;
                }
            }
        }
        out.copy_from_slice(inp);
    }
}

/// How far the RNNoise output lags its input (measured in the tests), so the dry/wet mix lines up.
pub const RNNOISE_LATENCY: usize = 480;

struct DelayLine {
    buf: Vec<f32>,
    pos: usize,
}

impl DelayLine {
    fn new(n: usize) -> DelayLine {
        DelayLine { buf: vec![0.0; n.max(1)], pos: 0 }
    }

    #[inline]
    fn push(&mut self, x: f32) -> f32 {
        let y = self.buf[self.pos];
        self.buf[self.pos] = x;
        self.pos = (self.pos + 1) % self.buf.len();
        y
    }
}

pub struct Denoiser {
    mode: DenoiseMode,
    strength: f32,
    profile: Vec<f32>,
    rnn: Option<Box<nnnoiseless::DenoiseState<'static>>>,
    rnn_started: bool,
    dry: DelayLine,
    spec: Option<Spectral>,
    dfn: Option<DfnWorker>,
    buf_a: Vec<f32>,
    buf_b: Vec<f32>,
}

impl Denoiser {
    fn new() -> Denoiser {
        Denoiser {
            mode: DenoiseMode::Off,
            strength: 0.7,
            profile: Vec::new(),
            rnn: None,
            rnn_started: false,
            dry: DelayLine::new(RNNOISE_LATENCY),
            spec: None,
            dfn: None,
            buf_a: vec![0.0; HOP],
            buf_b: vec![0.0; HOP],
        }
    }

    /// Start loading DeepFilterNet in the background (idempotent).
    pub fn preload_dfn(&mut self) {
        if self.dfn.is_none() {
            self.dfn = Some(DfnWorker::spawn());
        }
    }

    pub fn dfn_ready(&self) -> bool {
        self.dfn.as_ref().is_some_and(DfnWorker::is_ready)
    }

    fn set(&mut self, mode: DenoiseMode, strength: f32, profile: &[f32]) {
        if mode != self.mode {
            self.mode = mode;
            match mode {
                DenoiseMode::Rnnoise => {
                    self.rnn = Some(nnnoiseless::DenoiseState::new());
                    self.rnn_started = false;
                }
                DenoiseMode::Spectral => self.spec = Some(Spectral::new()),
                DenoiseMode::Dfn => self.preload_dfn(),
                DenoiseMode::Off => {}
            }
        }
        self.strength = strength;
        if self.profile != profile {
            self.profile = profile.to_vec();
        }
    }

    /// DeepFilterNet's attenuation limit for a strength: 3 dB at 0 up to 80 dB at 1.
    pub fn dfn_atten_db(strength: f32) -> f32 {
        3.0 + 77.0 * strength.clamp(0.0, 1.0).powf(1.5)
    }

    fn process(&mut self, inp: &[f32], out: &mut [f32]) {
        match self.mode {
            DenoiseMode::Off => out.copy_from_slice(inp),
            DenoiseMode::Rnnoise => {
                let rnn = self.rnn.as_mut().unwrap();
                for (d, s) in self.buf_a.iter_mut().zip(inp) {
                    *d = *s * 32768.0;
                }
                rnn.process_frame(&mut self.buf_b, &self.buf_a);
                let wet = self.strength;
                for i in 0..HOP {
                    let dry = self.dry.push(inp[i]);
                    out[i] = if self.rnn_started { dry * (1.0 - wet) + self.buf_b[i] / 32768.0 * wet } else { dry };
                }
                self.rnn_started = true;
            }
            DenoiseMode::Spectral => {
                let strength = self.strength;
                self.spec.as_mut().unwrap().process(inp, out, strength, &self.profile);
            }
            DenoiseMode::Dfn => {
                let atten = Self::dfn_atten_db(self.strength);
                match &self.dfn {
                    Some(d) => d.process(inp, out, atten),
                    None => out.copy_from_slice(inp),
                }
            }
        }
    }
}

// ================================================================================================
// The chain
// ================================================================================================

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameMeters {
    pub in_peak: f32,
    pub in_rms: f32,
    pub out_peak: f32,
    pub out_rms: f32,
    pub gate_open: bool,
    pub comp_gr_db: f32,
    pub limit_gr_db: f32,
    pub mouth_gr_db: f32,
    pub momentary_lufs: f32,
}

pub struct Chain {
    p: DspParams,
    in_gain: f32,
    out_gain: f32,
    hpf: [Biquad; 2],
    pub den: Denoiser,
    mouth: Mouth,
    gate: Gate,
    eq: [Biquad; EQ_BANDS],
    eq_active: [bool; EQ_BANDS],
    deess: DeEsser,
    comp: Compressor,
    lim: Limiter,
    loud: ebur128::EbuR128,
    a: Vec<f32>,
    b: Vec<f32>,
    /// While `Some`, the signal going into the denoiser is collected (for learning a noise profile).
    learn: Option<Vec<f32>>,
}

impl Chain {
    pub fn new(p: DspParams) -> Chain {
        let mut c = Chain {
            p: DspParams::neutral(),
            in_gain: 1.0,
            out_gain: 1.0,
            hpf: [Biquad::IDENTITY; 2],
            den: Denoiser::new(),
            mouth: Mouth::new(),
            gate: Gate::new(),
            eq: [Biquad::IDENTITY; EQ_BANDS],
            eq_active: [false; EQ_BANDS],
            deess: DeEsser::new(),
            comp: Compressor::new(),
            lim: Limiter::new(),
            loud: ebur128::EbuR128::new(1, FS as u32, ebur128::Mode::M).expect("ebur128"),
            a: vec![0.0; HOP],
            b: vec![0.0; HOP],
            learn: None,
        };
        c.set_params(p);
        c
    }

    /// Start collecting the signal that reaches the denoiser (after gain and high-pass).
    pub fn start_learning(&mut self) {
        self.learn = Some(Vec::new());
    }

    /// Stop collecting and turn what was heard into a noise profile (empty if nothing was heard).
    pub fn finish_learning(&mut self) -> Vec<f32> {
        self.learn.take().map(|s| learn_profile(&s)).unwrap_or_default()
    }

    pub fn params(&self) -> &DspParams {
        &self.p
    }

    pub fn set_params(&mut self, mut p: DspParams) {
        p.sanitize();
        self.in_gain = db_to_lin(p.in_gain_db);
        self.out_gain = db_to_lin(p.out_gain_db);
        // 4th-order Butterworth
        self.hpf[0].retune(&Biquad::highpass(p.hpf_hz as f64, 0.5412));
        self.hpf[1].retune(&Biquad::highpass(p.hpf_hz as f64, 1.3066));
        self.den.set(p.denoise, p.denoise_strength, &p.noise_profile);
        if p.mouth_on && !self.p.mouth_on {
            self.mouth.reset(); // nothing old may come out of the delay
        }
        self.mouth.tune(&p);
        self.gate.tune(&p);
        for b in 0..EQ_BANDS {
            self.eq[b].retune(&eq_filter(b, p.eq_gains[b]));
            self.eq_active[b] = p.eq_gains[b].abs() > 0.05;
        }
        self.deess.tune(&p);
        self.comp.tune(&p);
        self.lim.tune(&p);
        self.p = p;
    }

    /// Process one frame of `HOP` samples.
    pub fn process(&mut self, inp: &[f32], out: &mut [f32]) -> FrameMeters {
        debug_assert!(inp.len() == HOP && out.len() == HOP);
        let mut m = FrameMeters::default();
        let mut in_sq = 0.0f32;
        for i in 0..HOP {
            let x = inp[i] * self.in_gain;
            m.in_peak = m.in_peak.max(x.abs());
            in_sq += x * x;
            self.a[i] = if self.p.hpf_on {
                let y = self.hpf[0].process(x);
                self.hpf[1].process(y)
            } else {
                x
            };
        }
        if let Some(buf) = self.learn.as_mut() {
            buf.extend_from_slice(&self.a);
        }
        self.den.process(&self.a, &mut self.b);
        std::mem::swap(&mut self.a, &mut self.b);
        if self.p.mouth_on {
            self.mouth.process(&self.a, &mut self.b);
            std::mem::swap(&mut self.a, &mut self.b);
            m.mouth_gr_db = self.mouth.take_gr();
        }
        if self.p.gate_on {
            for s in self.a.iter_mut() {
                *s = self.gate.process(*s);
            }
            m.gate_open = self.gate.open;
        } else {
            m.gate_open = true;
        }
        if self.p.eq_on {
            for b in 0..EQ_BANDS {
                if self.eq_active[b] {
                    for s in self.a.iter_mut() {
                        *s = self.eq[b].process(*s);
                    }
                }
            }
        }
        if self.p.deess_on {
            for s in self.a.iter_mut() {
                *s = self.deess.process(*s);
            }
        }
        if self.p.comp_on {
            for s in self.a.iter_mut() {
                *s = self.comp.process(*s);
            }
            m.comp_gr_db = self.comp.take_gr();
        }
        let mut out_sq = 0.0f32;
        for i in 0..HOP {
            let mut y = self.a[i] * self.out_gain;
            if self.p.limit_on {
                y = self.lim.process(y);
            }
            if !y.is_finite() {
                y = 0.0;
            }
            m.out_peak = m.out_peak.max(y.abs());
            out_sq += y * y;
            out[i] = y;
        }
        if self.p.limit_on {
            m.limit_gr_db = self.lim.take_gr();
        }
        m.in_rms = (in_sq / HOP as f32).sqrt();
        m.out_rms = (out_sq / HOP as f32).sqrt();
        let _ = self.loud.add_frames_f32(out);
        m.momentary_lufs = self.loud.loudness_momentary().unwrap_or(-70.0) as f32;
        m
    }

    /// How many samples the chain delays the signal in its current setting (for aligning files).
    pub fn latency_samples(&self) -> usize {
        let den = match self.p.denoise {
            DenoiseMode::Off => 0,
            DenoiseMode::Rnnoise => RNNOISE_LATENCY,
            DenoiseMode::Spectral => HOP,
            DenoiseMode::Dfn => 2 * HOP,
        };
        den + if self.p.mouth_on { MOUTH_LATENCY } else { 0 } + if self.p.limit_on { LOOKAHEAD - 1 } else { 0 }
    }
}

/// Run a whole buffer through the chain (offline processing; waits for the DeepFilterNet load).
pub fn process_buffer(p: &DspParams, input: &[f32]) -> Vec<f32> {
    let mut chain = Chain::new(p.clone());
    if p.denoise == DenoiseMode::Dfn {
        while !chain.den.dfn_ready() {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }
    let lat = chain.latency_samples();
    let total = input.len() + lat;
    let frames = total.div_ceil(HOP);
    let mut out = Vec::with_capacity(frames * HOP);
    let mut inb = [0.0f32; HOP];
    let mut outb = [0.0f32; HOP];
    for f in 0..frames {
        for i in 0..HOP {
            inb[i] = input.get(f * HOP + i).copied().unwrap_or(0.0);
        }
        chain.process(&inb, &mut outb);
        out.extend_from_slice(&outb);
    }
    out.drain(..lat.min(out.len()));
    out.truncate(input.len());
    out
}

// ================================================================================================
// Spectrum analyzer for the window
// ================================================================================================

pub struct Analyzer {
    fft: Arc<dyn RealToComplex<f32>>,
    scratch: Vec<Complex<f32>>,
    win: Vec<f32>,
    ring: Vec<f32>,
    frame: Vec<f32>,
    spec: Vec<Complex<f32>>,
    edges: Vec<(usize, usize)>,
    pub bands: [f32; SPECTRUM_BANDS],
}

const ANALYZER_N: usize = 2048;

impl Analyzer {
    pub fn new() -> Analyzer {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(ANALYZER_N);
        let win = (0..ANALYZER_N).map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / ANALYZER_N as f64).cos()) as f32).collect();
        let hz_per_bin = FS / ANALYZER_N as f64;
        let (lo, hi) = (40.0f64, 16000.0f64);
        let edges = (0..SPECTRUM_BANDS)
            .map(|b| {
                let f0 = lo * (hi / lo).powf(b as f64 / SPECTRUM_BANDS as f64);
                let f1 = lo * (hi / lo).powf((b + 1) as f64 / SPECTRUM_BANDS as f64);
                let a = (f0 / hz_per_bin).floor() as usize;
                let z = ((f1 / hz_per_bin).ceil() as usize).max(a + 1);
                (a.max(1), z.min(ANALYZER_N / 2))
            })
            .collect();
        Analyzer { scratch: fft.make_scratch_vec(), fft, win, ring: vec![0.0; ANALYZER_N], frame: vec![0.0; ANALYZER_N], spec: vec![Complex::new(0.0, 0.0); ANALYZER_N / 2 + 1], edges, bands: [0.0; SPECTRUM_BANDS] }
    }

    pub fn push(&mut self, frame: &[f32]) {
        self.ring.copy_within(frame.len().., 0);
        let n = self.ring.len();
        self.ring[n - frame.len()..].copy_from_slice(frame);
    }

    /// Update `bands` to 0..1 (-90..-10 dBFS), fast attack, slow decay.
    pub fn update(&mut self) {
        for (d, (s, w)) in self.frame.iter_mut().zip(self.ring.iter().zip(&self.win)) {
            *d = *s * *w;
        }
        let _ = self.fft.process_with_scratch(&mut self.frame, &mut self.spec, &mut self.scratch);
        for (b, &(a, z)) in self.edges.iter().enumerate() {
            let m = self.spec[a..z.max(a + 1)].iter().map(|c| c.norm()).fold(0.0f32, f32::max);
            let db = lin_to_db(m / (ANALYZER_N as f32 / 4.0));
            let v = ((db + 90.0) / 80.0).clamp(0.0, 1.0);
            let cur = self.bands[b];
            self.bands[b] = if v > cur { cur + (v - cur) * 0.7 } else { cur + (v - cur) * 0.18 };
        }
    }
}

impl Default for Analyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ================================================================================================
// Tests
// ================================================================================================

#[cfg(test)]
pub mod tests {
    use super::*;

    pub fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
    }

    pub fn tone(f: f64, amp: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| amp * (2.0 * PI * f * i as f64 / FS).sin() as f32).collect()
    }

    /// Deterministic white noise in -1..1.
    pub fn noise(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / 8_388_608.0 - 1.0
            })
            .collect()
    }

    /// Voiced-speech-like bursts: a gliding fundamental with 10 harmonics, 1 s on / 1 s off.
    pub fn speechlike(n: usize, amp: f32) -> Vec<f32> {
        let mut phase = 0.0f64;
        (0..n)
            .map(|i| {
                let t = i as f64 / FS;
                let on = t % 2.0 < 1.0;
                let env = if on { (PI * (t % 2.0)).sin().powi(2) as f32 } else { 0.0 };
                let f0 = 130.0 + 25.0 * (t * 2.5).sin();
                phase += 2.0 * PI * f0 / FS;
                let s: f32 = (1..=10).map(|h| (phase * h as f64).sin() as f32 / h as f32 * if h > 6 { 0.6 } else { 1.0 }).sum();
                amp * env * s
            })
            .collect()
    }

    fn db(x: f32) -> f32 {
        lin_to_db(x)
    }

    fn through(mut f: impl FnMut(f32) -> f32, x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| f(v)).collect()
    }

    #[test]
    fn peaking_filter_hits_its_gain_at_the_centre() {
        let b = Biquad::peaking(1000.0, 1.2, 6.0);
        assert!((20.0 * b.magnitude(1000.0).log10() - 6.0).abs() < 0.05);
        assert!(20.0 * b.magnitude(100.0).log10().abs() < 0.5);
        let s = Biquad::low_shelf(80.0, -6.0);
        assert!((20.0 * s.magnitude(20.0).log10() + 6.0).abs() < 0.5, "shelf floor");
        assert!(20.0 * s.magnitude(5000.0).log10().abs() < 0.1, "shelf passband");
        let h = Biquad::high_shelf(10000.0, 4.0);
        assert!((20.0 * h.magnitude(22000.0).log10() - 4.0).abs() < 0.8);
    }

    #[test]
    fn measured_eq_curve_agrees_with_the_filters() {
        let g = [-3.0, 1.0, -2.0, 0.0, 1.0, 2.5, 2.0, 2.0];
        let curve = eq_response_db(&g, &EQ_FREQS);
        // each centre frequency sits near its own gain (neighbouring bands overlap a little)
        for b in 1..EQ_BANDS - 1 {
            assert!((curve[b] - g[b]).abs() < 2.5, "band {b}: {} vs {}", curve[b], g[b]);
        }
        // and actually filtering a tone agrees with the analytic response
        let mut chain = Chain::new(DspParams { eq_on: true, eq_gains: [0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0], ..DspParams::neutral() });
        let x = tone(2500.0, 0.1, FS as usize);
        let mut y = Vec::new();
        let mut o = [0.0f32; HOP];
        for f in x.chunks_exact(HOP) {
            chain.process(f, &mut o);
            y.extend_from_slice(&o);
        }
        let gain = db(rms(&y[24000..])) - db(rms(&x[24000..]));
        assert!(gain > 5.0 && gain < 7.0, "{gain}");
    }

    #[test]
    fn highpass_removes_rumble_and_keeps_voice() {
        let mut h = [Biquad::highpass(80.0, 0.5412), Biquad::highpass(80.0, 1.3066)];
        let mut run = |f: f64| {
            let x = tone(f, 0.5, 48000);
            let y = through(|v| { let t = h[0].process(v); h[1].process(t) }, &x);
            db(rms(&y[24000..])) - db(rms(&x[24000..]))
        };
        assert!(run(30.0) < -20.0);
        assert!(run(1000.0).abs() < 0.3);
    }

    #[test]
    fn gate_closes_on_noise_and_opens_on_speech() {
        let p = DspParams { gate_on: true, gate_threshold_db: -40.0, gate_range_db: 30.0, ..DspParams::neutral() };
        let mut g = Gate::new();
        g.tune(&p);
        let quiet = noise(48000, 1).iter().map(|v| v * 0.003).collect::<Vec<_>>(); // about -51 dBFS peak
        let y = through(|v| g.process(v), &quiet);
        assert!(db(rms(&y[24000..])) < db(rms(&quiet[24000..])) - 20.0, "noise not gated");
        let loud = tone(500.0, 0.3, 24000);
        let y2 = through(|v| g.process(v), &loud);
        assert!(db(rms(&y2[12000..])) > db(rms(&loud[12000..])) - 1.0, "speech not passed");
    }

    #[test]
    fn compressor_follows_its_ratio() {
        let p = DspParams { comp_on: true, comp_threshold_db: -30.0, comp_ratio: 4.0, comp_knee_db: 0.0, comp_auto_makeup: false, comp_makeup_db: 0.0, ..DspParams::neutral() };
        let mut c = Compressor::new();
        c.tune(&p);
        // a tone at -10 dBFS peak: 20 dB over, so 15 dB of reduction (20 * (1 - 1/4))
        let x = tone(300.0, db_to_lin(-10.0), 96000);
        let y = through(|v| c.process(v), &x);
        let red = db(rms(&x[48000..])) - db(rms(&y[48000..]));
        assert!((red - 15.0).abs() < 1.5, "reduction {red}");
        // below threshold nothing happens
        let mut c2 = Compressor::new();
        c2.tune(&p);
        let q = tone(300.0, db_to_lin(-45.0), 48000);
        let yq = through(|v| c2.process(v), &q);
        assert!((db(rms(&q[24000..])) - db(rms(&yq[24000..]))).abs() < 0.3);
    }

    #[test]
    fn limiter_never_passes_the_ceiling() {
        let p = DspParams { limit_on: true, limit_ceiling_db: -3.0, ..DspParams::neutral() };
        let mut l = Limiter::new();
        l.tune(&p);
        let ceiling = db_to_lin(-3.0);
        let mut x = noise(96000, 7);
        for v in x.iter_mut() {
            *v *= 4.0; // way over full scale
        }
        x.extend(tone(200.0, 3.0, 48000));
        let y = through(|v| l.process(v), &x);
        let peak = y.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!(peak <= ceiling + 1e-6, "peak {peak} over {ceiling}");
        // and it still passes quiet material untouched (after the delay)
        let q = tone(500.0, 0.1, 9600);
        let mut l2 = Limiter::new();
        l2.tune(&p);
        let yq = through(|v| l2.process(v), &q);
        assert!((rms(&yq[2000..]) - rms(&q[2000..])).abs() < 0.001);
    }

    #[test]
    fn deesser_cuts_sibilance_more_than_vowels() {
        let p = DspParams { deess_on: true, deess_freq_hz: 6500.0, deess_amount: 0.8, ..DspParams::neutral() };
        let mut d = DeEsser::new();
        d.tune(&p);
        let s = tone(6500.0, 0.3, 48000);
        let ys = through(|v| d.process(v), &s);
        let cut_s = db(rms(&s[24000..])) - db(rms(&ys[24000..]));
        let mut d2 = DeEsser::new();
        d2.tune(&p);
        let v = tone(400.0, 0.3, 48000);
        let yv = through(|x| d2.process(x), &v);
        let cut_v = db(rms(&v[24000..])) - db(rms(&yv[24000..]));
        assert!(cut_s > 5.0 && cut_v < 0.5, "sibilance cut {cut_s} dB, vowel cut {cut_v} dB");
    }

    /// Run `x` through one denoiser; returns the output (same length) and the lag it introduced.
    fn denoise_all(mode: DenoiseMode, strength: f32, x: &[f32], profile: &[f32]) -> Vec<f32> {
        let mut d = Denoiser::new();
        d.set(mode, strength, profile);
        if mode == DenoiseMode::Dfn {
            while !d.dfn_ready() {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        }
        let mut out = Vec::new();
        let mut o = vec![0.0f32; HOP];
        for f in x.chunks_exact(HOP) {
            d.process(f, &mut o);
            out.extend_from_slice(&o);
        }
        out
    }

    /// Mean power over the quiet seconds (t % 2 in 1.3..1.9) and the active ones (0.3..0.8), after warm-up.
    fn quiet_and_active(x: &[f32]) -> (f32, f32) {
        let (mut q, mut a, mut nq, mut na) = (0.0f64, 0.0f64, 0usize, 0usize);
        for (i, &v) in x.iter().enumerate() {
            if i < 3 * FS as usize {
                continue;
            }
            let t = (i as f64 / FS) % 2.0;
            if t > 1.3 && t < 1.9 {
                q += (v * v) as f64;
                nq += 1;
            } else if t > 0.3 && t < 0.8 {
                a += (v * v) as f64;
                na += 1;
            }
        }
        (db((q / nq as f64).sqrt() as f32), db((a / na as f64).sqrt() as f32))
    }

    fn noisy_speech(secs: usize) -> (Vec<f32>, Vec<f32>) {
        let n = secs * FS as usize;
        let clean = speechlike(n, 0.25);
        let nz = noise(n, 99);
        let noisy = clean.iter().zip(&nz).map(|(c, z)| c + 0.03 * z).collect();
        (clean, noisy)
    }

    #[test]
    fn spectral_denoiser_cuts_stationary_noise_and_keeps_the_voice() {
        let (_, noisy) = noisy_speech(10);
        let (q_in, a_in) = quiet_and_active(&noisy);
        let out = denoise_all(DenoiseMode::Spectral, 0.8, &noisy, &[]);
        let (q_out, a_out) = quiet_and_active(&out);
        assert!(q_in - q_out > 10.0, "noise only reduced {} dB", q_in - q_out);
        assert!(a_in - a_out < 3.0, "voice lost {} dB", a_in - a_out);
    }

    #[test]
    fn spectral_denoiser_with_a_learned_profile() {
        let (_, noisy) = noisy_speech(10);
        // learn from two seconds of pure noise recorded "in the room"
        let sample: Vec<f32> = noise(2 * FS as usize, 5).iter().map(|v| v * 0.03).collect();
        let profile = learn_profile(&sample);
        assert_eq!(profile.len(), BINS);
        let (q_in, a_in) = quiet_and_active(&noisy);
        let out = denoise_all(DenoiseMode::Spectral, 0.8, &noisy, &profile);
        let (q_out, a_out) = quiet_and_active(&out);
        assert!(q_in - q_out > 12.0, "noise only reduced {} dB", q_in - q_out);
        assert!(a_in - a_out < 3.0, "voice lost {} dB", a_in - a_out);
    }

    #[test]
    fn rnnoise_reduces_noise_and_its_latency_matches_the_constant() {
        let (_, noisy) = noisy_speech(10);
        let (q_in, _) = quiet_and_active(&noisy);
        let out = denoise_all(DenoiseMode::Rnnoise, 1.0, &noisy, &[]);
        let (q_out, _) = quiet_and_active(&out);
        assert!(q_in - q_out > 6.0, "noise only reduced {} dB", q_in - q_out);
        // the wet signal lags the dry by RNNOISE_LATENCY: find the lag of the strongest correlation
        let seg = 4 * FS as usize..6 * FS as usize;
        let mut best = (0usize, f32::MIN);
        for lag in (0..1200).step_by(1) {
            let c: f32 = noisy[seg.clone()].iter().zip(&out[seg.start + lag..seg.end + lag]).map(|(a, b)| a * b).sum();
            if c > best.1 {
                best = (lag, c);
            }
        }
        println!("rnnoise lag = {} samples (one frame is {RNNOISE_LATENCY})", best.0);
        // the peak of a correlation against a denoised harmonic signal is only good to a few samples
        assert!(best.0.abs_diff(RNNOISE_LATENCY) <= 8, "RNNoise lags {} samples, constant says {RNNOISE_LATENCY}", best.0);
    }

    #[test]
    fn deepfilternet_removes_noise_and_keeps_the_voice() {
        let (clean, noisy) = noisy_speech(10);
        let (q_in, a_in) = quiet_and_active(&noisy);
        let out = denoise_all(DenoiseMode::Dfn, 1.0, &noisy, &[]);
        let (q_out, a_out) = quiet_and_active(&out);
        println!("dfn: noise-only {q_in:.1} -> {q_out:.1} dB, active {a_in:.1} -> {a_out:.1} dB (clean {:.1})", quiet_and_active(&clean).1);
        assert!(q_in - q_out > 15.0, "noise only reduced {} dB", q_in - q_out);
        // This is a synthetic harmonic signal, not speech, and the network was trained on speech,
        // so it may take some of it away: only demand that most of the energy survives. How it
        // treats a real voice has to be judged by ear (the app records the untouched take too).
        assert!(a_in - a_out < 12.0, "voice lost {} dB", a_in - a_out);
    }

    #[test]
    fn chain_preset_podcast_is_safe_and_fast() {
        let (_, noisy) = noisy_speech(10);
        let started = std::time::Instant::now();
        let out = process_buffer(&DspParams::default(), &noisy);
        let rtf = started.elapsed().as_secs_f32() / 10.0;
        println!("podcast preset: RTF {rtf:.3}");
        assert_eq!(out.len(), noisy.len());
        assert!(out.iter().all(|v| v.is_finite()));
        let peak = out.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!(peak <= db_to_lin(-1.0) + 1e-5, "peak {peak}");
        assert!(rtf < 0.5, "too slow for real time: RTF {rtf}");
        let (q_in, _) = quiet_and_active(&noisy);
        let (q_out, a_out) = quiet_and_active(&out);
        assert!(q_out < q_in - 15.0, "quiet part {q_in} -> {q_out}");
        assert!(a_out > -40.0, "voice vanished: {a_out} dB");
    }

    #[test]
    fn learned_profile_and_presets_survive_sanitize() {
        let mut p = DspParams::default();
        p.noise_profile = vec![1.0; BINS];
        assert!(p.apply_preset("stream"));
        assert_eq!(p.noise_profile.len(), BINS, "profile kept across presets");
        assert!(!p.apply_preset("nonsense"));
        p.denoise_strength = f32::NAN;
        p.eq_gains[3] = 99.0;
        p.sanitize();
        assert_eq!((p.denoise_strength, p.eq_gains[3]), (0.7, 15.0));
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(serde_json::from_str::<DspParams>(&json).unwrap(), p);
    }

    #[test]
    fn analyzer_puts_a_tone_in_the_right_band() {
        let mut a = Analyzer::new();
        let x = tone(1000.0, 0.5, 48000);
        for f in x.chunks_exact(HOP) {
            a.push(f);
            a.update();
        }
        let top = a.bands.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;
        let f0 = 40.0 * (16000.0f64 / 40.0).powf(top as f64 / SPECTRUM_BANDS as f64);
        let f1 = 40.0 * (16000.0f64 / 40.0).powf((top + 1) as f64 / SPECTRUM_BANDS as f64);
        assert!(f0 <= 1000.0 * 1.05 && f1 >= 1000.0 / 1.05, "peak band {top} covers {f0:.0}..{f1:.0} Hz");
    }
}
