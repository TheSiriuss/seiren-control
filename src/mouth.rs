//! Mouth noise reduction: lip smacks, tongue clicks, saliva crackle, chewing, gulps, "wet" sounds.
//!
//! Two stages in one block, with one fixed delay of `MOUTH_LATENCY` samples:
//!
//!  1. **Click scoop.** Mouth clicks are impulsive: a spike of a few milliseconds in the band
//!     above 2.5 kHz that stands far above the level of that band just before *and* just after it.
//!     A vowel or a fricative builds up and stays, so it never looks like that. The detector judges
//!     each moment 8 ms late, so it can look at what follows; when it finds a click the signal
//!     is crossfaded to a copy of itself that has been low-passed at 1.6 kHz, for a few milliseconds,
//!     so the spike goes and the voice underneath keeps its body (outside a click the output is the
//!     input, only delayed).
//!
//!  2. **Orphan-sound ducking.** Each 10 ms frame is classed as *voiced* (periodic and heavy in
//!     the low frequencies: vowels, "м", "н", "л") or not, and as *active* (clearly above the noise
//!     floor) or not. An active frame with no voiced frame within 40 ms before or 50 ms after it
//!     belongs to no word: a smack, a chew, a gulp, saliva, a breath. If it is also quiet compared
//!     with the speaker's usual voice (real consonants are not), that frame is turned down. A
//!     sound that touches a word stays, because voice follows or precedes it closely. If the
//!     stream hardly ever contains voiced sound (whispering, something that is not speech) the
//!     stage stays out of the way.
//!
//! No allocation per frame; the heavy parts are built once.

use crate::dsp::{db_to_lin, lin_to_db, Biquad, DspParams, FS, HOP};

// ------------------------------------------------------------------------------------------------
// Stage 1: click scoop
// ------------------------------------------------------------------------------------------------

const RING1: usize = 1024; // power of two
const MASK1: usize = RING1 - 1;
/// The output lags the newest sample by this much (10 ms).
const DELAY1: usize = 480;
/// The detector judges the sample this far behind the newest one.
const CENTER: usize = 400;
/// Running windows, in samples behind the newest: [BEFORE_FAR, BEFORE_NEAR) and [AFTER_FAR, AFTER_NEAR).
const BEFORE_FAR: usize = 640;
const BEFORE_NEAR: usize = 448;
const AFTER_FAR: usize = 256;
const AFTER_NEAR: usize = 16;
/// The click detector listens above this (4th-order high-pass) ...
const DET_HZ: f64 = 2500.0;
/// ... and during a click the signal is replaced by its low-pass at this frequency.
const CUT_HZ: f64 = 1600.0;
/// Group delay of that low-pass at low frequencies, so the untouched signal lines up with it.
const TAU: usize = 12;
/// A spike below this peak (about -55 dBFS) is not worth chasing.
const CLICK_FLOOR: f32 = 0.0018;

fn coef(ms: f32) -> f32 {
    1.0 - (-1.0 / (ms.max(0.01) * 0.001 * FS as f32)).exp()
}

struct Scoop {
    det: [Biquad; 2],
    cut: [Biquad; 2],
    x: Vec<f32>,
    low: Vec<f32>,
    h: Vec<f32>,
    target: Vec<f32>,
    n: usize,
    sum_before: f64,
    sum_after: f64,
    ratio: f32,
    depth: f32,
    gain: f32,
    att: f32,
    rel: f32,
    max_gr: f32,
}

impl Scoop {
    fn new() -> Scoop {
        Scoop {
            det: [Biquad::highpass(DET_HZ, 0.5412), Biquad::highpass(DET_HZ, 1.3066)],
            cut: [Biquad::lowpass(CUT_HZ, 0.5412), Biquad::lowpass(CUT_HZ, 1.3066)],
            x: vec![0.0; RING1],
            low: vec![0.0; RING1],
            h: vec![0.0; RING1],
            target: vec![1.0; RING1],
            n: 0,
            sum_before: 0.0,
            sum_after: 0.0,
            ratio: 7.0,
            depth: 0.1,
            gain: 1.0,
            att: coef(0.2),
            rel: coef(2.5),
            max_gr: 0.0,
        }
    }

    fn tune(&mut self, amount: f32, sens: f32) {
        // how far above the surroundings a spike must stand: 24 dB (careful) .. 10 dB (eager)
        self.ratio = db_to_lin(24.0 - 14.0 * sens);
        self.depth = db_to_lin(-(10.0 + 24.0 * amount));
    }

    #[inline]
    fn sq(&self, behind: usize) -> f64 {
        let v = self.h[self.n.wrapping_sub(behind) & MASK1] as f64;
        v * v
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let a = self.det[0].process(x);
        let high = self.det[1].process(a);
        let b = self.cut[0].process(x);
        let low = self.cut[1].process(b);
        let n = self.n;
        let i = n & MASK1;
        self.x[i] = x;
        self.low[i] = low;
        self.h[i] = high;
        self.target[i] = 1.0;
        // slide both windows by one sample: the newest sample of each comes in, the oldest goes out
        self.sum_before += self.sq(BEFORE_NEAR + 1) - self.sq(BEFORE_FAR + 1);
        self.sum_after += self.sq(AFTER_NEAR + 1) - self.sq(AFTER_FAR + 1);

        if n >= BEFORE_FAR + 2 && self.h[n.wrapping_sub(CENTER) & MASK1].abs() > CLICK_FLOOR * 0.25 {
            let c = n - CENTER;
            let mut peak = 0.0f32;
            for d in c - 12..=c + 12 {
                peak = peak.max(self.h[d & MASK1].abs());
            }
            if peak > CLICK_FLOOR {
                let before = (self.sum_before.max(0.0) / (BEFORE_FAR - BEFORE_NEAR) as f64).sqrt() as f32;
                let after = (self.sum_after.max(0.0) / (AFTER_FAR - AFTER_NEAR) as f64).sqrt() as f32;
                if peak > self.ratio * before.max(after) {
                    for j in c - 24..=c + 144 {
                        let t = &mut self.target[j & MASK1];
                        *t = t.min(self.depth);
                    }
                }
            }
        }

        self.n += 1;
        if n < DELAY1 {
            return 0.0;
        }
        let o = (n - DELAY1) & MASK1;
        let t = self.target[o];
        self.gain += (t - self.gain) * if t < self.gain { self.att } else { self.rel };
        if self.gain < 0.99 {
            self.max_gr = self.max_gr.max(-lin_to_db(self.gain));
        }
        let dry = self.x[(n - DELAY1).wrapping_sub(TAU) & MASK1];
        self.gain * dry + (1.0 - self.gain) * self.low[o]
    }
}

// ------------------------------------------------------------------------------------------------
// Stage 2: sounds that belong to no word
// ------------------------------------------------------------------------------------------------

/// Frames of look-ahead: voice that starts within this many frames protects what comes before it.
const LA: usize = 5;
/// Frames of memory: voice that ended within this many frames protects what comes after it.
const KB: usize = 4;
const FEAT_RING: usize = 16;
const DEC: usize = 6; // 48 kHz -> 8 kHz for the pitch search
const DEC_RING: usize = 512;
const PITCH_WINDOW: usize = 256; // 32 ms
const LAG_MIN: usize = 20; // 400 Hz
const LAG_MAX: usize = 100; // 80 Hz
const VOICED_R: f32 = 0.6;
const FLOOR_FRAMES: usize = 300; // the noise floor is the quietest frame of the last 3 s
/// The last `PRE` samples of a frame already move towards the next frame's gain.
const PRE: usize = 240;

#[derive(Clone, Copy, Default)]
struct Feat {
    db: f32,
    voiced: bool,
    active: bool,
}

struct Orphans {
    low: [Biquad; 2],
    dec_acc: f32,
    dec_n: usize,
    dec: Vec<f32>,
    dec_pos: usize,
    frames: Vec<[f32; HOP]>,
    feat: [Feat; FEAT_RING],
    goal: [f32; FEAT_RING],
    hist: Vec<f32>,
    hist_pos: usize,
    m: usize,
    ref_db: f32,
    voiced_count: f32,
    active_count: f32,
    protect_db: f32,
    depth: f32,
    gain: f32,
    att: f32,
    rel: f32,
    max_gr: f32,
}

impl Orphans {
    fn new() -> Orphans {
        let lp = Biquad::lowpass(800.0, 0.7071);
        Orphans {
            low: [lp, lp],
            dec_acc: 0.0,
            dec_n: 0,
            dec: vec![0.0; DEC_RING],
            dec_pos: 0,
            frames: vec![[0.0; HOP]; LA + 2],
            feat: [Feat::default(); FEAT_RING],
            goal: [1.0; FEAT_RING],
            hist: vec![-120.0; FLOOR_FRAMES],
            hist_pos: 0,
            m: 0,
            ref_db: -32.0,
            voiced_count: 0.0,
            active_count: 0.0,
            protect_db: 20.0,
            depth: 0.1,
            gain: 1.0,
            att: coef(2.5),
            rel: coef(5.0),
            max_gr: 0.0,
        }
    }

    fn tune(&mut self, amount: f32, sens: f32) {
        // frames quieter than (the speaker's voice - protect) count as mouth noise
        self.protect_db = 30.0 - 20.0 * sens;
        self.depth = db_to_lin(-(6.0 + 24.0 * amount));
    }

    /// Normalised autocorrelation peak of the low band over the last 32 ms: ~0.8 for a vowel,
    /// ~0.2 for noise, a smack or a hiss.
    fn periodicity(&self) -> f32 {
        let at = |back: usize| self.dec[(self.dec_pos + DEC_RING - 1 - back) % DEC_RING];
        let mut e0 = 0.0f32;
        for t in 0..PITCH_WINDOW {
            e0 += at(t) * at(t);
        }
        if e0 < 1e-12 {
            return 0.0;
        }
        let mut best = 0.0f32;
        for lag in LAG_MIN..=LAG_MAX {
            let (mut xy, mut yy) = (0.0f32, 0.0f32);
            for t in 0..PITCH_WINDOW {
                let y = at(t + lag);
                xy += at(t) * y;
                yy += y * y;
            }
            if xy > 0.0 && yy > 1e-12 {
                best = best.max(xy / (e0 * yy).sqrt());
            }
        }
        best
    }

    fn analyse(&mut self, frame: &[f32]) -> Feat {
        let mut sq = 0.0f32;
        let mut low_sq = 0.0f32;
        for &x in frame {
            sq += x * x;
            let first = self.low[0].process(x);
            let l = self.low[1].process(first);
            low_sq += l * l;
            self.dec_acc += l;
            self.dec_n += 1;
            if self.dec_n == DEC {
                self.dec[self.dec_pos] = self.dec_acc / DEC as f32;
                self.dec_pos = (self.dec_pos + 1) % DEC_RING;
                self.dec_acc = 0.0;
                self.dec_n = 0;
            }
        }
        let db = lin_to_db((sq / HOP as f32).sqrt());
        let low_db = lin_to_db((low_sq / HOP as f32).sqrt());
        let voiced = low_db > -64.0 && low_db > db - 9.0 && self.periodicity() > VOICED_R;
        self.hist[self.hist_pos] = db;
        self.hist_pos = (self.hist_pos + 1) % FLOOR_FRAMES;
        let floor = self.hist.iter().fold(f32::MAX, |a, &b| a.min(b)).max(-78.0);
        let active = db > -66.0 && db > floor + 9.0;
        Feat { db, voiced, active }
    }

    fn process(&mut self, inp: &[f32], out: &mut [f32]) {
        let m = self.m;
        let slot = m % self.frames.len();
        self.frames[slot].copy_from_slice(inp);
        let f = self.analyse(inp);
        self.feat[m % FEAT_RING] = f;
        // the speaker's usual voice level: it follows voiced frames only (so a long pause does not
        // erase it), rising fast to louder ones and sinking over a few seconds of speech
        if f.voiced {
            let k = if f.db > self.ref_db { 0.05 } else { 0.002 };
            self.ref_db = (self.ref_db + (f.db - self.ref_db) * k).clamp(-60.0, 0.0);
        }
        // how much of the activity so far was voice (a minute of memory); the prior of 8 frames
        // makes a fresh start behave as speech
        let keep = 1.0 - 1.0 / 6000.0;
        self.voiced_count *= keep;
        self.active_count *= keep;
        if f.voiced {
            self.voiced_count += 1.0;
        }
        if f.voiced || f.active {
            self.active_count += 1.0;
        }
        let speaker = (self.voiced_count + 8.0) / (self.active_count + 8.0) >= 0.1;

        // decide the frame LA frames back: it can see LA frames ahead and KB frames behind
        if m >= LA {
            let h = m - LA;
            let near_voice = (h.saturating_sub(KB)..=m).any(|k| self.feat[k % FEAT_RING].voiced);
            let fh = self.feat[h % FEAT_RING];
            let duck = speaker && fh.active && !near_voice && fh.db < self.ref_db - self.protect_db;
            self.goal[h % FEAT_RING] = if duck { self.depth } else { 1.0 };
        }
        // the frame before it is the one that leaves now
        if m > LA {
            let o = m - LA - 1;
            let (g_now, g_next) = (self.goal[o % FEAT_RING], self.goal[(o + 1) % FEAT_RING]);
            let src = &self.frames[o % self.frames.len()];
            for i in 0..HOP {
                let t = if i >= HOP - PRE { g_now.min(g_next) } else { g_now };
                self.gain += (t - self.gain) * if t < self.gain { self.att } else { self.rel };
                out[i] = src[i] * self.gain;
            }
            if self.gain < 0.99 {
                self.max_gr = self.max_gr.max(-lin_to_db(self.gain));
            }
        } else {
            out.fill(0.0);
        }
        self.m += 1;
    }
}

// ------------------------------------------------------------------------------------------------
// Both stages
// ------------------------------------------------------------------------------------------------

/// Total delay of the block, in samples (about 70 ms).
pub const MOUTH_LATENCY: usize = DELAY1 + TAU + (LA + 1) * HOP;

pub struct Mouth {
    scoop: Scoop,
    orphans: Orphans,
    mid: Vec<f32>,
    amount: f32,
    sens: f32,
}

impl Mouth {
    pub fn new() -> Mouth {
        let mut m = Mouth { scoop: Scoop::new(), orphans: Orphans::new(), mid: vec![0.0; HOP], amount: 0.6, sens: 0.5 };
        m.tune(&DspParams::neutral());
        m
    }

    /// `mouth_amount` is how far a mouth sound is turned down (6..30 dB), `mouth_sens` how readily
    /// a sound is taken for mouth noise.
    pub fn tune(&mut self, p: &DspParams) {
        self.amount = p.mouth_amount;
        self.sens = p.mouth_sens;
        self.scoop.tune(self.amount, self.sens);
        self.orphans.tune(self.amount, self.sens);
    }

    /// Forget everything heard so far (used when the effect is switched on, so no old audio
    /// comes out of the delay).
    pub fn reset(&mut self) {
        let (amount, sens) = (self.amount, self.sens);
        *self = Mouth::new();
        self.amount = amount;
        self.sens = sens;
        self.scoop.tune(amount, sens);
        self.orphans.tune(amount, sens);
    }

    /// One frame of `HOP` samples in, one out, `MOUTH_LATENCY` samples late.
    pub fn process(&mut self, inp: &[f32], out: &mut [f32]) {
        for (m, &x) in self.mid.iter_mut().zip(inp) {
            *m = self.scoop.process(x);
        }
        self.orphans.process(&self.mid, out);
    }

    /// The deepest reduction (dB, positive) heard since the last call.
    pub fn take_gr(&mut self) -> f32 {
        let a = std::mem::take(&mut self.scoop.max_gr);
        let b = std::mem::take(&mut self.orphans.max_gr);
        a.max(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::tests::{noise, rms};
    use std::f64::consts::PI;

    /// Voiced "words": a harmonic tone with a gliding pitch and 15 ms ramps, silence elsewhere.
    fn words(total_s: f64, spans: &[(f64, f64)], amp: f32) -> Vec<f32> {
        let mut phase = 0.0f64;
        (0..(total_s * FS) as usize)
            .map(|i| {
                let t = i as f64 / FS;
                let mut env = 0.0f32;
                for &(a, b) in spans {
                    if t >= a && t < b {
                        env = ((t - a) / 0.015).min(1.0).min((b - t) / 0.015) as f32;
                    }
                }
                phase += 2.0 * PI * (125.0 + 20.0 * (t * 3.0).sin()) / FS;
                let s: f32 = (1..=10).map(|h| (phase * h as f64).sin() as f32 / h as f32 * if h > 6 { 0.6 } else { 1.0 }).sum();
                amp * env * s
            })
            .collect()
    }

    fn shaped(n: usize, seed: u32, lo: f64, hi: f64) -> Vec<f32> {
        let (mut hp, mut lp) = (Biquad::highpass(lo, 0.7071), Biquad::lowpass(hi, 0.7071));
        noise(n, seed).into_iter().map(|v| lp.process(hp.process(v))).collect()
    }

    fn to_rms(mut v: Vec<f32>, r: f32) -> Vec<f32> {
        let k = r / rms(&v);
        v.iter_mut().for_each(|s| *s *= k);
        v
    }

    fn put(sig: &mut [f32], at_s: f64, ev: &[f32]) {
        let i = (at_s * FS) as usize;
        for (k, v) in ev.iter().enumerate() {
            sig[i + k] += v;
        }
    }

    fn db_in(x: &[f32], a: f64, b: f64) -> f32 {
        lin_to_db(rms(&x[(a * FS) as usize..(b * FS) as usize]))
    }

    /// A tongue/lip click: 3 ms of 2.5-9 kHz noise dying away in 0.7 ms.
    fn click(peak: f32) -> Vec<f32> {
        let mut v = shaped(144, 99, 2500.0, 9000.0);
        for (i, s) in v.iter_mut().enumerate() {
            *s *= (-(i as f32) / (0.0007 * FS as f32)).exp();
        }
        let p = v.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        v.iter().map(|s| s / p * peak).collect()
    }

    /// A lip smack: 30 ms of 1-5 kHz noise decaying with a 10 ms time constant.
    fn smack(r: f32) -> Vec<f32> {
        let mut v = shaped(1440, 7, 1000.0, 5000.0);
        for (i, s) in v.iter_mut().enumerate() {
            *s *= (-(i as f32) / (0.010 * FS as f32)).exp();
        }
        to_rms(v, r)
    }

    /// Saliva / water in the mouth: 0.8-3 kHz noise chopped at 40 Hz for 200 ms.
    fn gurgle(r: f32) -> Vec<f32> {
        let mut v = shaped(9600, 11, 800.0, 3000.0);
        for (i, s) in v.iter_mut().enumerate() {
            *s *= 0.5 + 0.5 * (2.0 * PI * 40.0 * i as f64 / FS).sin() as f32;
        }
        to_rms(v, r)
    }

    /// A sibilant of `ms` milliseconds: 4-8 kHz noise with 10 ms ramps.
    fn hiss(ms: usize, r: f32) -> Vec<f32> {
        let n = ms * 48;
        let mut v = shaped(n, 21, 4000.0, 8000.0);
        for (i, s) in v.iter_mut().enumerate() {
            *s *= (i as f32 / 480.0).min(1.0).min((n - i) as f32 / 480.0);
        }
        to_rms(v, r)
    }

    fn hf(x: &[f32]) -> Vec<f32> {
        let mut h = Biquad::highpass(2500.0, 0.7071);
        x.iter().map(|&v| h.process(v)).collect()
    }

    fn mouth() -> Mouth {
        let mut m = Mouth::new();
        m.tune(&DspParams { mouth_on: true, ..DspParams::neutral() });
        m
    }

    /// Run a whole signal through and return it with the delay taken out.
    fn run(m: &mut Mouth, x: &[f32]) -> Vec<f32> {
        let frames = (x.len() + MOUTH_LATENCY).div_ceil(HOP);
        let mut out = Vec::with_capacity(frames * HOP);
        let (mut inb, mut ob) = ([0.0f32; HOP], [0.0f32; HOP]);
        for f in 0..frames {
            for i in 0..HOP {
                inb[i] = x.get(f * HOP + i).copied().unwrap_or(0.0);
            }
            m.process(&inb, &mut ob);
            out.extend_from_slice(&ob);
        }
        out.drain(..MOUTH_LATENCY);
        out.truncate(x.len());
        out
    }

    fn add_floor(x: &mut [f32]) {
        let nz = noise(x.len(), 3);
        for (a, b) in x.iter_mut().zip(nz) {
            *a += b * 0.0003; // about -75 dBFS of room tone
        }
    }

    #[test]
    fn a_clean_voice_passes_unharmed_and_exactly_delayed() {
        let spans = [(0.3, 1.2), (1.5, 2.6), (2.9, 3.7)];
        let mut x = words(4.0, &spans, 0.12);
        add_floor(&mut x);
        let y = run(&mut mouth(), &x);
        for &(a, b) in &spans {
            let (i, j) = (((a + 0.2) * FS) as usize, ((b - 0.2) * FS) as usize);
            let err: Vec<f32> = (i..j).map(|k| y[k] - x[k]).collect();
            let rel = lin_to_db(rms(&err)) - lin_to_db(rms(&x[i..j]));
            assert!(rel < -30.0, "word {a}-{b}: error {rel} dB");
        }
    }




    #[test]
    fn a_steady_hiss_does_not_look_like_a_click() {
        let hs = to_rms(shaped(96000, 4, 3000.0, 9000.0), 0.02);
        let mut s = Scoop::new();
        s.tune(0.6, 0.5);
        let y: Vec<f32> = hs.iter().map(|&v| s.process(v)).collect();
        let (a, b) = (9600, 90000);
        let d = lin_to_db(rms(&y[a + DELAY1..b + DELAY1])) - lin_to_db(rms(&hs[a..b]));
        assert!(d.abs() < 0.5, "hiss changed by {d} dB");
        assert!(s.max_gr < 1.0, "scoop engaged on hiss: {} dB", s.max_gr);
    }

    #[test]
    fn clicks_are_scooped_in_pauses_and_inside_words() {
        let spans = [(0.3, 1.2), (1.9, 2.8)];
        let mut x = words(3.4, &spans, 0.12);
        add_floor(&mut x);
        let times = [0.8, 1.55, 2.4];
        for &t in &times {
            put(&mut x, t, &click(0.2));
        }
        let y = run(&mut mouth(), &x);
        let (hx, hy) = (hf(&x), hf(&y));
        for &t in &times {
            let before = db_in(&hx, t - 0.001, t + 0.004);
            let after = db_in(&hy, t - 0.001, t + 0.004);
            println!("click at {t}: {before:.1} -> {after:.1} dB above 2.5 kHz");
            assert!(before - after > 12.0, "click at {t} s only went down by {} dB", before - after);
        }
        // and the words around them are intact
        for &(a, b) in &spans {
            let (i, j) = (((a + 0.05) * FS) as usize, ((b - 0.05) * FS) as usize);
            let d = lin_to_db(rms(&y[i..j])) - lin_to_db(rms(&x[i..j]));
            assert!(d.abs() < 0.6, "word {a}-{b} changed by {d} dB");
        }
    }

    #[test]
    fn smacks_and_wet_sounds_between_words_are_turned_down() {
        let spans = [(0.3, 1.0), (1.1, 1.8), (4.2, 5.0)];
        let mut x = words(5.4, &spans, 0.12);
        add_floor(&mut x);
        put(&mut x, 2.5, &smack(0.005));
        put(&mut x, 3.2, &gurgle(0.004));
        let y = run(&mut mouth(), &x);
        for (name, a, b) in [("smack", 2.5, 2.53), ("gurgle", 3.2, 3.4)] {
            let (before, after) = (db_in(&x, a, b), db_in(&y, a, b));
            println!("{name}: {before:.1} -> {after:.1} dBFS");
            assert!(before - after > 10.0, "{name} only went down by {} dB", before - after);
        }
        for &(a, b) in &spans {
            let (i, j) = (((a + 0.05) * FS) as usize, ((b - 0.05) * FS) as usize);
            let d = lin_to_db(rms(&y[i..j])) - lin_to_db(rms(&x[i..j]));
            assert!(d.abs() < 0.5, "word {a}-{b} changed by {d} dB");
        }
    }

    #[test]
    fn sounds_that_touch_a_word_are_kept() {
        let spans = [(0.5, 1.3), (2.5, 3.3), (4.0, 4.8)];
        let mut x = words(5.4, &spans, 0.12);
        add_floor(&mut x);
        // a loud "ш" running into the vowel, and a quiet "ф" that ends just as the voice starts
        put(&mut x, 0.5 - 0.09, &hiss(90, 0.03));
        put(&mut x, 2.5 - 0.06, &hiss(60, 0.004));
        // a trailing breath of a consonant right after the voice stops
        put(&mut x, 4.8, &hiss(60, 0.004));
        let y = run(&mut mouth(), &x);
        let kept = |a: f64, b: f64| db_in(&y, a, b) - db_in(&x, a, b);
        assert!(kept(0.41, 0.50) > -3.0, "loud hiss before a word lost {} dB", -kept(0.41, 0.50));
        assert!(kept(2.46, 2.50) > -3.0, "quiet hiss just before a word lost {} dB", -kept(2.46, 2.50));
        assert!(kept(4.8, 4.84) > -3.0, "trailing hiss lost {} dB", -kept(4.8, 4.84));
    }

    #[test]
    fn the_chain_applies_it_and_hands_back_aligned_audio() {
        let spans = [(0.3, 1.0), (1.1, 1.8), (4.2, 5.0)];
        let mut x = words(5.4, &spans, 0.12);
        add_floor(&mut x);
        put(&mut x, 2.5, &smack(0.005));
        let p = DspParams { mouth_on: true, ..DspParams::neutral() };
        let y = crate::dsp::process_buffer(&p, &x);
        assert_eq!(y.len(), x.len());
        assert!(db_in(&x, 2.5, 2.53) - db_in(&y, 2.5, 2.53) > 10.0, "smack survived the chain");
        let d = db_in(&y, 0.35, 0.95) - db_in(&x, 0.35, 0.95);
        assert!(d.abs() < 0.5, "word changed by {d} dB or misaligned");
        // switched off, nothing is delayed or touched
        let off = crate::dsp::process_buffer(&DspParams::neutral(), &x);
        assert!(x.iter().zip(&off).all(|(a, b)| (a - b).abs() < 1e-6));
    }

    #[test]
    fn settings_saved_before_this_existed_load_with_it_off() {
        let mut obj = serde_json::to_value(DspParams::neutral()).unwrap().as_object().unwrap().clone();
        for k in ["mouth_on", "mouth_amount", "mouth_sens"] {
            obj.remove(k);
        }
        let p: DspParams = serde_json::from_value(serde_json::Value::Object(obj)).unwrap();
        assert!(!p.mouth_on, "an old file must not switch a 70 ms delay on by itself");
        assert!(p.mouth_amount > 0.0 && p.mouth_sens > 0.0);
    }

    #[test]
    fn whispering_is_left_alone() {
        // 40 s of unvoiced "speech": 0.8 s of noise, 0.3 s of rest, over and over
        let n = 40 * 48000;
        let mut x = to_rms(shaped(n, 5, 300.0, 4000.0), 0.02);
        for (i, s) in x.iter_mut().enumerate() {
            if (i as f64 / FS) % 1.1 > 0.8 {
                *s = 0.0;
            }
        }
        let y = run(&mut mouth(), &x);
        let (a, b) = (20 * 48000, 40 * 48000);
        let d = lin_to_db(rms(&y[a..b])) - lin_to_db(rms(&x[a..b]));
        assert!(d > -1.5, "whisper turned down by {} dB", -d);
    }

    #[test]
    fn extreme_input_stays_finite_and_cheap() {
        let mut x: Vec<f32> = noise(96000, 8).iter().map(|v| v * 4.0).collect();
        x.extend(vec![0.0; 48000]);
        x.extend(vec![0.9; 4800]);
        x.extend(words(20.0, &[(0.5, 3.0), (4.0, 9.0), (10.0, 19.0)], 0.2));
        let started = std::time::Instant::now();
        let y = run(&mut mouth(), &x);
        let took = started.elapsed().as_secs_f32();
        let secs = x.len() as f32 / FS as f32;
        println!("{secs:.0} s of audio in {took:.2} s");
        assert!(y.iter().all(|v| v.is_finite() && v.abs() < 20.0));
        assert!(took < secs / 15.0, "too slow: {took:.2} s for {secs:.0} s");
    }
}
