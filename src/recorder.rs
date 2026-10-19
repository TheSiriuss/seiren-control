//! Writing a take to disk. Frames arrive over a channel and are streamed into a 24-bit WAV (the
//! engine never waits for the disk); when the take ends the writer thread finishes the file,
//! optionally brings it to -16 LUFS, optionally converts it with ffmpeg, and reports back.

use crate::dsp::{FS, HOP};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{channel, Sender};
use std::sync::Arc;
use std::thread;

pub const TARGET_LUFS: f64 = -16.0;
const PEAK_CEILING_DB: f64 = -1.0;
const WAVE_BINS: usize = 480;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Format {
    #[default]
    Wav,
    Flac,
    Opus,
    Mp3,
}

impl Format {
    pub const ALL: [Format; 4] = [Format::Wav, Format::Flac, Format::Opus, Format::Mp3];

    pub fn ext(self) -> &'static str {
        match self {
            Format::Wav => "wav",
            Format::Flac => "flac",
            Format::Opus => "opus",
            Format::Mp3 => "mp3",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Format::Wav => "WAV 24 бит",
            Format::Flac => "FLAC",
            Format::Opus => "Opus",
            Format::Mp3 => "MP3",
        }
    }

    fn ffmpeg_args(self) -> &'static [&'static str] {
        match self {
            Format::Wav => &[],
            Format::Flac => &["-c:a", "flac", "-compression_level", "8"],
            Format::Opus => &["-c:a", "libopus", "-b:a", "128k"],
            Format::Mp3 => &["-c:a", "libmp3lame", "-q:a", "2"],
        }
    }
}

/// Which of the formats this machine can write: WAV always, the rest need ffmpeg with the encoder.
pub fn available_formats() -> Vec<Format> {
    let enc = Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default();
    let has = |name: &str| enc.lines().any(|l| l.split_whitespace().nth(1) == Some(name));
    Format::ALL
        .into_iter()
        .filter(|f| match f {
            Format::Wav => true,
            Format::Flac => has("flac"),
            Format::Opus => has("libopus"),
            Format::Mp3 => has("libmp3lame"),
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct RecordOpts {
    pub dir: PathBuf,
    pub format: Format,
    pub keep_raw: bool,
    pub normalize: bool,
}

#[derive(Clone, Debug)]
pub struct Take {
    pub path: PathBuf,
    pub raw_path: Option<PathBuf>,
    pub secs: f32,
    /// (min, max) of the processed take in `WAVE_BINS` slices, for the waveform.
    pub peaks: Vec<(f32, f32)>,
    /// Integrated loudness of the finished file.
    pub lufs: Option<f32>,
    pub note: Option<String>,
}

pub type Done = Arc<dyn Fn(Result<Take, String>) + Send + Sync>;

pub struct Recorder {
    tx: Sender<(Vec<f32>, Vec<f32>)>,
    handle: thread::JoinHandle<()>,
    frames: u64,
}

fn local_timestamp() -> String {
    unsafe {
        let t = libc::time(std::ptr::null_mut());
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&t, &mut tm);
        format!("{:04}-{:02}-{:02} {:02}-{:02}-{:02}", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec)
    }
}

fn wav_spec() -> hound::WavSpec {
    hound::WavSpec { channels: 1, sample_rate: FS as u32, bits_per_sample: 24, sample_format: hound::SampleFormat::Int }
}

type Writer = hound::WavWriter<std::io::BufWriter<fs::File>>;

#[inline]
fn to_i24(x: f32) -> i32 {
    (x.clamp(-1.0, 1.0) * 8_388_607.0) as i32
}

impl Recorder {
    /// Create the file(s) and start the writer thread. `done` is called from that thread when
    /// the take is completely finished (normalised, converted).
    pub fn start(opts: RecordOpts, done: Done) -> Result<Recorder, String> {
        fs::create_dir_all(&opts.dir).map_err(|e| format!("не удалось создать папку {}: {e}", opts.dir.display()))?;
        let stamp = local_timestamp();
        let wav_path = opts.dir.join(format!("Seiren {stamp}.wav"));
        let raw_wav = opts.dir.join(format!("Seiren {stamp} (оригинал).wav"));
        let mut main = hound::WavWriter::create(&wav_path, wav_spec()).map_err(|e| format!("не удалось создать файл: {e}"))?;
        let mut raw = if opts.keep_raw {
            Some(hound::WavWriter::create(&raw_wav, wav_spec()).map_err(|e| format!("не удалось создать файл: {e}"))?)
        } else {
            None
        };
        let (tx, rx) = channel::<(Vec<f32>, Vec<f32>)>();
        let handle = thread::spawn(move || {
            let mut peaks: Vec<(f32, f32)> = Vec::new();
            let mut samples = 0u64;
            let mut failure: Option<String> = None;
            while let Ok((processed, dry)) = rx.recv() {
                if failure.is_some() {
                    continue;
                }
                let (mut lo, mut hi) = (0.0f32, 0.0f32);
                for &s in &processed {
                    lo = lo.min(s);
                    hi = hi.max(s);
                    if let Err(e) = main.write_sample(to_i24(s)) {
                        failure = Some(format!("запись на диск не удалась: {e}"));
                        break;
                    }
                }
                peaks.push((lo, hi));
                samples += processed.len() as u64;
                if let Some(w) = raw.as_mut() {
                    for &s in &dry {
                        let _ = w.write_sample(to_i24(s));
                    }
                }
            }
            let result = (|| -> Result<Take, String> {
                if let Some(f) = failure {
                    return Err(f);
                }
                main.finalize().map_err(|e| format!("не удалось закрыть файл: {e}"))?;
                if let Some(w) = raw {
                    w.finalize().map_err(|e| format!("не удалось закрыть файл: {e}"))?;
                }
                let mut note = None;
                let lufs = if opts.normalize {
                    match normalize_wav(&wav_path) {
                        Ok(l) => l,
                        Err(e) => {
                            note = Some(format!("нормализация не удалась: {e}"));
                            None
                        }
                    }
                } else {
                    measure_lufs(&wav_path).ok()
                };
                let mut final_path = wav_path.clone();
                let mut final_raw = opts.keep_raw.then(|| raw_wav.clone());
                if opts.format != Format::Wav {
                    match convert(&wav_path, opts.format) {
                        Ok(p) => final_path = p,
                        Err(e) => note = Some(format!("не удалось перекодировать, оставлен WAV: {e}")),
                    }
                    if opts.keep_raw {
                        if let Ok(p) = convert(&raw_wav, opts.format) {
                            final_raw = Some(p);
                        }
                    }
                }
                Ok(Take { path: final_path, raw_path: final_raw, secs: samples as f32 / FS as f32, peaks: downsample_peaks(&peaks), lufs, note })
            })();
            done(result);
        });
        Ok(Recorder { tx, handle, frames: 0 })
    }

    pub fn push(&mut self, processed: &[f32], dry: &[f32]) {
        self.frames += 1;
        let _ = self.tx.send((processed.to_vec(), dry.to_vec()));
    }

    /// Seconds recorded so far (counted in audio frames, so it follows the sound, not the clock).
    pub fn elapsed(&self) -> f32 {
        self.frames as f32 * HOP as f32 / FS as f32
    }

    /// End the take. The writer thread finishes the file and calls `done`; join the handle to wait.
    pub fn stop(self) -> thread::JoinHandle<()> {
        self.handle
    }
}

fn downsample_peaks(p: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if p.is_empty() {
        return Vec::new();
    }
    let bins = WAVE_BINS.min(p.len());
    (0..bins)
        .map(|b| {
            let (a, z) = (b * p.len() / bins, ((b + 1) * p.len() / bins).max(b * p.len() / bins + 1));
            p[a..z.min(p.len())].iter().fold((0.0f32, 0.0f32), |(lo, hi), &(l, h)| (lo.min(l), hi.max(h)))
        })
        .collect()
}

/// Integrated loudness (LUFS) and sample peak of a 24-bit mono WAV, streamed.
fn analyze_wav(path: &Path) -> Result<(f64, f64), String> {
    let mut r = hound::WavReader::open(path).map_err(|e| e.to_string())?;
    let mut meter = ebur128::EbuR128::new(1, r.spec().sample_rate, ebur128::Mode::I).map_err(|e| e.to_string())?;
    let mut buf = Vec::with_capacity(HOP * 100);
    let mut peak = 0.0f64;
    for s in r.samples::<i32>() {
        let v = s.map_err(|e| e.to_string())? as f32 / 8_388_608.0;
        peak = peak.max(v.abs() as f64);
        buf.push(v);
        if buf.len() == buf.capacity() {
            meter.add_frames_f32(&buf).map_err(|e| e.to_string())?;
            buf.clear();
        }
    }
    meter.add_frames_f32(&buf).map_err(|e| e.to_string())?;
    let l = meter.loudness_global().map_err(|e| e.to_string())?;
    Ok((l, peak))
}

pub fn measure_lufs(path: &Path) -> Result<f32, String> {
    let (l, _) = analyze_wav(path)?;
    if l.is_finite() { Ok(l as f32) } else { Err("тишина".into()) }
}

/// Gain (dB) that brings loudness `lufs` to the target without the sample peak passing the ceiling.
pub fn normalization_gain_db(lufs: f64, peak: f64) -> f64 {
    let want = TARGET_LUFS - lufs;
    let room = PEAK_CEILING_DB - 20.0 * peak.max(1e-9).log10();
    want.min(room).clamp(-24.0, 24.0)
}

/// Bring the file to the target loudness in place (streamed). Returns the loudness it ended at.
pub fn normalize_wav(path: &Path) -> Result<Option<f32>, String> {
    let (lufs, peak) = analyze_wav(path)?;
    if !lufs.is_finite() {
        return Ok(None); // silence: nothing to do
    }
    let gain_db = normalization_gain_db(lufs, peak);
    let g = 10f64.powf(gain_db / 20.0) as f32;
    let tmp = path.with_extension("normalizing.wav");
    {
        let mut r = hound::WavReader::open(path).map_err(|e| e.to_string())?;
        let mut w: Writer = hound::WavWriter::create(&tmp, r.spec()).map_err(|e| e.to_string())?;
        for s in r.samples::<i32>() {
            let v = s.map_err(|e| e.to_string())? as f32 / 8_388_608.0 * g;
            w.write_sample(to_i24(v)).map_err(|e| e.to_string())?;
        }
        w.finalize().map_err(|e| e.to_string())?;
    }
    fs::rename(&tmp, path).map_err(|e| e.to_string())?;
    Ok(Some((lufs + gain_db) as f32))
}

/// Convert a WAV to `format` with ffmpeg and remove the WAV; returns the new path.
fn convert(wav: &Path, format: Format) -> Result<PathBuf, String> {
    let out = wav.with_extension(format.ext());
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-i"])
        .arg(wav)
        .args(format.ffmpeg_args())
        .arg(&out)
        .stdin(Stdio::null())
        .status()
        .map_err(|e| format!("ffmpeg: {e}"))?;
    if !status.success() {
        let _ = fs::remove_file(&out);
        return Err("ffmpeg завершился с ошибкой".into());
    }
    let _ = fs::remove_file(wav);
    Ok(out)
}

/// The waveform of a take as SVG-style path commands in a `w` x `h` box, centred on `h / 2`.
pub fn wave_path(peaks: &[(f32, f32)], w: f32, h: f32) -> String {
    if peaks.is_empty() {
        return String::new();
    }
    let mut s = String::with_capacity(peaks.len() * 24);
    let mid = h / 2.0;
    let scale = (h / 2.0 - 1.0).max(1.0);
    let top = peaks.iter().fold(0.05f32, |a, &(l, hi)| a.max(hi).max(-l));
    for (i, &(lo, hi)) in peaks.iter().enumerate() {
        let x = (i as f32 + 0.5) * w / peaks.len() as f32;
        let y1 = mid - (hi / top).clamp(0.0, 1.0) * scale;
        let y2 = mid - (lo / top).clamp(-1.0, 0.0) * scale;
        s.push_str(&format!("M {x:.1} {y1:.1} L {x:.1} {:.1} ", y2.max(y1 + 1.0)));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::tests::{noise, tone};
    use std::sync::Mutex;

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("seiren-test-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    fn record(opts: RecordOpts, signal: &[f32]) -> Result<Take, String> {
        let result: Arc<Mutex<Option<Result<Take, String>>>> = Arc::new(Mutex::new(None));
        let r2 = result.clone();
        let mut rec = Recorder::start(opts, Arc::new(move |t| *r2.lock().unwrap() = Some(t)))?;
        for f in signal.chunks_exact(HOP) {
            rec.push(f, f);
        }
        let _ = rec.stop();
        for _ in 0..600 {
            if let Some(t) = result.lock().unwrap().take() {
                return t;
            }
            thread::sleep(std::time::Duration::from_millis(50));
        }
        Err("timeout".into())
    }

    #[test]
    fn a_take_is_written_with_the_right_length_and_a_waveform() {
        let dir = tmpdir("len");
        let x = tone(440.0, 0.5, 3 * FS as usize);
        let t = record(RecordOpts { dir: dir.clone(), format: Format::Wav, keep_raw: true, normalize: false }, &x).unwrap();
        assert!((t.secs - 3.0).abs() < 0.02, "{}", t.secs);
        let r = hound::WavReader::open(&t.path).unwrap();
        assert_eq!((r.spec().channels, r.spec().sample_rate, r.spec().bits_per_sample), (1, 48000, 24));
        assert_eq!(r.len() as usize, x.len());
        assert!(t.raw_path.as_ref().is_some_and(|p| p.exists()), "raw copy kept next to it");
        assert_eq!(t.peaks.len(), 300, "one slice per 10 ms frame while that is fewer than {WAVE_BINS}");
        assert!(t.peaks.iter().all(|&(lo, hi)| lo < -0.4 && hi > 0.4), "a sine fills every slice");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn normalizing_hits_the_target_loudness() {
        let dir = tmpdir("norm");
        let x: Vec<f32> = noise(6 * FS as usize, 3).iter().map(|v| v * 0.02).collect(); // quiet
        let before = {
            let t = record(RecordOpts { dir: dir.clone(), format: Format::Wav, keep_raw: false, normalize: false }, &x).unwrap();
            t.lufs.unwrap()
        };
        let t = record(RecordOpts { dir: dir.clone(), format: Format::Wav, keep_raw: false, normalize: true }, &x).unwrap();
        let after = measure_lufs(&t.path).unwrap();
        println!("loudness {before:.1} -> {after:.1} LUFS (target {TARGET_LUFS})");
        assert!(before < -30.0);
        // noise is peaky, so the peak guard may stop it short, but it must be far louder and never above
        assert!(after > before + 10.0 && after < -15.0, "{after}");
        let (_, peak) = analyze_wav(&t.path).unwrap();
        assert!(20.0 * peak.log10() <= PEAK_CEILING_DB + 0.1, "peak {} dBFS", 20.0 * peak.log10());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn normalization_gain_math() {
        assert_eq!(normalization_gain_db(-26.0, 0.05).round(), 10.0);
        // a loud file is turned down
        assert_eq!(normalization_gain_db(-10.0, 0.2).round(), -6.0);
        // headroom limits the boost: peak at -3 dBFS leaves only 2 dB
        assert!((normalization_gain_db(-30.0, 0.7079) - 2.0).abs() < 0.1);
    }

    #[test]
    fn converts_to_flac_when_ffmpeg_can() {
        if !available_formats().contains(&Format::Flac) {
            eprintln!("no ffmpeg flac encoder here, skipped");
            return;
        }
        let dir = tmpdir("flac");
        let x = tone(300.0, 0.3, FS as usize);
        let t = record(RecordOpts { dir: dir.clone(), format: Format::Flac, keep_raw: false, normalize: false }, &x).unwrap();
        assert_eq!(t.path.extension().unwrap(), "flac");
        assert!(t.path.metadata().unwrap().len() > 1000);
        assert!(!t.path.with_extension("wav").exists(), "the intermediate WAV is removed");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn waveform_path_is_drawable() {
        let s = wave_path(&[(-0.5, 0.5), (-1.0, 1.0)], 100.0, 40.0);
        assert!(s.starts_with("M ") && s.matches('M').count() == 2);
        assert_eq!(wave_path(&[], 100.0, 40.0), "");
    }

    #[test]
    fn a_bad_folder_is_reported_up_front() {
        let r = Recorder::start(RecordOpts { dir: PathBuf::from("/proc/nope/x"), format: Format::Wav, keep_raw: false, normalize: false }, Arc::new(|_| {}));
        assert!(r.is_err());
    }
}
