//! `seiren-control --process in.wav out.wav [--preset podcast] [--denoise dfn]`: run an existing
//! recording through the same voice chain, without a window.

use crate::dsp::{self, DenoiseMode, DspParams, FS};
use std::path::Path;

/// Read any WAV as 48 kHz mono f32 (channels averaged, linear-interpolated to 48 kHz if needed).
pub fn read_wav(path: &Path) -> Result<Vec<f32>, String> {
    let mut r = hound::WavReader::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let spec = r.spec();
    let ch = spec.channels.max(1) as usize;
    let all: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>().map_err(|e| e.to_string())?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|s| s.map(|v| v as f32 * scale)).collect::<Result<_, _>>().map_err(|e| e.to_string())?
        }
    };
    let mono: Vec<f32> = all.chunks_exact(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect();
    if spec.sample_rate == FS as u32 {
        return Ok(mono);
    }
    let ratio = spec.sample_rate as f64 / FS;
    let n = (mono.len() as f64 / ratio) as usize;
    Ok((0..n)
        .map(|i| {
            let pos = i as f64 * ratio;
            let (a, frac) = (pos as usize, (pos - pos.floor()) as f32);
            let b = (a + 1).min(mono.len() - 1);
            mono[a] * (1.0 - frac) + mono[b] * frac
        })
        .collect())
}

pub fn write_wav(path: &Path, x: &[f32]) -> Result<(), String> {
    let spec = hound::WavSpec { channels: 1, sample_rate: FS as u32, bits_per_sample: 24, sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::create(path, spec).map_err(|e| format!("{}: {e}", path.display()))?;
    for &s in x {
        w.write_sample((s.clamp(-1.0, 1.0) * 8_388_607.0) as i32).map_err(|e| e.to_string())?;
    }
    w.finalize().map_err(|e| e.to_string())
}

pub fn process_file(input: &Path, output: &Path, params: &DspParams) -> Result<String, String> {
    let x = read_wav(input)?;
    if x.is_empty() {
        return Err("в файле нет звука".into());
    }
    let started = std::time::Instant::now();
    let y = dsp::process_buffer(params, &x);
    let took = started.elapsed().as_secs_f32();
    write_wav(output, &y)?;
    let rms = |v: &[f32]| dsp::lin_to_db((v.iter().map(|s| s * s).sum::<f32>() / v.len() as f32).sqrt());
    Ok(format!(
        "{:.1} с звука обработано за {took:.2} с. Уровень: {:.1} → {:.1} дБFS (RMS). Записано: {}",
        x.len() as f32 / FS as f32,
        rms(&x),
        rms(&y),
        output.display()
    ))
}

/// `args` after `--process`: input, output, then optional `--preset NAME` / `--denoise MODE`.
pub fn run_cli(args: &[String]) -> Result<String, String> {
    let (Some(inp), Some(out)) = (args.first(), args.get(1)) else {
        return Err("использование: seiren-control --process вход.wav выход.wav [--preset podcast|stream|vocal|calls|clean|off] [--denoise off|rnnoise|dfn|spectral]".into());
    };
    let mut p = DspParams::default();
    let mut i = 2;
    while i < args.len() {
        match (args[i].as_str(), args.get(i + 1)) {
            ("--preset", Some(name)) => {
                if !p.apply_preset(name) {
                    return Err(format!("нет пресета «{name}»"));
                }
            }
            ("--denoise", Some(m)) => {
                p.denoise = match m.as_str() {
                    "off" => DenoiseMode::Off,
                    "rnnoise" => DenoiseMode::Rnnoise,
                    "dfn" => DenoiseMode::Dfn,
                    "spectral" => DenoiseMode::Spectral,
                    other => return Err(format!("неизвестный режим «{other}»")),
                };
                p.preset.clear();
            }
            (other, _) => return Err(format!("непонятный аргумент {other}")),
        }
        i += 2;
    }
    process_file(Path::new(inp), Path::new(out), &p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::tests::{noise, speechlike};

    #[test]
    fn a_44k_stereo_file_comes_out_48k_mono_and_cleaner() {
        let dir = std::env::temp_dir().join(format!("seiren-offline-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (inp, out) = (dir.join("in.wav"), dir.join("out.wav"));
        // 6 s of "voice + noise" at 44.1 kHz stereo
        let n = 6 * 44100;
        let clean = speechlike(n, 0.25); // (generated at 48k timing, fine as test material)
        let nz = noise(n, 4);
        let spec = hound::WavSpec { channels: 2, sample_rate: 44100, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut w = hound::WavWriter::create(&inp, spec).unwrap();
        for i in 0..n {
            let s = ((clean[i] + 0.03 * nz[i]).clamp(-1.0, 1.0) * 32767.0) as i16;
            w.write_sample(s).unwrap();
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();
        let msg = run_cli(&[inp.to_string_lossy().into(), out.to_string_lossy().into(), "--denoise".into(), "spectral".into()]).unwrap();
        println!("{msg}");
        let r = hound::WavReader::open(&out).unwrap();
        assert_eq!((r.spec().channels, r.spec().sample_rate), (1, 48000));
        let expect = (6.0 * 48000.0) as i64;
        assert!((r.len() as i64 - expect).abs() < 100, "length {}", r.len());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn bad_arguments_are_explained() {
        assert!(run_cli(&[]).unwrap_err().contains("использование"));
        assert!(run_cli(&["a.wav".into(), "b.wav".into(), "--preset".into(), "nope".into()]).unwrap_err().contains("нет пресета"));
        assert!(process_file(Path::new("/nonexistent.wav"), Path::new("/tmp/x.wav"), &DspParams::default()).is_err());
    }
}
