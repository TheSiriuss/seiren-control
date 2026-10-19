//! The microphone's sound card controls (capture gain, self-monitoring, headphones) through `amixer`.
//! No root needed: the user's `audio` group already owns the card.

use std::collections::HashMap;
use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::mic::{PID, VID};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ctl {
    Gain,
    Monitor,
    MonitorOn,
    Headphones,
}

impl Ctl {
    fn name(self) -> &'static str {
        match self {
            Ctl::Gain => "Mic Capture Volume",
            Ctl::Monitor => "Mic Playback Volume",
            Ctl::MonitorOn => "Mic Playback Switch",
            Ctl::Headphones => "Speaker Playback Volume",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Val {
    Pct(u8),
    On(bool),
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Values {
    pub gain: Option<u8>,
    pub monitor: Option<u8>,
    pub monitor_on: Option<bool>,
    pub headphones: Option<u8>,
}

pub fn pct_to_raw(pct: u8, lo: i64, hi: i64) -> i64 {
    lo + ((hi - lo) * pct.min(100) as i64 + 50) / 100
}

pub fn raw_to_pct(raw: i64, lo: i64, hi: i64) -> u8 {
    if hi <= lo {
        return 0;
    }
    let raw = raw.clamp(lo, hi);
    (((raw - lo) * 100 + (hi - lo) / 2) / (hi - lo)) as u8
}

/// Move all channels by the same amount, so a balance the user set on purpose stays.
pub fn shift_levels(old: &[i64], new0: i64, lo: i64, hi: i64) -> Vec<i64> {
    let delta = old.first().map_or(0, |o| new0 - o);
    old.iter().map(|v| (v + delta).clamp(lo, hi)).collect()
}

#[derive(Debug, PartialEq)]
struct Ctrl {
    min: i64,
    max: i64,
    boolean: bool,
    values: Vec<i64>,
}

/// Parse `amixer cget` output (run with LC_ALL=C).
fn parse_cget(out: &str) -> Option<Ctrl> {
    let head = out.lines().find(|l| l.trim_start().starts_with("; type="))?;
    let field = |k: &str| head.split(',').find_map(|p| p.trim().trim_start_matches("; ").strip_prefix(k)).map(str::to_string);
    let kind = field("type=")?;
    let vals = out.lines().find_map(|l| l.trim_start().strip_prefix(": values="))?;
    match kind.as_str() {
        "BOOLEAN" => Some(Ctrl {
            min: 0,
            max: 1,
            boolean: true,
            values: vals.split(',').map(|v| (v.trim() == "on") as i64).collect(),
        }),
        "INTEGER" => Some(Ctrl {
            min: field("min=")?.parse().ok()?,
            max: field("max=")?.parse().ok()?,
            boolean: false,
            values: vals.split(',').filter_map(|v| v.trim().parse().ok()).collect(),
        }),
        _ => None,
    }
}

/// ALSA card index of the microphone, found by its USB id.
pub fn find_card() -> Option<u32> {
    let want = format!("{:04x}:{:04x}", VID, PID);
    (0..32u32).find(|n| {
        fs::read_to_string(format!("/proc/asound/card{n}/usbid")).map(|s| s.trim().to_lowercase() == want).unwrap_or(false)
    })
}

pub enum Audio {
    Alsa(u32),
    /// Keeps values in memory and touches nothing.
    Demo(Mutex<HashMap<Ctl, i64>>),
}

fn amixer(card: u32, args: &[&str]) -> Option<String> {
    let out = Command::new("amixer").env("LC_ALL", "C").arg("-c").arg(card.to_string()).args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

impl Audio {
    pub fn new(demo: bool) -> Option<Audio> {
        if demo {
            let m = HashMap::from([(Ctl::Gain, 72), (Ctl::Monitor, 40), (Ctl::MonitorOn, 0), (Ctl::Headphones, 64)]);
            return Some(Audio::Demo(Mutex::new(m)));
        }
        find_card().map(Audio::Alsa)
    }

    fn read(card: u32, ctl: Ctl) -> Option<Ctrl> {
        parse_cget(&amixer(card, &["cget", &format!("name={}", ctl.name())])?)
    }

    pub fn get(&self, ctl: Ctl) -> Option<Val> {
        match self {
            Audio::Demo(m) => {
                let v = *m.lock().unwrap().get(&ctl)?;
                Some(if ctl == Ctl::MonitorOn { Val::On(v != 0) } else { Val::Pct(v as u8) })
            }
            Audio::Alsa(card) => {
                let c = Self::read(*card, ctl)?;
                let first = *c.values.first()?;
                Some(if c.boolean { Val::On(first != 0) } else { Val::Pct(raw_to_pct(first, c.min, c.max)) })
            }
        }
    }

    pub fn set(&self, ctl: Ctl, val: Val) -> bool {
        match self {
            Audio::Demo(m) => {
                m.lock().unwrap().insert(ctl, match val { Val::Pct(p) => p as i64, Val::On(b) => b as i64 });
                true
            }
            Audio::Alsa(card) => {
                let Some(c) = Self::read(*card, ctl) else { return false };
                let arg = match val {
                    Val::On(b) => vec![if b { "on" } else { "off" }; c.values.len().max(1)].join(","),
                    Val::Pct(p) => shift_levels(&c.values, pct_to_raw(p, c.min, c.max), c.min, c.max)
                        .iter()
                        .map(i64::to_string)
                        .collect::<Vec<_>>()
                        .join(","),
                };
                amixer(*card, &["cset", &format!("name={}", ctl.name()), &arg]).is_some()
            }
        }
    }

    pub fn all(&self) -> Values {
        let pct = |c| match self.get(c) { Some(Val::Pct(p)) => Some(p), _ => None };
        Values {
            gain: pct(Ctl::Gain),
            monitor: pct(Ctl::Monitor),
            monitor_on: match self.get(Ctl::MonitorOn) { Some(Val::On(b)) => Some(b), _ => None },
            headphones: pct(Ctl::Headphones),
        }
    }
}

pub enum AudioCmd {
    Set(Ctl, Val),
    Quit,
}

/// A thread that applies sound changes (newest value of each control wins) and re-reads the card
/// every few seconds, unless the user is dragging a slider (`busy`).
pub fn spawn(audio: Audio, emit: Arc<dyn Fn(Values) + Send + Sync>, busy: Arc<AtomicBool>) -> (Sender<AudioCmd>, JoinHandle<()>) {
    let (tx, rx) = channel::<AudioCmd>();
    let h = thread::spawn(move || {
        emit(audio.all());
        loop {
            match rx.recv_timeout(Duration::from_secs(4)) {
                Ok(first) => {
                    let mut pending: Vec<(Ctl, Val)> = Vec::new();
                    let mut quit = false;
                    let mut take = |c: AudioCmd| match c {
                        AudioCmd::Set(k, v) => {
                            pending.retain(|(pk, _)| *pk != k);
                            pending.push((k, v));
                        }
                        AudioCmd::Quit => quit = true,
                    };
                    take(first);
                    while let Ok(n) = rx.try_recv() {
                        take(n);
                    }
                    for (k, v) in pending {
                        audio.set(k, v);
                    }
                    if quit {
                        return;
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    if !busy.load(Ordering::Relaxed) {
                        emit(audio.all());
                    }
                }
                Err(RecvTimeoutError::Disconnected) => return,
            }
        }
    });
    (tx, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_volume_and_a_switch() {
        let vol = "numid=5,iface=MIXER,name='Mic Capture Volume'\n  ; type=INTEGER,access=rw---R--,values=2,min=0,max=522,step=0\n  : values=522,261\n  | dBminmax-min=-17.62dB,max=15.00dB\n";
        assert_eq!(parse_cget(vol), Some(Ctrl { min: 0, max: 522, boolean: false, values: vec![522, 261] }));
        let sw = "numid=7,iface=MIXER,name='Mic Playback Switch'\n  ; type=BOOLEAN,access=rw------,values=1\n  : values=off\n";
        assert_eq!(parse_cget(sw), Some(Ctrl { min: 0, max: 1, boolean: true, values: vec![0] }));
        assert_eq!(parse_cget("garbage"), None);
    }

    #[test]
    fn percent_maths_keeps_the_channel_balance() {
        assert_eq!((pct_to_raw(100, 0, 522), pct_to_raw(0, 0, 522), pct_to_raw(50, 0, 522)), (522, 0, 261));
        assert_eq!((raw_to_pct(522, 0, 522), raw_to_pct(261, 0, 522)), (100, 50));
        assert_eq!(shift_levels(&[10, 14], 20, 0, 29), vec![20, 24]);
        assert_eq!(shift_levels(&[28, 29], 29, 0, 29), vec![29, 29]);
    }

    #[test]
    fn demo_card_stores_values() {
        let a = Audio::new(true).unwrap();
        assert!(a.set(Ctl::Gain, Val::Pct(33)));
        assert_eq!(a.all().gain, Some(33));
        assert_eq!(a.all().monitor_on, Some(false));
    }
}
