//! What the window shows and sends, kept in ~/.config/seiren-control/settings.json.
//! The microphone cannot report its own effect or colour, so these are the last values sent.

use crate::mic::{
    Rgb, EFFECT_BREATHING, EFFECT_NONE, EFFECT_SPECTRUM, EFFECT_STATIC, EFFECT_WAVE,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// (id, label). The ids are what is stored on disk and passed between the UI and the worker.
pub const EFFECTS: &[(&str, &str)] = &[
    ("static", "Статика"),
    ("breath1", "Дыхание"),
    ("breath2", "Дыхание ×2"),
    ("soft", "Плавное дыхание"),
    ("breath0", "Дыхание: случайно"),
    ("spectrum", "Спектр"),
    ("wave", "Волна"),
    ("wave_rev", "Волна назад"),
    ("off", "Выключено"),
];

/// 5x slower than the 0x28 the first version called "slow" (the byte is a delay: bigger is slower).
pub const WAVE_SPEED_DEFAULT: u8 = 0xC8;
pub const WAVE_SPEED_FASTEST: u8 = 0x10;
pub const WAVE_SPEED_SLOWEST: u8 = 0xFF;
pub const SOFT_PERIOD_MIN: f32 = 2.0;
pub const SOFT_PERIOD_MAX: f32 = 30.0;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub effect: String,
    pub color1: String,
    pub color2: String,
    pub mute_color: String,
    pub brightness: u8,
    pub wave_speed: u8,
    /// Seconds per breath of the "soft" effect.
    pub soft_period: f32,
    pub live: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            effect: "static".into(),
            color1: "8b5cf6".into(),
            color2: "22d3ee".into(),
            mute_color: "ff0000".into(),
            brightness: 255,
            wave_speed: WAVE_SPEED_DEFAULT,
            soft_period: 10.0,
            live: true,
        }
    }
}

pub fn hex_rgb(h: &str) -> Rgb {
    let h = h.trim_start_matches('#');
    let p = |i: usize| h.get(i..i + 2).and_then(|s| u8::from_str_radix(s, 16).ok()).unwrap_or(255);
    if h.len() == 6 { [p(0), p(2), p(4)] } else { [255, 255, 255] }
}

pub fn rgb_hex(c: Rgb) -> String {
    format!("{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

impl Settings {
    pub fn sanitize(&mut self) {
        if !EFFECTS.iter().any(|(id, _)| *id == self.effect) {
            self.effect = "static".into();
        }
        for c in [&mut self.color1, &mut self.color2, &mut self.mute_color] {
            *c = rgb_hex(hex_rgb(c));
        }
        self.wave_speed = self.wave_speed.max(WAVE_SPEED_FASTEST);
        self.soft_period = if self.soft_period.is_finite() {
            self.soft_period.clamp(SOFT_PERIOD_MIN, SOFT_PERIOD_MAX)
        } else {
            10.0
        };
    }

    pub fn c1(&self) -> Rgb {
        hex_rgb(&self.color1)
    }
    pub fn c2(&self) -> Rgb {
        hex_rgb(&self.color2)
    }

    /// The firmware effect behind the chosen one: (effect id, colours, reverse, speed byte).
    /// "soft" is a static colour whose brightness the app moves itself.
    pub fn firmware_effect(&self) -> (u8, Vec<Rgb>, bool, u8) {
        let speed = self.wave_speed;
        match self.effect.as_str() {
            "off" => (EFFECT_NONE, vec![], false, speed),
            "static" | "soft" => (EFFECT_STATIC, vec![self.c1()], false, speed),
            "breath0" => (EFFECT_BREATHING, vec![], false, speed),
            "breath1" => (EFFECT_BREATHING, vec![self.c1()], false, speed),
            "breath2" => (EFFECT_BREATHING, vec![self.c1(), self.c2()], false, speed),
            "spectrum" => (EFFECT_SPECTRUM, vec![], false, speed),
            "wave" => (EFFECT_WAVE, vec![], false, speed),
            "wave_rev" => (EFFECT_WAVE, vec![], true, speed),
            _ => (EFFECT_STATIC, vec![self.c1()], false, speed),
        }
    }

    /// The mute colour to send. The firmware darkens the muted ring when its colour is exactly the
    /// static colour of the normal ring, so in that case it goes out one step off (blue + 1).
    /// Returns the colour and whether it was moved.
    pub fn muted_wire_color(&self) -> (Rgb, bool) {
        let mine = hex_rgb(&self.mute_color);
        let static_like = matches!(self.effect.as_str(), "static" | "soft");
        if static_like && self.c1() == mine {
            let b = if mine[2] == 0xFF { 0xFE } else { mine[2] + 1 };
            ([mine[0], mine[1], b], true)
        } else {
            (mine, false)
        }
    }
}

pub fn dir() -> PathBuf {
    std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("seiren-control")
}

pub fn path() -> PathBuf {
    dir().join("settings.json")
}

/// Settings from disk; the first run also picks up the colours of the earlier (Python) version.
/// The bool says whether a settings file existed.
pub fn load() -> (Settings, bool) {
    if let Ok(text) = fs::read_to_string(path()) {
        if let Ok(mut s) = serde_json::from_str::<Settings>(&text) {
            s.sanitize();
            return (s, true);
        }
    }
    let mut s = Settings::default();
    let mut existed = false;
    if let Ok(text) = fs::read_to_string(dir().join("state.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
            existed = true;
            let get = |k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_string);
            if let Some(c) = get("color1") { s.color1 = c; }
            if let Some(c) = get("color2") { s.color2 = c; }
            if let Some(c) = get("mute_color") { s.mute_color = c; }
            if let Some(e) = get("effect") { s.effect = e; }
            if let Some(b) = v.get("brightness").and_then(|x| x.as_u64()) { s.brightness = b.min(255) as u8; }
            if let Some(l) = v.get("live").and_then(|x| x.as_bool()) { s.live = l; }
        }
    }
    s.sanitize();
    (s, existed)
}

pub fn save(s: &Settings) {
    let p = path();
    if let Some(d) = p.parent() {
        let _ = fs::create_dir_all(d);
    }
    if let Ok(text) = serde_json::to_string_pretty(s) {
        let _ = fs::write(p, text);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colours_round_trip_and_bad_input_falls_back() {
        assert_eq!(hex_rgb("#00ff40"), [0, 255, 64]);
        assert_eq!(rgb_hex([0, 255, 64]), "00ff40");
        assert_eq!(hex_rgb("nonsense"), [255, 255, 255]);
    }

    #[test]
    fn mute_colour_equal_to_the_static_colour_is_moved_one_step() {
        let mut s = Settings { color1: "00ff40".into(), mute_color: "00ff40".into(), ..Default::default() };
        assert_eq!(s.muted_wire_color(), ([0, 255, 65], true));
        s.color1 = "0000ff".into();
        s.mute_color = "0000ff".into();
        assert_eq!(s.muted_wire_color(), ([0, 0, 254], true));
        s.effect = "spectrum".into();
        assert_eq!(s.muted_wire_color(), ([0, 0, 255], false));
    }

    #[test]
    fn sanitize_clamps_everything() {
        let mut s = Settings { effect: "nope".into(), wave_speed: 3, soft_period: f32::NAN, color1: "zz".into(), ..Default::default() };
        s.sanitize();
        assert_eq!((s.effect.as_str(), s.wave_speed, s.soft_period, s.color1.as_str()), ("static", 0x10, 10.0, "ffffff"));
    }
}
