//! A live preview of the ring in the window, and the HSV maths of the colour picker.
//!
//! The preview is an approximation: the microphone cannot be asked how fast its own effects run,
//! so the firmware breathing and spectrum speeds here are a typical guess. The soft breath is
//! exact (the app drives it itself) and the wave follows the speed byte that is sent.

use crate::device::soft_level;
use crate::mic::Rgb;
use crate::settings::Settings;
use slint::Color;

/// Segments around the ring.
pub const N: usize = 48;

type F3 = [f32; 3];

const OFF: F3 = [0.07, 0.07, 0.10];

pub fn hsv_to_f(h: f32, s: f32, v: f32) -> F3 {
    let h = h.rem_euclid(1.0) * 6.0;
    let (i, f) = (h.floor(), h - h.floor());
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f)));
    match i as i32 % 6 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Rgb {
    let c = hsv_to_f(h, s.clamp(0.0, 1.0), v.clamp(0.0, 1.0));
    [(c[0] * 255.0).round() as u8, (c[1] * 255.0).round() as u8, (c[2] * 255.0).round() as u8]
}

pub fn rgb_to_hsv(c: Rgb) -> (f32, f32, f32) {
    let (r, g, b) = (c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0);
    let (max, min) = (r.max(g).max(b), r.min(g).min(b));
    let d = max - min;
    let h = if d == 0.0 {
        0.0
    } else if max == r {
        ((g - b) / d).rem_euclid(6.0) / 6.0
    } else if max == g {
        ((b - r) / d + 2.0) / 6.0
    } else {
        ((r - g) / d + 4.0) / 6.0
    };
    (h, if max == 0.0 { 0.0 } else { d / max }, max)
}

fn f_of(c: Rgb) -> F3 {
    [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0]
}

fn mix(a: F3, b: F3, t: f32) -> F3 {
    let t = t.clamp(0.0, 1.0);
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

fn to_color(c: F3) -> Color {
    Color::from_rgb_u8((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8)
}

fn phase(t: f32, period: f32) -> f32 {
    0.5 - 0.5 * (std::f32::consts::TAU * t / period).cos()
}

pub struct Preview {
    pub seg: [Color; N],
    /// The mean colour of the ring, for the glow behind the window.
    pub ambient: Color,
}

pub fn render(s: &Settings, t: f32) -> Preview {
    let bf = s.brightness as f32 / 255.0;
    let (c1, c2) = (f_of(s.c1()), f_of(s.c2()));
    let mut ring = [OFF; N];
    let lit = |c: F3, k: f32| mix(OFF, c, k * bf);
    match s.effect.as_str() {
        "off" => {}
        "static" => ring = [lit(c1, 1.0); N],
        "soft" => {
            let k = soft_level(t, s.soft_period, 255) as f32 / 255.0;
            ring = [lit(c1, k); N];
        }
        "breath1" => ring = [lit(c1, phase(t, 3.0)); N],
        "breath2" => {
            let c = if (t / 3.0) as i32 % 2 == 0 { c1 } else { c2 };
            ring = [lit(c, phase(t, 3.0)); N];
        }
        "breath0" => {
            let cycle = (t / 3.0) as u32;
            let hue = (cycle.wrapping_mul(2_654_435_761) >> 8) % 360;
            ring = [lit(hsv_to_f(hue as f32 / 360.0, 1.0, 1.0), phase(t, 3.0)); N];
        }
        "spectrum" => ring = [lit(hsv_to_f(t / 10.0, 1.0, 1.0), 1.0); N],
        "wave" | "wave_rev" => {
            let dir = if s.effect == "wave" { 1.0 } else { -1.0 };
            let period = s.wave_speed as f32 * 0.04;
            for (i, seg) in ring.iter_mut().enumerate() {
                *seg = lit(hsv_to_f(i as f32 / N as f32 + dir * t / period, 1.0, 1.0), 1.0);
            }
        }
        _ => {}
    }
    let mean = ring.iter().fold([0.0; 3], |a, c| [a[0] + c[0], a[1] + c[1], a[2] + c[2]]).map(|x| x / N as f32);
    Preview { seg: ring.map(to_color), ambient: to_color(mean) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hsv_round_trips() {
        for c in [[255, 0, 0], [0, 255, 64], [12, 34, 254], [255, 255, 255], [0, 0, 0], [139, 92, 246]] {
            let (h, s, v) = rgb_to_hsv(c);
            let back = hsv_to_rgb(h, s, v);
            for i in 0..3 {
                assert!((back[i] as i32 - c[i] as i32).abs() <= 1, "{c:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn soft_breath_in_the_preview_is_the_same_curve_the_mic_gets() {
        let s = Settings { effect: "soft".into(), color1: "ffffff".into(), soft_period: 10.0, ..Default::default() };
        let dark = render(&s, 0.0).seg[0];
        let bright = render(&s, 5.0).seg[0];
        assert!(bright.red() > 240 && dark.red() < 40, "{} {}", dark.red(), bright.red());
    }

    #[test]
    fn a_wave_moves_and_off_is_dark() {
        let s = Settings { effect: "wave".into(), ..Default::default() };
        assert_ne!(render(&s, 0.0).seg[0], render(&s, 1.0).seg[0]);
        let off = Settings { effect: "off".into(), ..Default::default() };
        assert!(render(&off, 0.0).seg.iter().all(|c| c.red() < 30));
    }
}
