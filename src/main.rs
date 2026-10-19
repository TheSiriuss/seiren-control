//! Seiren Control: a control panel for the Razer Seiren V3 Chroma on Linux.
//! One binary, no daemon, no service. Unofficial: not affiliated with Razer.
//!
//! The lighting protocol was reverse engineered by Christoffer Hallas for periferi
//! (https://github.com/csfh/periferi, MIT licence); see mic.rs.

mod audio;
mod audio_ui;
mod device;
mod dsp;
mod engine;
mod offline;
mod recorder;
mod fx;
mod launcher;
mod mic;
mod mouth;
mod settings;
mod sys;

use audio::{AudioCmd, Ctl, Val, Values};
use device::{Cmd, ConnState, Event, Kind};
use mic::{Backend, DemoLog, Mic, MicError, Rgb, STORE_FLASH, STORE_NONE};
use settings::{hex_rgb, rgb_hex, Settings};
use slint::{Color, ComponentHandle, Model, TimerMode, VecModel};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

slint::include_modules!();

const HELP: &str = "\
seiren-control: control panel for the Razer Seiren V3 Chroma

  seiren-control             open the window
  seiren-control --status    print what it can see (changes nothing)
  seiren-control --demo      run against a pretend microphone (touches no hardware)
  seiren-control --process in.wav out.wav [--preset podcast] [--denoise dfn]
                             clean up an existing recording with the voice chain
  seiren-control --install-menu    add it to the application menu (with an icon); --uninstall-menu removes it
  seiren-control --grant-help      the terminal commands for the one-time access rule (no polkit / no pkexec)
  seiren-control --help

Unofficial; not affiliated with Razer. Protocol: https://github.com/csfh/periferi (MIT).
";

// The one-time permission rule (no daemon) lives in sys.rs: the kernel only lets root open /dev/hidraw*.

fn color(c: Rgb) -> Color {
    Color::from_rgb_u8(c[0], c[1], c[2])
}

fn rgb_of(c: Color) -> Rgb {
    [c.red(), c.green(), c.blue()]
}

// ---- slider <-> value mappings (the single source of truth; the .slint only draws) ------------

fn speed_to_slider(b: u8) -> f32 {
    (settings::WAVE_SPEED_SLOWEST - b) as f32 / (settings::WAVE_SPEED_SLOWEST - settings::WAVE_SPEED_FASTEST) as f32
}

fn slider_to_speed(v: f32) -> u8 {
    let span = (settings::WAVE_SPEED_SLOWEST - settings::WAVE_SPEED_FASTEST) as f32;
    (settings::WAVE_SPEED_SLOWEST as f32 - v.clamp(0.0, 1.0) * span).round() as u8
}

/// Relative to the 0x28 the first version called "slow".
fn speed_text(b: u8) -> String {
    let r = b as f32 / 40.0;
    if r >= 1.0 { format!("медленнее ×{r:.1}") } else { format!("быстрее ×{:.1}", 1.0 / r) }
}

fn period_to_slider(t: f32) -> f32 {
    (t / settings::SOFT_PERIOD_MIN).ln() / (settings::SOFT_PERIOD_MAX / settings::SOFT_PERIOD_MIN).ln()
}

fn slider_to_period(v: f32) -> f32 {
    let t = settings::SOFT_PERIOD_MIN * (settings::SOFT_PERIOD_MAX / settings::SOFT_PERIOD_MIN).powf(v.clamp(0.0, 1.0));
    (t * 2.0).round() / 2.0
}

fn period_text(t: f32) -> String {
    format!("{t:.1} с")
}

// ---- shared state ----------------------------------------------------------------------------

#[derive(Clone)]
struct Ctx {
    settings: Arc<Mutex<Settings>>,
    tx: Sender<Cmd>,
    persist: bool,
}

impl Ctx {
    fn edit(&self, f: impl FnOnce(&mut Settings)) -> Settings {
        let snapshot = {
            let mut g = self.settings.lock().unwrap();
            f(&mut g);
            g.sanitize();
            g.clone()
        };
        if self.persist {
            settings::save(&snapshot);
        }
        snapshot
    }

    fn live(&self, s: &Settings, kind: Kind) {
        if s.live {
            let _ = self.tx.send(Cmd::Light(s.clone(), kind));
        }
    }
}

fn slot_rgb(s: &Settings, slot: i32) -> Rgb {
    match slot {
        0 => s.c1(),
        1 => s.c2(),
        _ => hex_rgb(&s.mute_color),
    }
}

fn show_hsv(ui: &AppWindow, c: Rgb) {
    let (h, s, v) = fx::rgb_to_hsv(c);
    ui.set_hue(h);
    ui.set_sat(s);
    ui.set_val(v);
    ui.set_hue_color(color(fx::hsv_to_rgb(h, 1.0, 1.0)));
}

fn push_all(ui: &AppWindow, s: &Settings) {
    ui.set_effect(s.effect.clone().into());
    ui.set_c1(color(s.c1()));
    ui.set_c2(color(s.c2()));
    ui.set_cm(color(hex_rgb(&s.mute_color)));
    ui.set_brightness(s.brightness as f32 / 255.0);
    ui.set_speed(speed_to_slider(s.wave_speed));
    ui.set_speed_text(speed_text(s.wave_speed).into());
    ui.set_period(period_to_slider(s.soft_period));
    ui.set_period_text(period_text(s.soft_period).into());
    ui.set_live(s.live);
    show_hsv(ui, slot_rgb(s, ui.get_slot()));
}

thread_local! {
    static TOAST: slint::Timer = slint::Timer::default();
}

pub(crate) fn toast(ui: &AppWindow, text: &str) {
    ui.set_info(text.into());
    let w = ui.as_weak();
    TOAST.with(|t| {
        t.start(TimerMode::SingleShot, Duration::from_millis(4500), move || {
            if let Some(u) = w.upgrade() {
                u.set_info("".into());
            }
        })
    });
}

fn handle_event(ui: &AppWindow, settings: &Arc<Mutex<Settings>>, persist: bool, ev: Event) {
    match ev {
        Event::Conn { state, firmware, text } => {
            let (code, msg) = match state {
                ConnState::Ok => (1, format!("Прошивка {}", firmware.clone().unwrap_or_default())),
                ConnState::Missing => (2, "Микрофон не найден. Подключите его по USB.".to_string()),
                ConnState::NoAccess => (3, "Ядро не даёт открыть управляющий интерфейс. Это нужно разрешить один раз.".to_string()),
                ConnState::Error => (4, text),
            };
            ui.set_conn(code);
            ui.set_status_text(msg.into());
            if let Some(fw) = firmware {
                ui.set_fw(fw.into());
            }
        }
        Event::Brightness(b) => {
            let s = {
                let mut g = settings.lock().unwrap();
                g.brightness = b;
                g.clone()
            };
            if persist {
                settings::save(&s);
            }
            ui.set_brightness(b as f32 / 255.0);
        }
        Event::Info(t) | Event::Error(t) => toast(ui, &t),
    }
}

fn apply_audio(ui: &AppWindow, v: Values) {
    ui.set_audio_ok(v.gain.is_some() || v.monitor.is_some() || v.headphones.is_some());
    if ui.get_audio_drag() {
        return;
    }
    if let Some(p) = v.gain { ui.set_gain(p as f32 / 100.0) }
    if let Some(p) = v.monitor { ui.set_mon_vol(p as f32 / 100.0) }
    if let Some(p) = v.headphones { ui.set_phones(p as f32 / 100.0) }
    if let Some(b) = v.monitor_on { ui.set_mon_on(b) }
}

/// `effect=wave,speed=60,period=8,brightness=200,c1=ff0000,c2=00ff00,mute=0000ff` (screenshots).
fn apply_preview(s: &mut Settings, spec: &str) {
    for kv in spec.split(',') {
        let Some((k, v)) = kv.split_once('=') else { continue };
        match k {
            "effect" => s.effect = v.into(),
            "speed" => s.wave_speed = v.parse().unwrap_or(s.wave_speed),
            "period" => s.soft_period = v.parse().unwrap_or(s.soft_period),
            "brightness" => s.brightness = v.parse().unwrap_or(s.brightness),
            "c1" => s.color1 = v.into(),
            "c2" => s.color2 = v.into(),
            "mute" => s.mute_color = v.into(),
            _ => {}
        }
    }
    s.sanitize();
}

fn write_ppm(path: &str, buf: &slint::SharedPixelBuffer<slint::Rgba8Pixel>) -> std::io::Result<()> {
    let mut out = format!("P6\n{} {}\n255\n", buf.width(), buf.height()).into_bytes();
    for px in buf.as_bytes().chunks_exact(4) {
        out.extend_from_slice(&px[..3]);
    }
    std::fs::write(path, out)
}

/// winit creates the real window only once the event loop is running, so the icon is set on the first
/// tick that finds it (the taskbar shows it even before there is a menu entry).
fn set_window_icon(weak: slint::Weak<AppWindow>, attempt: u32) {
    use slint::winit_030::{winit, WinitWindowAccessor};
    let Some(ui) = weak.upgrade() else { return };
    let icon = winit::window::Icon::from_rgba(launcher::WINDOW_ICON_RGBA.to_vec(), launcher::WINDOW_ICON_SIZE, launcher::WINDOW_ICON_SIZE).ok();
    let done = icon.is_some_and(|icon| ui.window().with_winit_window(|w| w.set_window_icon(Some(icon))).is_some());
    if !done && attempt < 40 {
        slint::Timer::single_shot(Duration::from_millis(50), move || set_window_icon(weak, attempt + 1));
    }
}

fn run(demo: bool, shot: Option<String>, preview: Option<String>, page: Option<i32>, size: Option<(f32, f32)>) -> Result<(), Box<dyn std::error::Error>> {
    let _ = slint::set_xdg_app_id(launcher::APP_ID);
    let ui = AppWindow::new()?;
    let persist = !demo;
    let (mut s, existed) = if demo { (Settings::default(), true) } else { settings::load() };
    if let Some(p) = &preview {
        apply_preview(&mut s, p);
    }
    let settings = Arc::new(Mutex::new(s.clone()));

    // ---- workers
    let backend = if demo { Backend::Demo(DemoLog::default()) } else { Backend::Hid };
    let emit: device::Emit = {
        let (weak, settings) = (ui.as_weak(), settings.clone());
        Arc::new(move |ev| {
            let settings = settings.clone();
            let _ = weak.upgrade_in_event_loop(move |ui| handle_event(&ui, &settings, persist, ev));
        })
    };
    let (tx, dev_thread) = device::spawn(backend, emit, !existed);
    let ctx = Ctx { settings: settings.clone(), tx: tx.clone(), persist };

    let drag = Arc::new(AtomicBool::new(false));
    let audio_workers = audio::Audio::new(demo).map(|a| {
        let weak = ui.as_weak();
        audio::spawn(
            a,
            Arc::new(move |v| {
                let _ = weak.upgrade_in_event_loop(move |ui| apply_audio(&ui, v));
            }),
            drag.clone(),
        )
    });
    let atx: Option<Sender<AudioCmd>> = audio_workers.as_ref().map(|(t, _)| t.clone());

    push_all(&ui, &s);

    // ---- the recording page
    let audio = audio_ui::AudioApp::new(&ui, demo);
    if let Some(p) = page {
        ui.set_page(p);
        audio.set_page(p);
    }
    if let Some((w, h)) = size {
        ui.window().set_size(slint::LogicalSize::new(w, h));
    }

    // ---- window -> device
    {
        let (c, w) = (ctx.clone(), ui.as_weak());
        ui.on_set_effect(move |id| {
            let s = c.edit(|s| s.effect = id.to_string());
            if let Some(ui) = w.upgrade() {
                ui.set_effect(id);
            }
            c.live(&s, Kind::Full);
        });
    }
    {
        let w = ui.as_weak();
        let c = ctx.clone();
        ui.on_slot_selected(move |slot| {
            if let Some(ui) = w.upgrade() {
                ui.set_slot(slot);
                show_hsv(&ui, slot_rgb(&c.settings.lock().unwrap(), slot));
            }
        });
    }
    // a colour was chosen for the selected slot (by the sliders or a preset)
    let set_slot_color = {
        let (c, w) = (ctx.clone(), ui.as_weak());
        Rc::new(move |rgb: Rgb| {
            let Some(ui) = w.upgrade() else { return };
            let slot = ui.get_slot();
            let s = c.edit(|s| match slot {
                0 => s.color1 = rgb_hex(rgb),
                1 => s.color2 = rgb_hex(rgb),
                _ => s.mute_color = rgb_hex(rgb),
            });
            ui.set_c1(color(s.c1()));
            ui.set_c2(color(s.c2()));
            ui.set_cm(color(hex_rgb(&s.mute_color)));
            if slot == 2 {
                if s.live {
                    let _ = c.tx.send(Cmd::Mute(s));
                }
            } else {
                c.live(&s, Kind::Full);
            }
        })
    };
    {
        let (w, set) = (ui.as_weak(), set_slot_color.clone());
        ui.on_hsv_edited(move |h, s, v| {
            if let Some(ui) = w.upgrade() {
                ui.set_hue_color(color(fx::hsv_to_rgb(h, 1.0, 1.0)));
            }
            set(fx::hsv_to_rgb(h, s, v));
        });
    }
    {
        let (w, set) = (ui.as_weak(), set_slot_color.clone());
        ui.on_preset(move |col| {
            let rgb = rgb_of(col);
            if let Some(ui) = w.upgrade() {
                show_hsv(&ui, rgb);
            }
            set(rgb);
        });
    }
    {
        let c = ctx.clone();
        ui.on_brightness_edited(move |v| {
            let s = c.edit(|s| s.brightness = (v.clamp(0.0, 1.0) * 255.0).round() as u8);
            c.live(&s, Kind::Brightness);
        });
    }
    {
        let (c, w) = (ctx.clone(), ui.as_weak());
        ui.on_speed_edited(move |v| {
            let s = c.edit(|s| s.wave_speed = slider_to_speed(v));
            if let Some(ui) = w.upgrade() {
                ui.set_speed_text(speed_text(s.wave_speed).into());
            }
            c.live(&s, Kind::Full);
        });
    }
    {
        let (c, w) = (ctx.clone(), ui.as_weak());
        ui.on_period_edited(move |v| {
            let s = c.edit(|s| s.soft_period = slider_to_period(v));
            if let Some(ui) = w.upgrade() {
                ui.set_period_text(period_text(s.soft_period).into());
            }
            c.live(&s, Kind::Brightness);
        });
    }
    {
        let c = ctx.clone();
        ui.on_live_toggled(move |on| {
            c.edit(|s| s.live = on);
        });
    }
    {
        let c = ctx.clone();
        ui.on_apply(move || {
            let s = c.settings.lock().unwrap().clone();
            let _ = c.tx.send(Cmd::Apply(s, STORE_NONE));
        });
    }
    {
        let c = ctx.clone();
        ui.on_save(move || {
            let s = c.settings.lock().unwrap().clone();
            let _ = c.tx.send(Cmd::Apply(s, STORE_FLASH));
        });
    }
    {
        let tx = tx.clone();
        ui.on_retry(move || {
            let _ = tx.send(Cmd::Connect);
        });
    }
    {
        let (tx, w) = (tx.clone(), ui.as_weak());
        ui.on_grant_confirm(move || {
            let (tx, w) = (tx.clone(), w.clone());
            std::thread::spawn(move || {
                let rc = if sys::which("pkexec").is_some() {
                    std::process::Command::new("pkexec")
                        .args(["sh", "-c", sys::RULE_SCRIPT, "sh", &sys::rule_line(sys::user_group().as_deref()), sys::RULE_PATH])
                        .status()
                        .map(|s| s.code().unwrap_or(-1))
                        .unwrap_or(127)
                } else {
                    127
                };
                let _ = w.upgrade_in_event_loop(move |ui| {
                    if rc == 0 {
                        toast(&ui, "Доступ выдан.");
                        let _ = tx.send(Cmd::Connect);
                    } else if rc == 127 {
                        toast(&ui, "Нет pkexec или окна пароля (polkit). Команды для терминала: seiren-control --grant-help");
                    } else {
                        toast(&ui, &format!("Доступ не выдан (код {rc}). Можно повторить."));
                    }
                });
            });
        });
    }

    // ---- window -> sound card
    if let Some(atx) = atx {
        let send = move |ctl: Ctl, val: Val| {
            let _ = atx.send(AudioCmd::Set(ctl, val));
        };
        let pct = |v: f32| Val::Pct((v.clamp(0.0, 1.0) * 100.0).round() as u8);
        let s1 = send.clone();
        ui.on_gain_edited(move |v| s1(Ctl::Gain, pct(v)));
        let s2 = send.clone();
        ui.on_mon_vol_edited(move |v| s2(Ctl::Monitor, pct(v)));
        let s3 = send.clone();
        ui.on_phones_edited(move |v| s3(Ctl::Headphones, pct(v)));
        ui.on_mon_toggled(move |on| send(Ctl::MonitorOn, Val::On(on)));
    }

    // ---- the ring preview, ~50 frames a second; only changed segments are touched
    let model = Rc::new(VecModel::from(vec![Color::from_rgb_u8(18, 18, 26); fx::N]));
    ui.set_seg(model.clone().into());
    let t0 = Instant::now();
    let frame_timer = Rc::new(slint::Timer::default());
    {
        let (w, settings, drag, audio) = (ui.as_weak(), settings.clone(), drag.clone(), audio.clone());
        frame_timer.start(TimerMode::Repeated, Duration::from_millis(33), move || {
            let Some(ui) = w.upgrade() else { return };
            if ui.get_page() == 1 {
                // the recording page: no ring preview (it is not on screen)
                audio.tick(&ui);
                return;
            }
            audio.tick(&ui);
            let s = settings.lock().unwrap().clone();
            let p = fx::render(&s, t0.elapsed().as_secs_f32());
            for (i, c) in p.seg.iter().enumerate() {
                if model.row_data(i) != Some(*c) {
                    model.set_row_data(i, *c);
                }
            }
            if ui.get_ambient() != p.ambient {
                ui.set_ambient(p.ambient);
            }
            drag.store(ui.get_audio_drag(), Ordering::Relaxed);
        });
    }

    // lighting animates the ring at 30 fps; the meters of the recording page are fine at 20 and
    // every frame costs a few milliseconds of drawing
    let interval = |page: i32| Duration::from_millis(if page == 1 { 50 } else { 33 });
    {
        let (a, t) = (audio.clone(), frame_timer.clone());
        ui.on_page_changed(move |p| {
            a.set_page(p);
            t.set_interval(interval(p));
        });
    }
    if let Some(p) = page {
        frame_timer.set_interval(interval(p));
    }

    if let Some(path) = shot {
        let w = ui.as_weak();
        slint::Timer::single_shot(Duration::from_millis(1400), move || {
            if let Some(ui) = w.upgrade() {
                match ui.window().take_snapshot() {
                    Ok(buf) => {
                        if let Err(e) = write_ppm(&path, &buf) {
                            eprintln!("seiren-control: cannot write {path}: {e}");
                        }
                    }
                    Err(e) => eprintln!("seiren-control: snapshot failed: {e}"),
                }
            }
            let _ = slint::quit_event_loop();
        });
    }

    set_window_icon(ui.as_weak(), 0);
    ui.run()?;

    audio.shutdown();
    // closing: let the worker put the ring back where it was (a soft breath may be mid-dim)
    let _ = tx.send(Cmd::Quit);
    let _ = dev_thread.join();
    if let Some((atx, h)) = audio_workers {
        let _ = atx.send(AudioCmd::Quit);
        let _ = h.join();
    }
    Ok(())
}

fn print_status() {
    match mic::find_node() {
        Some(n) => println!("управляющий интерфейс: {n}"),
        None => println!("управляющий интерфейс: не найден (микрофон подключён?)"),
    }
    let mut m = Mic::new(Backend::Hid);
    match m.open() {
        Ok(()) => {
            println!("доступ: есть");
            match m.firmware() {
                Ok(fw) => println!("прошивка: {fw}"),
                Err(e) => println!("прошивка: {e}"),
            }
            match m.brightness(STORE_FLASH) {
                Ok(b) => println!("яркость в памяти микрофона: {b}/255"),
                Err(e) => println!("яркость: {e}"),
            }
        }
        Err(MicError::NoAccess(_)) => println!("доступ: ЗАПРЕЩЁН (откройте окно и нажмите «Разрешить доступ…»)"),
        Err(MicError::NotFound) => println!("доступ: -"),
        Err(e) => println!("ошибка: {e}"),
    }
    match audio::Audio::new(false) {
        Some(a) => {
            let v = a.all();
            println!("звуковая карта: есть");
            println!("  усиление: {:?}%  самопрослушивание: {:?}  громкость его: {:?}%  наушники: {:?}%", v.gain, v.monitor_on, v.monitor, v.headphones);
        }
        None => println!("звуковая карта: не найдена"),
    }
    println!("звуковой сервер: {}", engine::server_name().unwrap_or_else(|| "не отвечает (нужен PipeWire с pipewire-pulse или PulseAudio)".into()));
    println!("программы:");
    for (name, what) in sys::HELPERS {
        println!("  {:<13} {:<5} {what}", name, if sys::which(name).is_some() { "есть" } else { "нет" });
    }
}

/// For the command-line modes: die quietly when the reader goes away (`... | head`), like any Unix
/// tool. The window keeps Rust's default (SIGPIPE ignored) so a dead `pacat` cannot kill it.
fn quiet_pipes() {
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--install-menu" || a == "--uninstall-menu") {
        let r = if args.iter().any(|a| a == "--install-menu") { launcher::install() } else { launcher::uninstall() };
        match r {
            Ok(msg) => println!("{msg}"),
            Err(e) => {
                eprintln!("seiren-control: {e}");
                std::process::exit(1);
            }
        }
        return;
    }
    if args.iter().any(|a| a == "--help" || a == "-h" || a == "--status" || a == "--process" || a == "--grant-help") {
        quiet_pipes();
    }
    if args.iter().any(|a| a == "--grant-help") {
        print!("{}", sys::grant_help());
        return;
    }
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print!("{HELP}");
        return;
    }
    if args.iter().any(|a| a == "--status") {
        print_status();
        return;
    }
    if let Some(i) = args.iter().position(|a| a == "--process") {
        match offline::run_cli(&args[i + 1..]) {
            Ok(msg) => println!("{msg}"),
            Err(e) => {
                eprintln!("seiren-control: {e}");
                std::process::exit(1);
            }
        }
        return;
    }
    let value = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
    let shot = value("--screenshot");
    // a screenshot never touches the real devices
    let demo = shot.is_some() || args.iter().any(|a| a == "--demo");
    if let Err(e) = run(
        demo,
        shot,
        value("--preview"),
        value("--page").and_then(|p| p.parse().ok()),
        value("--size").and_then(|s| s.split_once('x').and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))),
    ) {
        eprintln!("seiren-control: {e}");
        std::process::exit(1);
    }
}
