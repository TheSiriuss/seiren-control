//! The Razer Seiren V3 Chroma control protocol: one numbered 64-byte HID feature report
//! (report id 7, vendor page 0xFF53) spoken over /dev/hidraw.
//!
//!   byte 0 report id | 1 status | 2 transaction | 6 data size | 7 class | 8 command | 9.. arguments
//!   request: byte 63 = XOR of bytes 1..62        reply: byte 62 = XOR of bytes 2..61
//!
//! The protocol was reverse engineered by Christoffer Hallas for periferi
//! (https://github.com/csfh/periferi, MIT). This is an independent re-implementation of the
//! parts used here; frames are checked byte for byte against periferi's C encoder in the tests.

use std::collections::VecDeque;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::sync::{Arc, Mutex};
use std::thread::sleep;
use std::time::Duration;

pub const VID: u16 = 0x1532;
pub const PID: u16 = 0x056F;
pub const REPORT_LEN: usize = 64;
const REPORT_ID: u8 = 0x07;
const TRANSACTION: u8 = 0x52;
const REQUEST_ARGS_MAX: usize = 54;
const REPLY_ARGS_MAX: usize = 53;

pub const ST_BUSY: u8 = 1;
pub const ST_OK: u8 = 2;
pub const ST_FAILURE: u8 = 3;
pub const ST_TIMEOUT: u8 = 4;
pub const ST_NOT_SUPPORTED: u8 = 5;

pub const EFFECT_NONE: u8 = 0;
pub const EFFECT_STATIC: u8 = 1;
pub const EFFECT_BREATHING: u8 = 2;
pub const EFFECT_SPECTRUM: u8 = 3;
pub const EFFECT_WAVE: u8 = 4;

/// "Now only": lost when the microphone is replugged.
pub const STORE_NONE: u8 = 0;
/// Kept in the microphone's own memory across a replug.
pub const STORE_FLASH: u8 = 1;
/// The zone number of the ring colour shown while the microphone is muted.
pub const ZONE_MUTE: u8 = 0x19;

pub type Rgb = [u8; 3];
pub type Frame = [u8; REPORT_LEN];

#[derive(Debug, Clone, PartialEq)]
pub enum MicError {
    NotFound,
    NoAccess(String),
    Disconnected,
    Io(String),
    Protocol(String),
    Device(String),
}

impl fmt::Display for MicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MicError::NotFound => write!(f, "микрофон не найден"),
            MicError::NoAccess(n) => write!(f, "нет доступа к {n}"),
            MicError::Disconnected => write!(f, "микрофон отключён"),
            MicError::Io(e) => write!(f, "ошибка обмена: {e}"),
            MicError::Protocol(e) => write!(f, "ошибка протокола: {e}"),
            MicError::Device(e) => write!(f, "микрофон ответил: {e}"),
        }
    }
}

impl std::error::Error for MicError {}

fn xor(data: &[u8], lo: usize, hi: usize) -> u8 {
    data[lo..=hi].iter().fold(0, |a, b| a ^ b)
}

/// Build a request frame. `size` is the declared data size when it differs from `args.len()`
/// (queries declare a size but carry zeros).
pub fn encode(cls: u8, cmd: u8, args: &[u8], size: Option<usize>) -> Result<Frame, MicError> {
    let n = size.unwrap_or(args.len());
    if n > REQUEST_ARGS_MAX || args.len() > n {
        return Err(MicError::Protocol("слишком много аргументов для одного кадра".into()));
    }
    let mut f = [0u8; REPORT_LEN];
    f[0] = REPORT_ID;
    f[2] = TRANSACTION;
    f[6] = n as u8;
    f[7] = cls;
    f[8] = cmd;
    f[9..9 + args.len()].copy_from_slice(args);
    f[63] = xor(&f, 1, 62);
    Ok(f)
}

#[derive(Debug, PartialEq)]
pub enum DecodeError {
    /// The reply answers another command (an older answer still in the queue).
    Mismatch,
    Bad(String),
}

/// Check a reply against the request it answers: `(status, argument bytes)`.
pub fn decode(f: &Frame, cls: u8, cmd: u8) -> Result<(u8, Vec<u8>), DecodeError> {
    if f[0] != REPORT_ID {
        return Err(DecodeError::Bad("это не ответ Seiren".into()));
    }
    if f[62] != xor(f, 2, 61) {
        return Err(DecodeError::Bad("неверная контрольная сумма ответа".into()));
    }
    if f[7] != cls || f[8] != cmd {
        return Err(DecodeError::Mismatch);
    }
    if f[6] as usize > REPLY_ARGS_MAX {
        return Err(DecodeError::Bad("ответ слишком длинный".into()));
    }
    Ok((f[1], f[9..9 + f[6] as usize].to_vec()))
}

/// Arguments of the effect command (class 0x0f, command 0x02).
pub fn effect_args(
    storage: u8,
    effect: u8,
    colors: &[Rgb],
    reverse: bool,
    speed: u8,
    zone: u8,
) -> Result<Vec<u8>, MicError> {
    let mut a = [0u8; 12];
    a[0] = storage;
    a[1] = zone;
    a[2] = effect;
    let n = colors.len();
    if n > 2 {
        return Err(MicError::Protocol("не больше двух цветов".into()));
    }
    let len = match effect {
        EFFECT_NONE | EFFECT_SPECTRUM => 6,
        EFFECT_WAVE => {
            a[3] = reverse as u8;
            a[4] = if speed == 0 { 0x28 } else { speed };
            6
        }
        EFFECT_STATIC => {
            if n < 1 {
                return Err(MicError::Protocol("статике нужен цвет".into()));
            }
            a[5] = 1;
            a[6..9].copy_from_slice(&colors[0]);
            9
        }
        EFFECT_BREATHING => {
            if n == 0 {
                6 // random colours
            } else {
                a[3] = n as u8;
                a[5] = n as u8;
                a[6..9].copy_from_slice(&colors[0]);
                if n == 2 {
                    a[9..12].copy_from_slice(&colors[1]);
                    12
                } else {
                    9
                }
            }
        }
        _ => return Err(MicError::Protocol("неизвестный эффект".into())),
    };
    Ok(a[..len].to_vec())
}

// ------------------------------------------------------------------------------------------------
// Transports
// ------------------------------------------------------------------------------------------------

pub trait Transport: Send {
    fn send(&mut self, f: &Frame) -> Result<(), MicError>;
    /// `Ok(None)`: nothing (complete) to read yet.
    fn recv(&mut self) -> Result<Option<Frame>, MicError>;
}

const fn hid_ioctl(nr: u64) -> u64 {
    // _IOC(_IOC_READ | _IOC_WRITE, 'H', nr, REPORT_LEN)
    (3 << 30) | ((REPORT_LEN as u64) << 16) | ((b'H' as u64) << 8) | nr
}
const HIDIOCSFEATURE: u64 = hid_ioctl(0x06);
const HIDIOCGFEATURE: u64 = hid_ioctl(0x07);

struct Hidraw {
    file: File,
}

impl Transport for Hidraw {
    fn send(&mut self, f: &Frame) -> Result<(), MicError> {
        let rc = unsafe { libc::ioctl(self.file.as_raw_fd(), HIDIOCSFEATURE as _, f.as_ptr()) };
        if rc < 0 {
            let e = io::Error::last_os_error();
            return Err(if e.raw_os_error() == Some(libc::ENODEV) {
                MicError::Disconnected
            } else {
                MicError::Io(e.to_string())
            });
        }
        Ok(())
    }

    fn recv(&mut self) -> Result<Option<Frame>, MicError> {
        let mut buf = [0u8; REPORT_LEN];
        buf[0] = REPORT_ID;
        let rc = unsafe { libc::ioctl(self.file.as_raw_fd(), HIDIOCGFEATURE as _, buf.as_mut_ptr()) };
        if rc < 0 {
            let e = io::Error::last_os_error();
            return if e.raw_os_error() == Some(libc::ENODEV) { Err(MicError::Disconnected) } else { Ok(None) };
        }
        Ok(if rc as usize == REPORT_LEN { Some(buf) } else { None })
    }
}

/// Every request a demo microphone has received.
#[derive(Clone, Default)]
pub struct DemoLog(pub Arc<Mutex<Vec<Frame>>>);

#[cfg(test)]
impl DemoLog {
    /// (class, command, arguments) of every request so far.
    pub fn commands(&self) -> Vec<(u8, u8, Vec<u8>)> {
        self.0.lock().unwrap().iter().map(|f| (f[7], f[8], f[9..9 + f[6] as usize].to_vec())).collect()
    }
    pub fn clear(&self) {
        self.0.lock().unwrap().clear();
    }
}

/// A microphone that answers like the real one and writes nothing anywhere.
struct Demo {
    log: DemoLog,
    queue: VecDeque<Frame>,
}

impl Transport for Demo {
    fn send(&mut self, f: &Frame) -> Result<(), MicError> {
        self.log.0.lock().unwrap().push(*f);
        let (cls, cmd) = (f[7], f[8]);
        let mut r = [0u8; REPORT_LEN];
        r[0] = REPORT_ID;
        r[1] = ST_OK;
        r[2] = TRANSACTION;
        r[7] = cls;
        r[8] = cmd;
        match (cls, cmd) {
            (0x00, 0x81) => {
                r[6] = 2;
                r[9] = 1;
                r[10] = 3;
            }
            (0x0F, 0x84) => {
                r[6] = 3;
                r[9] = f[9];
                r[11] = 0x66;
            }
            _ => {}
        }
        r[62] = xor(&r, 2, 61);
        self.queue.push_back(r);
        Ok(())
    }

    fn recv(&mut self) -> Result<Option<Frame>, MicError> {
        Ok(self.queue.pop_front())
    }
}

#[derive(Clone)]
pub enum Backend {
    Hid,
    Demo(DemoLog),
}

/// /dev/hidrawN of the microphone's control interface (vendor page 0xFF53), if plugged in.
pub fn find_node() -> Option<String> {
    let want = format!("HID_ID=0003:{:08X}:{:08X}", VID, PID);
    let mut dirs: Vec<_> = fs::read_dir("/sys/class/hidraw").ok()?.flatten().collect();
    dirs.sort_by_key(|d| d.file_name());
    for d in dirs {
        let dev = d.path().join("device");
        let Ok(uevent) = fs::read_to_string(dev.join("uevent")) else { continue };
        if !uevent.to_uppercase().contains(&want) {
            continue;
        }
        let Ok(desc) = fs::read(dev.join("report_descriptor")) else { continue };
        if desc.windows(5).any(|w| w == [0x06, 0x53, 0xFF, 0x09, 0x04]) {
            return Some(format!("/dev/{}", d.file_name().to_string_lossy()));
        }
    }
    None
}

pub struct Mic {
    backend: Backend,
    t: Option<Box<dyn Transport>>,
}

impl Mic {
    pub fn new(backend: Backend) -> Self {
        Mic { backend, t: None }
    }

    pub fn is_open(&self) -> bool {
        self.t.is_some()
    }

    pub fn close(&mut self) {
        self.t = None;
    }

    pub fn open(&mut self) -> Result<(), MicError> {
        self.close();
        if let Backend::Demo(log) = &self.backend {
            self.t = Some(Box::new(Demo { log: log.clone(), queue: VecDeque::new() }));
            return Ok(());
        }
        let node = find_node().ok_or(MicError::NotFound)?;
        match OpenOptions::new().read(true).write(true).custom_flags(libc::O_CLOEXEC).open(&node) {
            Ok(file) => {
                self.t = Some(Box::new(Hidraw { file }));
                Ok(())
            }
            Err(e) if e.kind() == io::ErrorKind::PermissionDenied => Err(MicError::NoAccess(node)),
            Err(e) => Err(MicError::Io(format!("{node}: {e}"))),
        }
    }

    fn call(&mut self, cls: u8, cmd: u8, args: &[u8], size: Option<usize>) -> Result<(u8, Vec<u8>), MicError> {
        let frame = encode(cls, cmd, args, size)?;
        let t = self.t.as_mut().ok_or(MicError::Disconnected)?;
        if let Err(e) = t.send(&frame) {
            self.close();
            return Err(e);
        }
        for _ in 0..50 {
            sleep(Duration::from_millis(20));
            let raw = match self.t.as_mut().ok_or(MicError::Disconnected)?.recv() {
                Ok(Some(r)) => r,
                Ok(None) => continue,
                Err(e) => {
                    self.close();
                    return Err(e);
                }
            };
            match decode(&raw, cls, cmd) {
                Ok((ST_BUSY, _)) | Err(DecodeError::Mismatch) => continue,
                Ok(reply) => return Ok(reply),
                Err(DecodeError::Bad(why)) => return Err(MicError::Protocol(why)),
            }
        }
        Err(MicError::Device("не ответил".into()))
    }

    fn run(&mut self, cls: u8, cmd: u8, args: &[u8], size: Option<usize>, want: usize) -> Result<Vec<u8>, MicError> {
        let (status, data) = self.call(cls, cmd, args, size)?;
        if status != ST_OK || data.len() < want {
            let why = match status {
                ST_FAILURE => "ошибка".to_string(),
                ST_TIMEOUT => "таймаут".to_string(),
                ST_NOT_SUPPORTED => "не поддерживается".to_string(),
                ST_OK => "слишком короткий ответ".to_string(),
                s => format!("статус 0x{s:02x}"),
            };
            return Err(MicError::Device(why));
        }
        Ok(data)
    }

    pub fn firmware(&mut self) -> Result<String, MicError> {
        let d = self.run(0x00, 0x81, &[], Some(2), 2)?;
        Ok(format!("{}.{}", d[0], d[1]))
    }

    pub fn brightness(&mut self, storage: u8) -> Result<u8, MicError> {
        Ok(self.run(0x0F, 0x84, &[storage, 0, 0], Some(3), 3)?[2])
    }

    pub fn set_brightness(&mut self, storage: u8, value: u8) -> Result<(), MicError> {
        self.run(0x0F, 0x04, &[storage, 0, value], None, 0).map(|_| ())
    }

    pub fn set_effect(&mut self, storage: u8, effect: u8, colors: &[Rgb], reverse: bool, speed: u8) -> Result<(), MicError> {
        let args = effect_args(storage, effect, colors, reverse, speed, 0)?;
        self.run(0x0F, 0x02, &args, None, 0).map(|_| ())
    }

    pub fn set_mute_color(&mut self, storage: u8, rgb: Rgb) -> Result<(), MicError> {
        let args = effect_args(storage, EFFECT_STATIC, &[rgb], false, 0, ZONE_MUTE)?;
        self.run(0x0F, 0x02, &args, None, 0).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(f: &[u8]) -> String {
        f.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Every frame periferi's own C encoder produces (tests/c_reference_frames.txt, generated from
    /// periferi's razer.c) must come out of this implementation byte for byte.
    #[test]
    fn matches_the_c_reference_encoder() {
        let mut checked = 0;
        for line in include_str!("../tests/c_reference_frames.txt").lines() {
            let tok: Vec<&str> = line.split_whitespace().collect();
            let kv = |k: &str| -> u32 {
                tok.iter().find_map(|t| t.strip_prefix(&format!("{k}="))).unwrap().parse().unwrap()
            };
            let frame_hex = tok.last().unwrap();
            let want = if *frame_hex == "REJECT" { None } else { Some(*frame_hex) };
            let c1: Rgb = [0x00, 0xFF, 0x40];
            let c2: Rgb = [0x12, 0x34, 0xFE];
            // brightness lines exist once per storage; the storage is byte 9 of the C frame itself,
            // everything else (size, class, command, value, checksum) is recomputed here
            let storage_of_line =
                if frame_hex.len() >= 20 { u8::from_str_radix(&frame_hex[18..20], 16).unwrap_or(0) } else { 0 };
            let got: Result<Frame, MicError> = match tok[0] {
                "fw" => encode(0, 0x81, &[], Some(2)),
                "bright" => encode(0x0F, 0x04, &[storage_of_line, 0, 0xB7], None),
                "brightq" => encode(0x0F, 0x84, &[storage_of_line, 0, 0], Some(3)),
                "eff" => {
                    let (st, e, c, r, fast) = (kv("st") as u8, kv("e") as u8, kv("c") as usize, kv("r"), kv("f"));
                    let speed = if fast == 1 { 0x10 } else { 0x28 };
                    let colors: Vec<Rgb> = [c1, c2][..c].to_vec();
                    effect_args(st, e, &colors, r == 1, speed, 0).and_then(|a| encode(0x0F, 0x02, &a, None))
                }
                "mute" => {
                    let st = kv("st") as u8;
                    effect_args(st, EFFECT_STATIC, &[c1], false, 0, ZONE_MUTE).and_then(|a| encode(0x0F, 0x02, &a, None))
                }
                other => panic!("unknown line kind {other}"),
            };
            match (want, got) {
                (None, Err(_)) => {}
                (Some(w), Ok(g)) => assert_eq!(hex(&g), w, "{line}"),
                (w, g) => panic!("{line}: expected {w:?}, got {g:?}"),
            }
            checked += 1;
        }
        assert!(checked >= 100, "only {checked} frames checked");
    }

    #[test]
    fn decodes_real_replies_from_the_owners_microphone() {
        // firmware get reply captured from a real Seiren V3 Chroma
        let mut f = [0u8; REPORT_LEN];
        f[..11].copy_from_slice(&[7, 2, 0x52, 0, 0, 0, 2, 0, 0x81, 1, 1]);
        f[62] = 0xD1;
        assert_eq!(decode(&f, 0x00, 0x81), Ok((ST_OK, vec![1, 1])));
        // brightness reply: the argument byte is covered by the checksum
        let mut b = [0u8; REPORT_LEN];
        b[..12].copy_from_slice(&[7, 2, 0x52, 0, 0, 0, 3, 0x0F, 0x84, 1, 0, 0x66]);
        b[62] = 0xBD;
        assert_eq!(decode(&b, 0x0F, 0x84), Ok((ST_OK, vec![1, 0, 0x66])));
        b[11] = 0x67;
        assert!(matches!(decode(&b, 0x0F, 0x84), Err(DecodeError::Bad(_))));
        assert_eq!(decode(&f, 0x0F, 0x84), Err(DecodeError::Mismatch));
    }

    #[test]
    fn request_checksums_from_the_c_tests() {
        assert_eq!(encode(0, 0x81, &[], Some(2)).unwrap()[63], 0xD1);
        assert_eq!(encode(0x0F, 0x04, &[1, 4, 0xB7], None).unwrap()[63], 0xE8);
    }

    #[test]
    fn demo_microphone_round_trip() {
        let log = DemoLog::default();
        let mut m = Mic::new(Backend::Demo(log.clone()));
        m.open().unwrap();
        assert_eq!(m.firmware().unwrap(), "1.3");
        assert_eq!(m.brightness(STORE_FLASH).unwrap(), 0x66);
        m.set_effect(STORE_NONE, EFFECT_WAVE, &[], false, 0xC8).unwrap();
        assert_eq!(log.commands().last().unwrap().2, vec![0, 0, 4, 0, 0xC8, 0]);
    }
}
