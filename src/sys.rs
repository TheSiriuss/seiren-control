//! Small facts about the machine this runs on: which helper programs exist, and the one-time udev rule
//! that lets the logged-in user open the microphone's control interface.
//!
//! The rule works on every init system. `uaccess` hands the device to the active session where
//! systemd-logind or elogind is running; the `GROUP`/`MODE` part covers the rest (runit, OpenRC,
//! Slackware's rc scripts, no polkit seat tracking). The group is one the user is already in, so the
//! access works at once, without logging in again.

use std::path::PathBuf;

pub const RULE_PATH: &str = "/etc/udev/rules.d/70-seiren-v3-chroma.rules";
const RULE_MATCH: &str = r#"SUBSYSTEM=="hidraw", ATTRS{idVendor}=="1532", ATTRS{idProduct}=="056f""#;
/// `$1` = the rule text, `$2` = the file. Run as root (through pkexec, or by hand).
pub const RULE_SCRIPT: &str = r#"printf '%s\n' "$1" > "$2" && udevadm control --reload && udevadm trigger --subsystem-match=hidraw --action=change"#;

/// The first executable called `name` on PATH.
pub fn which(name: &str) -> Option<PathBuf> {
    use std::os::unix::fs::PermissionsExt;
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(name)).find(|p| p.metadata().map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0).unwrap_or(false))
}

/// Parse /etc/group into (name, gid).
fn group_table(text: &str) -> Vec<(String, u32)> {
    text.lines()
        .filter_map(|l| {
            let mut f = l.split(':');
            let name = f.next()?;
            let gid = f.nth(1)?.parse().ok()?;
            Some((name.to_string(), gid))
        })
        .collect()
}

/// `plugdev` if the user is in it (Debian, Void, Slackware), else the user's own primary group.
fn pick_group(table: &[(String, u32)], mine: &[u32], primary: u32) -> Option<String> {
    let id_of = |n: &str| table.iter().find(|(name, _)| name == n).map(|(_, g)| *g);
    if let Some(g) = id_of("plugdev").filter(|g| mine.contains(g)) {
        return table.iter().find(|(_, id)| *id == g).map(|(n, _)| n.clone());
    }
    table.iter().find(|(_, id)| *id == primary).map(|(n, _)| n.clone())
}

pub fn user_group() -> Option<String> {
    let (mut mine, primary) = unsafe {
        let n = libc::getgroups(0, std::ptr::null_mut());
        let mut v = vec![0 as libc::gid_t; n.max(0) as usize];
        let n = libc::getgroups(v.len() as i32, v.as_mut_ptr());
        v.truncate(n.max(0) as usize);
        (v, libc::getgid())
    };
    mine.push(primary);
    pick_group(&group_table(&std::fs::read_to_string("/etc/group").ok()?), &mine, primary)
}

fn safe_group_name(g: &str) -> bool {
    !g.is_empty() && g.len() <= 32 && g.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

/// The udev rule for this machine.
pub fn rule_line(group: Option<&str>) -> String {
    match group.filter(|g| safe_group_name(g)) {
        Some(g) => format!(r#"{RULE_MATCH}, TAG+="uaccess", GROUP="{g}", MODE="0660""#),
        None => format!(r#"{RULE_MATCH}, TAG+="uaccess""#),
    }
}

/// What to type in a terminal when there is no pkexec (no polkit, or no authentication agent running).
pub fn grant_help() -> String {
    let rule = rule_line(user_group().as_deref());
    format!(
        "Доступ к управляющему интерфейсу микрофона выдаётся одним файлом правил udev.\n\
         Выполните от root (sudo или doas):\n\n\
         sudo tee {RULE_PATH} <<'EOF'\n{rule}\nEOF\n\
         sudo udevadm control --reload\n\
         sudo udevadm trigger --subsystem-match=hidraw --action=change\n\n\
         Отменить: удалить {RULE_PATH} и повторить две последние команды.\n"
    )
}

/// (program, what it is for, what happens without it)
pub const HELPERS: &[(&str, &str)] = &[
    ("parec", "запись с микрофона (PipeWire или PulseAudio)"),
    ("pacat", "воспроизведение: самопрослушивание, виртуальный микрофон"),
    ("pactl", "поиск устройств, запасной виртуальный микрофон на модулях PulseAudio"),
    ("pw-loopback", "виртуальный микрофон «Seiren Clean» на PipeWire (без него: модули PulseAudio через pactl)"),
    ("amixer", "усиление и самопрослушивание звуковой карты (alsa-utils)"),
    ("ffmpeg", "FLAC / Opus / MP3 (необязательно; WAV пишется всегда)"),
    ("ffplay", "прослушать дубль в окне (необязательно; иначе paplay)"),
    ("pkexec", "кнопка «Разрешить доступ» (иначе: seiren-control --grant-help)"),
    ("xdg-open", "кнопка «Открыть папку» (необязательно)"),
    ("xdg-user-dir", "папка «Музыка» из xdg-user-dirs (иначе ~/Music)"),
];

#[cfg(test)]
mod tests {
    use super::*;

    const GROUPS: &str = "root:x:0:\nplugdev:x:46:alice\nusers:x:100:\nalice:x:1000:\nbroken line\n";

    #[test]
    fn parses_etc_group() {
        let t = group_table(GROUPS);
        assert_eq!(t.len(), 4);
        assert!(t.contains(&("plugdev".to_string(), 46)));
    }

    #[test]
    fn prefers_plugdev_when_the_user_is_in_it() {
        let t = group_table(GROUPS);
        assert_eq!(pick_group(&t, &[46, 1000], 1000).as_deref(), Some("plugdev"));
    }

    #[test]
    fn falls_back_to_the_primary_group() {
        let t = group_table(GROUPS);
        // plugdev exists but the user is not in it (Arch has none at all): own group, access is immediate
        assert_eq!(pick_group(&t, &[1000], 1000).as_deref(), Some("alice"));
        assert_eq!(pick_group(&t[..1], &[1000], 1000), None);
    }

    #[test]
    fn rule_has_uaccess_and_a_group() {
        let r = rule_line(Some("plugdev"));
        assert!(r.contains(r#"idVendor}=="1532""#) && r.contains(r#"idProduct}=="056f""#));
        assert!(r.contains(r#"TAG+="uaccess""#));
        assert!(r.ends_with(r#"GROUP="plugdev", MODE="0660""#), "{r}");
    }

    #[test]
    fn rule_without_a_group_or_with_a_suspicious_one_is_uaccess_only() {
        assert!(!rule_line(None).contains("GROUP"));
        assert!(!rule_line(Some("a\"b")).contains("GROUP"));
        assert!(!rule_line(Some("")).contains("GROUP"));
    }

    #[test]
    fn which_finds_sh_and_not_nonsense() {
        assert!(which("sh").is_some());
        assert!(which("definitely-not-a-program-7f3a").is_none());
    }
}
