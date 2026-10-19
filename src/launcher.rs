//! `seiren-control --install-menu` / `--uninstall-menu`: put the app into the desktop's application
//! menu (and its taskbar icon). Two small files in the user's own XDG data folder: no root, nothing
//! running in the background; the launcher simply starts this very binary.

use std::path::{Path, PathBuf};

const ICON_SVG: &str = include_str!("../assets/seiren-control.svg");
/// Also the X11 window class, so the taskbar groups the window under this launcher.
pub const APP_ID: &str = "seiren-control";

fn data_home() -> Result<PathBuf, String> {
    if let Some(d) = std::env::var_os("XDG_DATA_HOME").filter(|d| !d.is_empty()) {
        return Ok(PathBuf::from(d));
    }
    std::env::var_os("HOME").filter(|h| !h.is_empty()).map(|h| PathBuf::from(h).join(".local/share")).ok_or_else(|| "не задана переменная HOME".to_string())
}

fn desktop_path(data: &Path) -> PathBuf {
    data.join("applications").join(format!("{APP_ID}.desktop"))
}

fn icon_path(data: &Path) -> PathBuf {
    data.join("icons/hicolor/scalable/apps").join(format!("{APP_ID}.svg"))
}

/// A path as the `Exec` value of a desktop entry: quoted, with the escapes the specification asks
/// for (the quoting rule first, then the general string rule, which doubles every backslash).
fn exec_arg(p: &Path) -> String {
    let mut out = String::from("\"");
    for c in p.to_string_lossy().chars() {
        match c {
            '%' => out.push_str("%%"),
            '"' | '`' | '$' => {
                out.push_str("\\\\");
                out.push(c);
            }
            '\\' => out.push_str("\\\\\\\\"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

fn desktop_entry(exe: &Path) -> String {
    format!(
        "[Desktop Entry]\n\
         Type=Application\n\
         Version=1.0\n\
         Name=Seiren Control\n\
         GenericName=Panel for the Razer Seiren V3 Chroma\n\
         GenericName[ru]=Панель микрофона Razer Seiren\n\
         Comment=Lighting, voice recording and noise suppression for the Razer Seiren V3 Chroma\n\
         Comment[ru]=Подсветка, запись голоса и шумоподавление для Razer Seiren V3 Chroma\n\
         Exec={}\n\
         Icon={APP_ID}\n\
         Terminal=false\n\
         Categories=AudioVideo;Audio;Recorder;\n\
         Keywords=razer;seiren;microphone;rgb;микрофон;запись;шумоподавление;подсветка;\n\
         StartupWMClass={APP_ID}\n",
        exec_arg(exe)
    )
}

fn write(path: &Path, text: &str) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

fn install_into(data: &Path, exe: &Path) -> Result<String, String> {
    let (desktop, icon) = (desktop_path(data), icon_path(data));
    write(&icon, ICON_SVG)?;
    write(&desktop, &desktop_entry(exe))?;
    // the menu notices new files by itself; this only refreshes the MIME cache where the tool exists
    let _ = std::process::Command::new("update-desktop-database").arg(data.join("applications")).output();
    Ok(format!("Добавлено в меню приложений («Мультимедиа»):\n  {}\n  {}\nУбрать: seiren-control --uninstall-menu", desktop.display(), icon.display()))
}

fn uninstall_from(data: &Path) -> Result<String, String> {
    let mut removed = Vec::new();
    for p in [desktop_path(data), icon_path(data)] {
        match std::fs::remove_file(&p) {
            Ok(()) => removed.push(p.display().to_string()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(format!("{}: {e}", p.display())),
        }
    }
    let _ = std::process::Command::new("update-desktop-database").arg(data.join("applications")).output();
    Ok(if removed.is_empty() { "В меню его и не было.".to_string() } else { format!("Убрано из меню:\n  {}", removed.join("\n  ")) })
}

pub fn install() -> Result<String, String> {
    let exe = std::env::current_exe().map_err(|e| format!("не найти собственный путь: {e}"))?;
    install_into(&data_home()?, &exe)
}

pub fn uninstall() -> Result<String, String> {
    uninstall_from(&data_home()?)
}

/// The icon as 128 x 128 RGBA, for the window itself (the taskbar uses it before any menu entry).
pub const WINDOW_ICON_RGBA: &[u8] = include_bytes!("../assets/icon-128.rgba");
pub const WINDOW_ICON_SIZE: u32 = 128;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exec_paths_are_quoted_and_escaped() {
        assert_eq!(exec_arg(Path::new("/home/a b/seiren")), "\"/home/a b/seiren\"");
        assert_eq!(exec_arg(Path::new("/x/50%/s")), "\"/x/50%%/s\"");
        assert_eq!(exec_arg(Path::new("/x/$y")), "\"/x/\\\\$y\"");
        assert_eq!(exec_arg(Path::new("/x/a\"b")), "\"/x/a\\\\\"b\"");
    }

    #[test]
    fn installs_and_removes_both_files() {
        let dir = std::env::temp_dir().join(format!("seiren-menu-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let msg = install_into(&dir, Path::new("/opt/my apps/seiren-control")).unwrap();
        assert!(msg.contains("seiren-control.desktop"));
        let entry = std::fs::read_to_string(desktop_path(&dir)).unwrap();
        assert!(entry.contains("Exec=\"/opt/my apps/seiren-control\"\n"), "{entry}");
        assert!(entry.contains("Icon=seiren-control\n") && entry.contains("StartupWMClass=seiren-control\n"));
        assert!(std::fs::read_to_string(icon_path(&dir)).unwrap().starts_with("<svg"));
        // installing again over it is fine, and removing takes both away
        install_into(&dir, Path::new("/opt/x")).unwrap();
        assert!(uninstall_from(&dir).unwrap().contains("Убрано"));
        assert!(!desktop_path(&dir).exists() && !icon_path(&dir).exists());
        assert!(uninstall_from(&dir).unwrap().contains("не было"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_window_icon_is_a_full_rgba_square() {
        assert_eq!(WINDOW_ICON_RGBA.len() as u32, WINDOW_ICON_SIZE * WINDOW_ICON_SIZE * 4);
    }
}
