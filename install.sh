#!/bin/sh
# Build and install seiren-control from source.
# Arch (and CachyOS, Manjaro, EndeavourOS), Debian (and Ubuntu, Mint), Void and Slackware.
#
#   ./install.sh                 build with the Rust you have, install into ~/.local/bin, add the menu entry
#   ./install.sh --deps          also install the system packages first (asks sudo/doas for root)
#   ./install.sh --rustup        install Rust with rustup (into ~/.cargo, no root) if yours is missing or too old
#   ./install.sh --prefix DIR    install into DIR/bin instead of ~/.local/bin
#   ./install.sh --no-menu       do not add the application-menu entry
#   ./install.sh --uninstall     remove the binary and the menu entry
#
# Nothing here starts a service. The only thing that ever needs root is the one-time udev rule
# that the app itself offers (or `seiren-control --grant-help`).

set -eu

MIN_RUST_MAJOR=1
MIN_RUST_MINOR=96

prefix="${PREFIX:-$HOME/.local}"
want_deps=0
want_rustup=0
want_menu=1
uninstall=0

while [ $# -gt 0 ]; do
    case "$1" in
        --deps) want_deps=1 ;;
        --rustup) want_rustup=1 ;;
        --no-menu) want_menu=0 ;;
        --uninstall) uninstall=1 ;;
        --prefix) shift; prefix="${1:?--prefix needs a directory}" ;;
        -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

say() { printf '==> %s\n' "$*"; }
die() { printf 'install.sh: %s\n' "$*" >&2; exit 1; }

here=$(cd "$(dirname "$0")" && pwd)
bin="$prefix/bin/seiren-control"

if [ "$uninstall" = 1 ]; then
    [ -x "$bin" ] && "$bin" --uninstall-menu || true
    rm -f "$bin"
    say "removed $bin"
    say "the udev rule (if you created it) is /etc/udev/rules.d/70-seiren-v3-chroma.rules; delete it as root to undo"
    exit 0
fi

# ---- which distribution ---------------------------------------------------------------------------
distro=unknown
if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    for id in ${ID:-} ${ID_LIKE:-}; do
        case "$id" in
            arch) distro=arch; break ;;
            debian|ubuntu) distro=debian; break ;;
            void) distro=void; break ;;
            slackware) distro=slackware; break ;;
        esac
    done
fi
[ -f /etc/slackware-version ] && distro=slackware
say "distribution family: $distro"

# ---- system packages --------------------------------------------------------------------------------
as_root() {
    if [ "$(id -u)" = 0 ]; then "$@"
    elif command -v sudo >/dev/null 2>&1; then sudo "$@"
    elif command -v doas >/dev/null 2>&1; then doas "$@"
    else die "need root for: $* (install sudo or doas, or run this script as root)"; fi
}

install_deps() {
    # Build: a C compiler, pkg-config, fontconfig and freetype headers (the window text).
    # Run: X11/Wayland/GL libraries (loaded at run time), the PulseAudio client tools (they talk to
    # PipeWire's pulse layer too), alsa-utils, polkit for the access button, ffmpeg for FLAC/Opus/MP3.
    # An audio server (PipeWire with pipewire-pulse, or PulseAudio) is yours to choose; none is installed here.
    case "$distro" in
        arch)
            as_root pacman -S --needed --noconfirm base-devel pkgconf curl fontconfig freetype2 \
                libxkbcommon libxkbcommon-x11 libx11 libxcursor libxi libxrandr wayland mesa \
                libpulse alsa-utils ffmpeg polkit xdg-utils xdg-user-dirs
            ;;
        debian)
            as_root apt-get update
            as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential pkg-config curl ca-certificates \
                libfontconfig1-dev libfreetype6-dev \
                libxkbcommon0 libxkbcommon-x11-0 libx11-6 libxcursor1 libxi6 libxrandr2 \
                libwayland-client0 libwayland-cursor0 libwayland-egl1 libegl1 libgl1 \
                pulseaudio-utils alsa-utils ffmpeg xdg-utils xdg-user-dirs
            as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y pkexec \
                || as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y policykit-1 \
                || say "could not install pkexec; the access button will tell you the terminal commands instead"
            ;;
        void)
            as_root xbps-install -Sy base-devel pkg-config curl fontconfig-devel freetype-devel \
                libxkbcommon libX11 libXcursor libXi libXrandr wayland MesaLib libglvnd \
                pulseaudio-utils alsa-utils ffmpeg polkit xdg-utils xdg-user-dirs
            ;;
        slackware)
            say "Slackware: a full installation already has the compiler, pkg-config, fontconfig, freetype, X11,"
            say "PulseAudio, alsa-utils and polkit. Optional: ffmpeg (SlackBuilds.org) for FLAC/Opus/MP3,"
            say "PipeWire (-current); without it the virtual microphone uses PulseAudio modules."
            ;;
        *)
            say "unknown distribution: install a C compiler, pkg-config, fontconfig + freetype headers,"
            say "the PulseAudio client tools (parec, pacat, pactl), alsa-utils and, if you like, ffmpeg and polkit."
            ;;
    esac
}

[ "$want_deps" = 1 ] && install_deps

# ---- Rust ---------------------------------------------------------------------------------------------
[ -r "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

rust_ok() {
    command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1 || return 1
    v=$(rustc --version | cut -d' ' -f2)
    major=${v%%.*}; rest=${v#*.}; minor=${rest%%.*}
    [ "$major" -gt "$MIN_RUST_MAJOR" ] || { [ "$major" -eq "$MIN_RUST_MAJOR" ] && [ "$minor" -ge "$MIN_RUST_MINOR" ]; }
}

if ! rust_ok; then
    say "Rust $MIN_RUST_MAJOR.$MIN_RUST_MINOR or newer is needed (found: $(rustc --version 2>/dev/null || echo none)); distro packages are often older"
    if [ "$want_rustup" != 1 ]; then
        if [ -t 0 ]; then
            printf 'Install Rust with rustup into ~/.cargo (no root)? [y/N] '
            read -r ans || ans=n
            case "$ans" in y|Y|yes) want_rustup=1 ;; esac
        fi
    fi
    [ "$want_rustup" = 1 ] || die "install a newer Rust (https://rustup.rs) or re-run with --rustup"
    command -v curl >/dev/null 2>&1 || die "curl is needed to fetch rustup"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
    . "$HOME/.cargo/env"
    rust_ok || die "rustup finished but rustc is still too old: $(rustc --version)"
fi
say "using $(rustc --version)"

# ---- build and install -----------------------------------------------------------------------------
cd "$here"
say "building (the first build compiles the neural-network runtime and takes a few minutes)"
cargo build --release --locked

mkdir -p "$prefix/bin"
# copy beside, then rename: replacing a running binary in place fails with "text file busy"
cp target/release/seiren-control "$bin.new"
chmod 755 "$bin.new"
mv -f "$bin.new" "$bin"
say "installed $bin"

if [ "$want_menu" = 1 ]; then
    "$bin" --install-menu || say "could not add the menu entry (not fatal)"
fi

case ":$PATH:" in
    *":$prefix/bin:"*) ;;
    *) say "$prefix/bin is not on your PATH; add it, or run $bin directly" ;;
esac

say "what it can see on this machine:"
"$bin" --status || true
say "done. If the microphone says access is denied, open the window and press the access button (or run: seiren-control --grant-help)."
