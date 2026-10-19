# Seiren Control

[![build](https://github.com/TheSiriuss/seiren-control/actions/workflows/ci.yml/badge.svg)](https://github.com/TheSiriuss/seiren-control/actions/workflows/ci.yml)

A control panel for the Razer Seiren V3 Chroma on Linux. Rust + Slint, one binary, no daemon, no service.
Unofficial: not affiliated with Razer.

* **Lighting**: effects, colours, brightness, wave speed, a slow software "soft breathing", the colour of the
  ring while muted. The lighting protocol was reverse engineered by Christoffer Hallas for
  [periferi](https://github.com/csfh/periferi) (MIT); this is an independent re-implementation.
* **Recording and sound**: records the microphone through a voice chain, with live meters and spectrum.
  Noise suppression: RNNoise ([nnnoiseless](https://github.com/jneem/nnnoiseless), BSD-3),
  DeepFilterNet3 ([Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet), MIT/Apache-2.0, vendored in
  `vendor/`) and a spectral Wiener filter with a learned room-noise profile. Then high-pass, gate, 8-band EQ,
  a mouth-noise stage (see below),
  de-esser, compressor, look-ahead limiter, loudness (EBU R128) and normalisation to -16 LUFS. Takes are written
  as WAV 24-bit (and FLAC / Opus / MP3 through ffmpeg) with the untouched take beside them for comparison.
  "Seiren Clean" is a virtual microphone (PipeWire) carrying the processed sound to Discord, OBS, browsers.
* **Mouth noise** (`src/mouth.rs`): lip smacks, tongue clicks, chewing, saliva and "water in the mouth". A click
  scoop finds spikes of a few milliseconds above 2.5 kHz that stand far above the band just before and just after
  them (a vowel or a fricative builds up and stays) and crossfades those milliseconds to a 1.6 kHz low-passed copy.
  An orphan-sound stage classes each 10 ms frame as voiced (periodic, low-frequency heavy) or not; an audible
  frame with no voice within 40 ms before or 50 ms after it, and quiet next to the speaker's usual voice, is
  turned down by 6-30 dB. Consonants that touch a word stay; whispering switches the stage off by itself.
  It delays the sound by about 70 ms. Tested on synthetic words and mouth sounds only, not on recorded speech.

```
./seiren-control            open the window
./seiren-control --status   print what it can see (changes nothing)
./seiren-control --demo     pretend microphone, touches no hardware
./seiren-control --install-menu    add it to the application menu with an icon (--uninstall-menu removes it)
./seiren-control --grant-help      the terminal commands for the one-time access rule (no polkit)
./seiren-control --process in.wav out.wav [--preset podcast|stream|vocal|calls|clean|off] [--denoise dfn|rnnoise|spectral|off]
```

## Install

A prebuilt x86-64 binary (built on Slackware 15.0, so it runs on any glibc 2.33 or newer) is attached to each
[release](https://github.com/TheSiriuss/seiren-control/releases); it needs `fontconfig` and `freetype` installed.
Or build from source:

Works on Arch (CachyOS, Manjaro, EndeavourOS), Debian (Ubuntu, Mint), Void and Slackware. It is built from source, so it
does not care which libc or init system you have (systemd, runit, OpenRC, Slackware's rc scripts).

```
git clone https://github.com/TheSiriuss/seiren-control
cd seiren-control
./install.sh --deps --rustup      # system packages (asks for sudo/doas), Rust via rustup if yours is older than 1.96
```

`./install.sh --help` lists the options (`--prefix DIR`, `--no-menu`, `--uninstall`). The script puts the binary in
`~/.local/bin` and adds an application-menu entry. By hand, if you prefer: `cargo build --release`, the binary is
`target/release/seiren-control` (the first build compiles the neural runtime and takes a few minutes; Rust 1.96 or
newer, `pkg-config`, and the fontconfig and freetype development files are needed).

| | build and run-time packages `install.sh --deps` installs | notes |
|---|---|---|
| Arch | `base-devel pkgconf fontconfig freetype2 libxkbcommon libxkbcommon-x11 libx11 libxcursor libxi libxrandr wayland mesa libpulse alsa-utils ffmpeg polkit xdg-utils xdg-user-dirs ttf-dejavu` | `rust` from the repo is fine while it is 1.96+ |
| Debian / Ubuntu | `build-essential pkg-config libfontconfig1-dev libfreetype6-dev` + X11/Wayland/GL libs, `pulseaudio-utils alsa-utils ffmpeg pkexec xdg-utils xdg-user-dirs fonts-dejavu-core` | the distro `rustc` is too old: use `--rustup` |
| Void | `base-devel pkg-config fontconfig-devel freetype-devel` + X11/Wayland/GL libs, `pulseaudio-utils alsa-utils ffmpeg polkit xdg-utils xdg-user-dirs dejavu-fonts-ttf` (and `mesa mesa-dri libglvnd`) | |
| Slackware | nothing to install on a full installation | Rust through `--rustup`; ffmpeg is optional (SlackBuilds.org) |

The continuous-integration workflow builds and tests it inside Arch, Debian 12, Debian 13, Void and Slackware 15.0
containers (`.github/workflows/ci.yml`). That checks the build and the package names; it cannot check the audio, which
needs a real sound server and a microphone.

Test: `cargo test --release`. `cargo test --release -- --ignored real_microphone --nocapture` and `-- --ignored virtual_mic`
use the real microphone and a sound server; `-- --ignored hardware` drives the lighting (live values only, nothing saved).
`SEIREN_VIRTUAL_MIC=pulse` forces the PulseAudio-module virtual microphone even when `pw-loopback` exists.

## What it needs at run time

* A sound server that speaks the PulseAudio protocol: PipeWire with `pipewire-pulse`, or PulseAudio itself. The app uses
  its client tools `parec`, `pacat` and `pactl`, plus `amixer` (alsa-utils) for the sound card's gain and monitoring.
* The "Seiren Clean" virtual microphone is a `pw-loopback` pair on PipeWire; where `pw-loopback` is missing (PulseAudio,
  Slackware 15.0) the app loads two PulseAudio modules (`module-null-sink`, `module-remap-source`) instead and removes
  them when it stops.
* Optional: `ffmpeg`/`ffplay` (FLAC, Opus, MP3 and listening back), `xdg-open`, `xdg-user-dirs`.
* `seiren-control --status` prints which of these it found, next to the state of the microphone itself.

## Access to the microphone

The kernel lets only root open `/dev/hidraw*`. The window offers a one-time button that writes
`/etc/udev/rules.d/70-seiren-v3-chroma.rules` through `pkexec`. The rule has two halves so that it works without
systemd: `TAG+="uaccess"` (systemd-logind and elogind give the device to the active session) and `GROUP`/`MODE="0660"`
with a group you are already in (`plugdev` if you are in it, otherwise your own group), so access also works on runit,
OpenRC and Slackware, at once, without logging in again. No polkit or no authentication agent: run
`seiren-control --grant-help` and it prints the commands for a root shell. Delete the file to undo it.

Settings: `~/.config/seiren-control/settings.json` (lighting), `audio.json` (voice chain, formats).
Takes: `~/Музыка/Seiren/` (the XDG music folder).
