#!/usr/bin/env bash
# Build, install and self-test the LFM3-Audio app on a device or simulator.
#
#   ios/run.sh device <device-id>        # paired iPhone / iPad (unlocked)
#   ios/run.sh sim [<simulator-udid>]    # default: the booted simulator
#
# Copies the Q4_K_M LM + Q8_0 mmproj (the low-memory pair) into the app's
# Documents, then launches it with `--selftest`: ASR on the bundled JFK clip,
# TTS, and TTS transcribed back, with timings and the process footprint.
# Env: WEIGHTS (default ../../../weights/lm/lfm3-audio-2b-gguf), TEAM
# (signing team for the device build), BUNDLE_ID (default dev.rlx.lfm3audio;
# use one your team can sign), LM, MMPROJ (file names).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$here/../../.."
weights="${WEIGHTS:-$root/weights/lm/lfm3-audio-2b-gguf}"
lm="${LM:-LFM3-Audio-2B-Q4_K_M.gguf}"
mmproj="${MMPROJ:-mmproj-LFM3-Audio-2B-Q8_0.gguf}"
bundle="${BUNDLE_ID:-dev.rlx.lfm3audio}"
dd="${DERIVED_DATA:-$here/build/dd}"
kind="${1:?usage: run.sh device <id> | sim [udid]}"

export IPHONEOS_DEPLOYMENT_TARGET=${IPHONEOS_DEPLOYMENT_TARGET:-17.0}
target_dir="${CARGO_TARGET_DIR:-$root/target}"
build_lib() { # <rust target> <dest dir>
    (cd "$root" && cargo build -p rlx-lfm3-audio-ffi --release --features metal --target "$1")
    mkdir -p "$2"
    cp "$target_dir/$1/release/librlx_lfm3_audio_ffi.a" "$2/"
}

cd "$here"
LFM3A_TEAM="${TEAM:-}" LFM3A_BUNDLE_ID="$bundle" xcodegen generate >/dev/null
case "$kind" in
device)
    dev="${2:?device id (xcrun devicectl list devices)}"
    build_lib aarch64-apple-ios "$here/build"
    xcodebuild -project Lfm3Audio.xcodeproj -scheme Lfm3Audio -configuration Release \
        -destination generic/platform=iOS -derivedDataPath "$dd" build | tail -1
    app="$dd/Build/Products/Release-iphoneos/Lfm3Audio.app"
    xcrun devicectl device install app --device "$dev" "$app"
    for f in "$lm" "$mmproj"; do
        xcrun devicectl device copy to --device "$dev" --domain-type appDataContainer \
            --domain-identifier "$bundle" --source "$weights/$f" --destination "Documents/$f"
    done
    xcrun devicectl device process launch --device "$dev" --console --terminate-existing \
        "$bundle" --selftest
    ;;
sim)
    sim="${2:-booted}"
    build_lib aarch64-apple-ios-sim "$here/build/sim"
    xcodebuild -project Lfm3Audio.xcodeproj -scheme Lfm3Audio -configuration Release \
        -destination "generic/platform=iOS Simulator" -derivedDataPath "$dd" \
        LIBRARY_SEARCH_PATHS="$here/build/sim" build | tail -1
    xcrun simctl install "$sim" "$dd/Build/Products/Release-iphonesimulator/Lfm3Audio.app"
    docs="$(xcrun simctl get_app_container "$sim" "$bundle" data)/Documents"
    mkdir -p "$docs"
    # Real copies: the simulated app cannot open files behind a symlink to an
    # external volume (the open blocks on a privacy prompt).
    for f in "$lm" "$mmproj"; do
        if [ -L "$docs/$f" ] || [ ! -f "$docs/$f" ]; then
            cp -L "$weights/$f" "$docs/$f.part"
            mv "$docs/$f.part" "$docs/$f"
        fi
    done
    xcrun simctl launch --console-pty --terminate-running-process "$sim" "$bundle" --selftest
    ;;
*) echo "usage: run.sh device <id> | sim [udid]" >&2; exit 2 ;;
esac
