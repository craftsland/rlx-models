#!/usr/bin/env bash
# Build rlx-voice-link-ffi for iOS (device + simulator) and macOS and package
# it as swift/RlxVoiceLinkFFI.xcframework for the Swift package.
#   IOS=0 skips the iOS slices (macOS-only, e.g. without the iOS SDK).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
root="$(cd "$here/../.." && pwd)"
cd "$root"
out="$here/swift/RlxVoiceLinkFFI.xcframework"
lib=librlx_voice_link_ffi.a
args=()

build() { cargo build -p rlx-voice-link-ffi --release --target "$1"; }

build aarch64-apple-darwin
build x86_64-apple-darwin
mkdir -p target/universal-macos
lipo -create target/aarch64-apple-darwin/release/$lib target/x86_64-apple-darwin/release/$lib \
     -output target/universal-macos/$lib
args+=(-library target/universal-macos/$lib -headers "$here/include")

if [[ "${IOS:-1}" != 0 ]]; then
  build aarch64-apple-ios
  build aarch64-apple-ios-sim
  build x86_64-apple-ios
  mkdir -p target/universal-ios-sim
  lipo -create target/aarch64-apple-ios-sim/release/$lib target/x86_64-apple-ios/release/$lib \
       -output target/universal-ios-sim/$lib
  args+=(-library target/aarch64-apple-ios/release/$lib -headers "$here/include")
  args+=(-library target/universal-ios-sim/$lib -headers "$here/include")
fi

rm -rf "$out"
xcodebuild -create-xcframework "${args[@]}" -output "$out"
echo "wrote $out"
