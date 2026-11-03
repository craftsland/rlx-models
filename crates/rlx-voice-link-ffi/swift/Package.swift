// swift-tools-version:5.9
// Swift wrapper over rlx-voice-link-ffi: CoreBluetooth central for the
// nRF52840 voice-link firmware, utterance reassembly and on-device int8 ASR.
// Build the binary first: crates/rlx-voice-link-ffi/scripts/build-xcframework.sh
import PackageDescription

let package = Package(
    name: "RlxVoiceLink",
    platforms: [.iOS(.v15), .macOS(.v12)],
    products: [
        .library(name: "RlxVoiceLink", targets: ["RlxVoiceLink"]),
    ],
    targets: [
        .binaryTarget(name: "RlxVoiceLinkFFI", path: "RlxVoiceLinkFFI.xcframework"),
        .target(
            name: "RlxVoiceLink",
            dependencies: ["RlxVoiceLinkFFI"],
            linkerSettings: [.linkedFramework("CoreBluetooth")]
        ),
        .testTarget(name: "RlxVoiceLinkTests", dependencies: ["RlxVoiceLink"]),
    ]
)
