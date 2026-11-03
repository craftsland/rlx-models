// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

import Foundation
import RlxVoiceLinkFFI

/// On-device speech recognition: the int8 Conformer-CTC `.rcc` model
/// (`rlx-conformer-ctc export-mcu`), the same model and code the MCU runs.
public final class SpeechRecognizer {
    private let handle: OpaquePointer

    /// Load an `.rcc` file. Returns nil if it is not a valid model.
    public init?(modelURL: URL) {
        guard let data = try? Data(contentsOf: modelURL, options: .mappedIfSafe) else { return nil }
        let h = data.withUnsafeBytes { raw -> OpaquePointer? in
            rvl_asr_open(raw.bindMemory(to: UInt8.self).baseAddress, raw.count)
        }
        guard let h else { return nil }
        handle = h
    }

    deinit { rvl_asr_free(handle) }

    /// 16 kHz mono PCM → text, or nil on error.
    public func transcribe(_ pcm: [Int16]) -> String? {
        var out = [UInt8](repeating: 0, count: max(64, pcm.count / 160 * 32))
        let n = pcm.withUnsafeBufferPointer { p in
            out.withUnsafeMutableBufferPointer { o in
                rvl_asr_transcribe(handle, p.baseAddress, p.count, o.baseAddress, o.count)
            }
        }
        guard n >= 0 else { return nil }
        return String(decoding: out[0..<Int(n)], as: UTF8.self)
    }
}
