// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

import Foundation
import RlxVoiceLinkFFI

/// BLE audio notifications → complete utterances (16 kHz PCM). Lost packets
/// are filled with silence of the right length.
public final class UtteranceReceiver {
    private let handle: OpaquePointer

    /// Utterances longer than `maxSeconds` are cut.
    public init(maxSeconds: Int = 30) {
        handle = rvl_receiver_new(maxSeconds * 16_000)
    }

    deinit { rvl_receiver_free(handle) }

    /// Packets lost so far.
    public var lost: UInt32 { rvl_receiver_lost(handle) }

    /// Feed one notification; returns the utterance when one completes.
    public func push(_ packet: Data) -> [Int16]? {
        let r = packet.withUnsafeBytes { raw in
            rvl_receiver_push(handle, raw.bindMemory(to: UInt8.self).baseAddress, raw.count)
        }
        guard r == 1 else { return nil }
        var ptr: UnsafePointer<Int16>?
        let n = rvl_receiver_utterance(handle, &ptr)
        guard let ptr, n > 0 else { return [] }
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }
}
