// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

import Foundation
import XCTest
@testable import RlxVoiceLink

final class RlxVoiceLinkTests: XCTestCase {
    func testReceiverDropsMalformedPackets() {
        let r = UtteranceReceiver(maxSeconds: 1)
        XCTAssertNil(r.push(Data([0x20, 0, 0, 0, 0, 0])))  // wrong version
        XCTAssertNil(r.push(Data([1, 2])))                  // too short
        XCTAssertEqual(r.lost, 0)
    }

    func testReceiverCompletesAnUtterance() {
        // One packet with START|END, 4 samples of codes, codec state zero.
        let r = UtteranceReceiver(maxSeconds: 1)
        let pcm = r.push(Data([0x13, 0, 0, 0, 0, 0, 0x77, 0x77]))
        XCTAssertEqual(pcm?.count, 4)
    }

    func testRejectsInvalidModel() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("bad.rcc")
        try Data(repeating: 0, count: 64).write(to: url)
        XCTAssertNil(SpeechRecognizer(modelURL: url))
    }

    /// Set RLX_RCC to small-int8.rcc and RLX_WAV to a 16-bit 16 kHz clip.
    func testTranscribesWhenModelIsAvailable() throws {
        let env = ProcessInfo.processInfo.environment
        guard let rcc = env["RLX_RCC"], let wav = env["RLX_WAV"] else {
            throw XCTSkip("RLX_RCC / RLX_WAV not set")
        }
        let asr = try XCTUnwrap(SpeechRecognizer(modelURL: URL(fileURLWithPath: rcc)))
        let bytes = try Data(contentsOf: URL(fileURLWithPath: wav))
        // Find the "data" chunk of a canonical PCM WAV.
        var i = 12
        var pcm: [Int16] = []
        while i + 8 <= bytes.count {
            let id = String(decoding: bytes[i..<i + 4], as: UTF8.self)
            let size = bytes[i + 4..<i + 8].withUnsafeBytes { Int($0.loadUnaligned(as: UInt32.self)) }
            if id == "data" {
                pcm = bytes[i + 8..<min(bytes.count, i + 8 + size)].withUnsafeBytes {
                    Array($0.bindMemory(to: Int16.self))
                }
                break
            }
            i += 8 + size
        }
        let text = try XCTUnwrap(asr.transcribe(pcm))
        print("transcript: \(text)")
        XCTAssertFalse(text.isEmpty)
    }
}
