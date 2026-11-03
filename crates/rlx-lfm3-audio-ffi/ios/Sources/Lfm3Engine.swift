import AVFoundation
import Foundation

/// Swift wrapper over the rlx-lfm3-audio C ABI (one model, one thread).
final class Lfm3Engine {
    struct Failure: Error, CustomStringConvertible {
        let description: String
    }

    private var handle: OpaquePointer?

    static var lastError: String { String(cString: lfm3a_last_error()) }

    /// Process physical footprint in MiB (what iOS's memory limit counts).
    static var footprintMiB: Double { Double(lfm3a_footprint()) / 1_048_576 }

    init(lm: URL, mmproj: URL, device: String = "metal", lowMemory: Bool = true) throws {
        guard let h = lfm3a_open(lm.path, mmproj.path, device, lowMemory) else {
            throw Failure(description: "open: \(Self.lastError)")
        }
        handle = h
    }

    deinit { lfm3a_close(handle) }

    /// Transcribe 16 kHz mono samples.
    func transcribe(_ pcm: [Float], language: String? = "English") throws -> String {
        let out = pcm.withUnsafeBufferPointer { buf in
            lfm3a_transcribe(handle, buf.baseAddress, buf.count, language, 0)
        }
        guard let out else { throw Failure(description: "transcribe: \(Self.lastError)") }
        defer { lfm3a_free_string(out) }
        return String(cString: out)
    }

    /// Synthesize 24 kHz mono speech.
    func speak(_ text: String, language: String = "en", voice: String = "female", seed: UInt64 = 0) throws -> [Float] {
        var n = 0
        guard let pcm = lfm3a_speak(handle, text, language, voice, seed, &n) else {
            throw Failure(description: "speak: \(Self.lastError)")
        }
        defer { lfm3a_free_audio(pcm, n) }
        return Array(UnsafeBufferPointer(start: pcm, count: n))
    }
}

enum Wav {
    /// Mono float samples of an audio file, resampled to `rate`.
    static func read(_ url: URL, rate: Double = 16_000) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let want = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 1, interleaved: false)!
        let src = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
        try file.read(into: src)
        if file.processingFormat.sampleRate == rate && file.processingFormat.channelCount == 1 {
            return Array(UnsafeBufferPointer(start: src.floatChannelData![0], count: Int(src.frameLength)))
        }
        let conv = AVAudioConverter(from: file.processingFormat, to: want)!
        let cap = AVAudioFrameCount(Double(src.frameLength) * rate / file.processingFormat.sampleRate) + 1024
        let dst = AVAudioPCMBuffer(pcmFormat: want, frameCapacity: cap)!
        var fed = false
        var err: NSError?
        conv.convert(to: dst, error: &err) { _, status in
            if fed { status.pointee = .endOfStream; return nil }
            fed = true
            status.pointee = .haveData
            return src
        }
        if let err { throw err }
        return Array(UnsafeBufferPointer(start: dst.floatChannelData![0], count: Int(dst.frameLength)))
    }

    /// Write 24 kHz mono samples as a 16-bit WAV.
    static func write(_ pcm: [Float], to url: URL, rate: Double = 24_000) throws {
        let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 1, interleaved: false)!
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: rate,
            AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16,
        ]
        let file = try AVAudioFile(forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(pcm.count))!
        buf.frameLength = AVAudioFrameCount(pcm.count)
        pcm.withUnsafeBufferPointer { buf.floatChannelData![0].update(from: $0.baseAddress!, count: pcm.count) }
        try file.write(from: buf)
    }
}

/// The weights in the app's Documents folder: the LM `LFM3-Audio-2B-*.gguf`
/// (Q4_K_M preferred) and its `mmproj-*.gguf` (Q8_0 preferred).
enum Weights {
    static var documents: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0] }

    static func find() throws -> (lm: URL, mmproj: URL) {
        let files = (try? FileManager.default.contentsOfDirectory(at: documents, includingPropertiesForKeys: nil)) ?? []
        let gguf = files.filter { $0.pathExtension == "gguf" }
        func pick(_ ok: (String) -> Bool, prefer: String) -> URL? {
            let c = gguf.filter { ok($0.lastPathComponent) }.sorted { $0.lastPathComponent < $1.lastPathComponent }
            return c.first { $0.lastPathComponent.contains(prefer) } ?? c.first
        }
        guard let lm = pick({ !$0.hasPrefix("mmproj") }, prefer: "Q4_K_M"),
              let mm = pick({ $0.hasPrefix("mmproj") }, prefer: "Q8_0")
        else {
            throw Lfm3Engine.Failure(description: "copy the LM and mmproj GGUFs into Documents (\(documents.path))")
        }
        return (lm, mm)
    }
}
