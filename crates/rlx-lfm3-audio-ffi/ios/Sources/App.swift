import AVFoundation
import SwiftUI

@main
struct Lfm3AudioApp: App {
    init() {
        // `--selftest`: run ASR + TTS once, print results to the console and exit.
        if CommandLine.arguments.contains("--selftest") {
            Thread.detachNewThread { SelfTest.run() }
        }
    }

    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

enum SelfTest {
    static func log(_ s: String) {
        print("[lfm3a] \(s) (footprint \(String(format: "%.0f", Lfm3Engine.footprintMiB)) MiB)")
        fflush(stdout)
    }

    static func run() {
        do {
            let w = try Weights.find()
            log("weights \(w.lm.lastPathComponent) + \(w.mmproj.lastPathComponent)")
            var t = Date()
            let engine = try Lfm3Engine(lm: w.lm, mmproj: w.mmproj)
            log("open \(String(format: "%.2f", -t.timeIntervalSinceNow)) s")
            let pcm = try Wav.read(Bundle.main.url(forResource: "jfk_16k", withExtension: "wav")!)
            t = Date()
            let text = try engine.transcribe(pcm)
            log("asr \(String(format: "%.2f", -t.timeIntervalSinceNow)) s: \(text)")
            t = Date()
            let speech = try engine.speak("The weather is lovely today.")
            let secs = Double(speech.count) / 24_000
            log("tts \(String(format: "%.2f", -t.timeIntervalSinceNow)) s for \(String(format: "%.2f", secs)) s of audio")
            let out = Weights.documents.appendingPathComponent("selftest_tts.wav")
            try Wav.write(speech, to: out)
            let back = try engine.transcribe(try Wav.read(out))
            log("tts round trip: \(back)")
            log("PASS")
        } catch {
            log("FAIL \(error)")
        }
        exit(0)
    }
}

@MainActor
final class Model: ObservableObject {
    @Published var status = "Tap Load."
    @Published var transcript = ""
    @Published var text = "The quick brown fox jumps over the lazy dog."
    @Published var busy = false
    private var engine: Lfm3Engine?
    private var player: AVAudioPlayer?
    private let queue = DispatchQueue(label: "lfm3a")

    private func work(_ label: String, _ f: @escaping () throws -> String) {
        busy = true
        status = "\(label)…"
        let t = Date()
        queue.async {
            let result: String
            do { result = try f() } catch { result = "\(error)" }
            let secs = -t.timeIntervalSinceNow
            DispatchQueue.main.async {
                self.busy = false
                self.status = "\(label): \(String(format: "%.1f", secs)) s, \(String(format: "%.0f", Lfm3Engine.footprintMiB)) MiB"
                if !result.isEmpty { self.transcript = result }
            }
        }
    }

    func load() {
        work("Load") {
            let w = try Weights.find()
            let e = try Lfm3Engine(lm: w.lm, mmproj: w.mmproj)
            DispatchQueue.main.async { self.engine = e }
            return "\(w.lm.lastPathComponent) + \(w.mmproj.lastPathComponent)"
        }
    }

    func transcribeSample() {
        guard let e = engine else { status = "Load first."; return }
        work("Transcribe") {
            try e.transcribe(try Wav.read(Bundle.main.url(forResource: "jfk_16k", withExtension: "wav")!))
        }
    }

    func speak() {
        guard let e = engine else { status = "Load first."; return }
        let text = self.text
        work("Speak") {
            let pcm = try e.speak(text)
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("speech.wav")
            try Wav.write(pcm, to: url)
            DispatchQueue.main.async {
                try? AVAudioSession.sharedInstance().setCategory(.playback)
                self.player = try? AVAudioPlayer(contentsOf: url)
                self.player?.play()
            }
            return ""
        }
    }
}

struct ContentView: View {
    @StateObject private var m = Model()

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("LFM3-Audio-2B on rlx").font(.title2).bold()
            Text(m.status).font(.footnote).foregroundStyle(.secondary)
            HStack {
                Button("Load", action: m.load)
                Button("Transcribe JFK", action: m.transcribeSample)
            }
            .buttonStyle(.borderedProminent)
            .disabled(m.busy)
            TextField("Text to speak", text: $m.text, axis: .vertical).textFieldStyle(.roundedBorder)
            Button("Speak", action: m.speak).buttonStyle(.bordered).disabled(m.busy)
            ScrollView { Text(m.transcript).frame(maxWidth: .infinity, alignment: .leading) }
        }
        .padding()
    }
}
