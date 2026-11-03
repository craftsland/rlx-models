// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

import Foundation

/// Where a transcript goes next.
public protocol ChatModel {
    func reply(to text: String) async throws -> String
}

/// Any OpenAI-compatible `/v1/chat/completions` endpoint, e.g. `rlx-openai`
/// serving Qwen3 or Gemma on the Mac (`just openai-serve -- --engine qwen3 …`).
/// For a model inside an iOS app, link the families through
/// `rlx-models-plugin` (RLX.xcframework) and implement `ChatModel` on it.
public struct OpenAIChatModel: ChatModel {
    public var url: URL
    public var model: String
    public var system: String?

    public init(url: URL, model: String = "default", system: String? = nil) {
        self.url = url
        self.model = model
        self.system = system
    }

    public func reply(to text: String) async throws -> String {
        var messages: [[String: String]] = []
        if let system { messages.append(["role": "system", "content": system]) }
        messages.append(["role": "user", "content": text])
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: ["model": model, "messages": messages])
        let (data, _) = try await URLSession.shared.data(for: req)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let choices = json?["choices"] as? [[String: Any]]
        let message = choices?.first?["message"] as? [String: Any]
        guard let content = message?["content"] as? String else {
            throw URLError(.cannotParseResponse)
        }
        return content
    }
}
