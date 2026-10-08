import Foundation
import OnnxRuntimeBindings

enum VoiceBuildError: LocalizedError {
    case tooShort(Double)
    case tooLong(Double)
    case silent
    case missingModel(String)
    case badOutput(String)

    var errorDescription: String? {
        switch self {
        case .tooShort(let seconds): return String(format: "录音只有 %.1f 秒，太短了，请用 3～10 秒的片段", seconds)
        case .tooLong(let seconds): return String(format: "录音有 %.0f 秒，太长了（最多 30 秒），请用 3～10 秒的片段", seconds)
        case .silent: return "录音里没有声音"
        case .missingModel(let name): return "缺少模型文件 \(name)"
        case .badOutput(let detail): return "模型输出不对：\(detail)"
        }
    }
}

/// 录音 + 录音里说的话 -> 角色包里的五个张量。对应电脑端 tools/ref_pipeline.py 的 build_voice，
/// 那里调用的是原版的 PyTorch 模型，这里是 tools/export_voice_models.py 导出的 ONNX 版本。
enum VoiceBuilder {
    /// 三个模型，音色编码器的权重在旁边的 .bin 里
    static let modelFiles = ["hubert.onnx", "sv.onnx", "prompt_encoder_fp32.onnx", "prompt_encoder_fp32.bin"]
    static let minSeconds = 1.0
    static let maxSeconds = 30.0

    private struct Output {
        let data: Data
        let shape: [Int]
    }

    /// `models` 是模型文件名到位置的对应（hubert.onnx、sv.onnx、prompt_encoder_fp32.onnx）。
    /// `refBert` 是参考文字的中文语调特征，没有时用全零，与电脑上不装语调模型时相同。
    static func build(audio: URL, reference: ReferenceText, refBert: NSMutableData?, models: [String: URL],
                      threads: Int, progress: (String) -> Void) throws -> [TensorPack.Tensor] {
        func model(_ name: String) throws -> URL {
            guard let url = models[name] else { throw VoiceBuildError.missingModel(name) }
            return url
        }
        let hubert = try model("hubert.onnx")
        let speaker = try model("sv.onnx")
        let encoder = try model("prompt_encoder_fp32.onnx")

        progress("正在读取录音…")
        let (mono, rate) = try AudioLoader.loadMono(audio)
        let seconds = Double(mono.count) / rate
        guard seconds >= minSeconds else { throw VoiceBuildError.tooShort(seconds) }
        guard seconds <= maxSeconds else { throw VoiceBuildError.tooLong(seconds) }
        guard mono.contains(where: { $0 != 0 }) else { throw VoiceBuildError.silent }

        // 语气：16kHz，去掉结尾的静音，再补 0.3 秒静音（gsv_tts.TTS._get_prompt）
        var wave16k = try AudioLoader.resample(mono, from: rate, to: 16_000)
        let tail = tailOffset(wave16k)
        if tail > 0 {
            wave16k.removeLast(tail)
        }
        wave16k.append(contentsOf: [Float](repeating: 0, count: 4_800))

        // 音色：32kHz，峰值超过 1 时压回来；声纹用它再降到 16kHz 的 fbank（gsv_tts.TTS._get_spepc）
        var wave32k = try AudioLoader.resample(mono, from: rate, to: 32_000)
        let peak = wave32k.reduce(Float(0)) { max($0, abs($1)) }
        if peak > 1 {
            let scale = 1 / min(2, peak)
            wave32k = wave32k.map { $0 * scale }
        }
        let fbank = Fbank.compute(try AudioLoader.resample(wave32k, from: 32_000, to: 16_000))
        guard fbank.frames > 0 else { throw VoiceBuildError.tooShort(seconds) }

        // 三个模型加起来要几百 MB 内存，一个用完放掉再载入下一个
        progress("正在提取语气特征…")
        let ssl = try run(hubert, threads: threads,
                          inputs: ["waveform": (wave16k, [1, wave16k.count])], output: "ssl_content")
        guard ssl.shape.count == 3, ssl.shape[1] == 768 else {
            throw VoiceBuildError.badOutput("语气特征的形状是 \(ssl.shape)")
        }

        progress("正在提取声纹…")
        let embedding = try run(speaker, threads: threads,
                                inputs: ["fbank": (fbank.features, [1, fbank.frames, Fbank.melBins])], output: "sv_emb")
        let embeddingValues = embedding.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }

        progress("正在提取音色…")
        let timbre = try runAll(encoder, threads: threads, inputs: [
            "ref_audio": (wave32k, [1, wave32k.count]),
            "sv_emb": (embeddingValues, embedding.shape),
        ], outputs: ["ge", "ge_advanced"])
        guard let ge = timbre["ge"], let geAdvanced = timbre["ge_advanced"] else {
            throw VoiceBuildError.badOutput("音色编码器没有给出 ge")
        }

        let phones = reference.ids.count
        let bertBytes = phones * BertEngine.width * MemoryLayout<Float>.stride
        var bert = Data(count: bertBytes)
        if let refBert, refBert.length == bertBytes {
            bert = Data(bytes: refBert.bytes, count: bertBytes)
        }
        let ids = reference.ids.withUnsafeBufferPointer { Data(buffer: $0) }
        return [
            TensorPack.Tensor(name: "ref_seq", dtype: "i64", shape: [1, phones], data: ids),
            TensorPack.Tensor(name: "ref_bert", dtype: "f32", shape: [phones, BertEngine.width], data: bert),
            TensorPack.Tensor(name: "ssl_content", dtype: "f32", shape: ssl.shape, data: ssl.data),
            TensorPack.Tensor(name: "ge", dtype: "f32", shape: ge.shape, data: ge.data),
            TensorPack.Tensor(name: "ge_advanced", dtype: "f32", shape: geAdvanced.shape, data: geAdvanced.data),
        ]
    }

    /// 与 gsv_tts.TTS._find_threshold_offsets 相同：在最后 6400 个采样里按帧算音量，
    /// 返回最后一个有声帧之后还剩多少个采样（结尾的静音长度）。
    static func tailOffset(_ audio: [Float]) -> Int {
        let frameLength = 512
        let hop = 256
        let start = max(0, audio.count - 6_400)
        let tailLength = audio.count - start
        guard tailLength >= frameLength else { return 0 }
        let threshold = 0.01 * audio.reduce(Float(0)) { max($0, abs($1)) }
        var last = -1
        for frame in 0...((tailLength - frameLength) / hop) {
            var sum: Float = 0
            for index in 0..<frameLength {
                let value = audio[start + frame * hop + index]
                sum += value * value
            }
            if (sum / Float(frameLength)).squareRoot() > threshold {
                last = frame
            }
        }
        guard last >= 0 else { return 0 }
        return tailLength - (last * hop + frameLength)
    }

    private static func run(_ url: URL, threads: Int, inputs: [String: ([Float], [Int])], output: String) throws -> Output {
        guard let result = try runAll(url, threads: threads, inputs: inputs, outputs: [output])[output] else {
            throw VoiceBuildError.badOutput("\(url.lastPathComponent) 没有给出 \(output)")
        }
        return result
    }

    /// 载入一个模型、跑一次、把输出拷出来，然后整个放掉。
    private static func runAll(_ url: URL, threads: Int, inputs: [String: ([Float], [Int])],
                               outputs: [String]) throws -> [String: Output] {
        try autoreleasepool {
            let env = try ORTEnv(loggingLevel: .warning)
            let options = try ORTSessionOptions()
            try options.setIntraOpNumThreads(Int32(threads))
            try options.setGraphOptimizationLevel(.all)
            let session = try ORTSession(env: env, modelPath: url.path, sessionOptions: options)

            var feeds: [String: ORTValue] = [:]
            for (name, input) in inputs {
                let (values, shape) = input
                let data = values.withUnsafeBufferPointer { buffer in
                    NSMutableData(bytes: buffer.baseAddress, length: buffer.count * MemoryLayout<Float>.stride)
                }
                feeds[name] = try ORTValue(tensorData: data, elementType: .float, shape: shape.map { NSNumber(value: $0) })
            }
            let values = try session.run(withInputs: feeds, outputNames: Set(outputs), runOptions: nil)
            var result: [String: Output] = [:]
            for name in outputs {
                guard let value = values[name] else { continue }
                let shape = try value.tensorTypeAndShapeInfo().shape.map { $0.intValue }
                // tensorData 直接指向 value 内部的内存，拷出来之前 value 要活着
                let data = try withExtendedLifetime(value) { () -> Data in
                    let raw = try value.tensorData()
                    return Data(bytes: raw.bytes, count: raw.length)
                }
                let finite = data.withUnsafeBytes { $0.bindMemory(to: Float.self).allSatisfy { $0.isFinite } }
                guard finite else {
                    throw VoiceBuildError.badOutput("\(url.lastPathComponent) 的 \(name) 里有非有限数值")
                }
                result[name] = Output(data: data, shape: shape)
            }
            return result
        }
    }
}
