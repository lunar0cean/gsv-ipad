import Foundation
import OnnxRuntimeBindings

struct SynthStats {
    var phones = 0
    var tokens = 0
    var attempts = 0
    var seconds = 0.0
    var audioSeconds = 0.0
}

enum SynthError: LocalizedError {
    case missingModel(String)
    case missingOutput(String)
    case noSemantic

    var errorDescription: String? {
        switch self {
        case .missingModel(let name): return "缺少模型文件 \(name)"
        case .missingOutput(let name): return "模型没有给出 \(name)"
        case .noSemantic: return "没有生成任何语义，无法合成"
        }
    }
}

/// 音素 + 角色包 -> 波形。对应电脑端 tools/ref_pipeline.py 里的 OnnxSynth。
final class SynthEngine {
    static let modelFiles = [
        "t2s_encoder_fp32.onnx", "t2s_encoder_fp32.bin",
        "t2s_first_stage_decoder_fp32.onnx", "t2s_stage_decoder_fp32.onnx", "t2s_shared_fp32.bin",
        "vits_fp32.onnx", "vits_fp32.bin",
    ]
    static let sampleRate = 32_000

    private static let bertWidth = 1024
    private static let eos: Int64 = 1024
    private static let maxSteps = 1000
    private static let maxAttempts = 3
    private static let stopName = "stop_flag"

    private let env: ORTEnv
    private let encoder: ORTSession
    private let firstStage: ORTSession
    private let stage: ORTSession
    private let vocoder: ORTSession

    // 解码循环按位置传递状态：y、y_emb，然后是每层的 k、v 缓存
    private let firstOutputs: [String]
    private let stageInputs: [String]
    private let stageStateOutputs: [String]
    private let stageOutputSet: Set<String>

    init(modelDirectory: URL, threads: Int) throws {
        let env = try ORTEnv(loggingLevel: .warning)
        let options = try ORTSessionOptions()
        try options.setIntraOpNumThreads(Int32(threads))
        try options.setGraphOptimizationLevel(.all)

        func load(_ name: String) throws -> ORTSession {
            let path = modelDirectory.appendingPathComponent(name).path
            guard FileManager.default.fileExists(atPath: path) else {
                throw SynthError.missingModel(name)
            }
            return try ORTSession(env: env, modelPath: path, sessionOptions: options)
        }

        let encoder = try load("t2s_encoder_fp32.onnx")
        let firstStage = try load("t2s_first_stage_decoder_fp32.onnx")
        let stage = try load("t2s_stage_decoder_fp32.onnx")
        let vocoder = try load("vits_fp32.onnx")
        let stageOutputs = try stage.outputNames()

        self.env = env
        self.encoder = encoder
        self.firstStage = firstStage
        self.stage = stage
        self.vocoder = vocoder
        self.firstOutputs = try firstStage.outputNames()
        self.stageInputs = try stage.inputNames()
        self.stageStateOutputs = stageOutputs.filter { $0 != SynthEngine.stopName }
        self.stageOutputSet = Set(stageOutputs)
    }

    /// `isCancelled` 在解码循环的每一步都会被问一次；返回 true 时本次合成返回 nil。
    func synthesize(voice: TensorPack, phonemes: [Int64],
                    isCancelled: () -> Bool) throws -> (samples: [Float], stats: SynthStats)? {
        let clock = CFAbsoluteTimeGetCurrent()
        var stats = SynthStats()
        stats.phones = phonemes.count
        let textSeq = try SynthEngine.int64Value(phonemes, shape: [1, phonemes.count])

        // 语义数量远少于音素数量说明提前结束了，这一步便宜，直接重来
        var tokens: [Int64] = []
        for attempt in 1...SynthEngine.maxAttempts {
            stats.attempts = attempt
            guard let generated = try semanticTokens(voice: voice, textSeq: textSeq, phoneCount: phonemes.count,
                                                     isCancelled: isCancelled) else {
                return nil
            }
            tokens = generated
            if Double(tokens.count) >= Double(phonemes.count) * 0.8 {
                break
            }
        }
        guard !tokens.isEmpty else {
            throw SynthError.noSemantic
        }
        stats.tokens = tokens.count

        let inputs: [String: ORTValue] = [
            "text_seq": textSeq,
            "pred_semantic": try SynthEngine.int64Value(tokens, shape: [1, 1, tokens.count]),
            "ge": try voice.value("ge"),
            "ge_advanced": try voice.value("ge_advanced"),
        ]
        let outputs = try vocoder.run(withInputs: inputs, outputNames: ["audio"], runOptions: nil)
        guard let audio = outputs["audio"] else {
            throw SynthError.missingOutput("audio")
        }
        let audioData = try audio.tensorData() as Data
        let samples: [Float] = audioData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        stats.seconds = CFAbsoluteTimeGetCurrent() - clock
        stats.audioSeconds = Double(samples.count) / Double(SynthEngine.sampleRate)
        return (samples, stats)
    }

    private func semanticTokens(voice: TensorPack, textSeq: ORTValue, phoneCount: Int,
                                isCancelled: () -> Bool) throws -> [Int64]? {
        // 目标文本的 BERT 特征先用全零（日文本来就是全零；中文以后可以接上 RoBERTa）
        guard let zeros = NSMutableData(length: phoneCount * SynthEngine.bertWidth * MemoryLayout<Float>.stride) else {
            throw SynthError.missingOutput("text_bert")
        }
        let textBert = try ORTValue(tensorData: zeros, elementType: .float,
                                    shape: [NSNumber(value: phoneCount), NSNumber(value: SynthEngine.bertWidth)])
        let encoderInputs: [String: ORTValue] = [
            "ref_seq": try voice.value("ref_seq"),
            "text_seq": textSeq,
            "ref_bert": try voice.value("ref_bert"),
            "text_bert": textBert,
            "ssl_content": try voice.value("ssl_content"),
        ]
        let encoded = try encoder.run(withInputs: encoderInputs, outputNames: ["x", "prompts"], runOptions: nil)
        guard let x = encoded["x"] else {
            throw SynthError.missingOutput("x")
        }
        guard let prompts = encoded["prompts"] else {
            throw SynthError.missingOutput("prompts")
        }
        let promptLength = try prompts.tensorTypeAndShapeInfo().shape.last?.intValue ?? 0

        let first = try firstStage.run(withInputs: ["x": x, "prompts": prompts],
                                       outputNames: Set(firstOutputs), runOptions: nil)
        var state = try SynthEngine.ordered(first, names: firstOutputs)

        // 每一步的输出里有 48 个随长度增长的缓存张量，必须每步清一次自动释放池，否则内存会一路涨上去
        var stopped = false
        for _ in 0..<SynthEngine.maxSteps {
            if isCancelled() {
                return nil
            }
            stopped = try autoreleasepool {
                var inputs = [String: ORTValue](minimumCapacity: stageInputs.count)
                for (name, value) in zip(stageInputs, state) {
                    inputs[name] = value
                }
                let out = try stage.run(withInputs: inputs, outputNames: stageOutputSet, runOptions: nil)
                state = try SynthEngine.ordered(out, names: stageStateOutputs)
                guard let stop = out[SynthEngine.stopName] else {
                    throw SynthError.missingOutput(SynthEngine.stopName)
                }
                let flag = try SynthEngine.int64Array(stop).first ?? 0
                return flag != 0
            }
            if stopped {
                break
            }
        }

        var generated = Array(try SynthEngine.int64Array(state[0]).dropFirst(promptLength))
        if stopped, !generated.isEmpty {
            generated.removeLast()  // 触发结束的那一个不算
        }
        return generated.filter { $0 < SynthEngine.eos }
    }

    private static func ordered(_ values: [String: ORTValue], names: [String]) throws -> [ORTValue] {
        try names.map { name -> ORTValue in
            guard let value = values[name] else {
                throw SynthError.missingOutput(name)
            }
            return value
        }
    }

    private static func int64Array(_ value: ORTValue) throws -> [Int64] {
        let data = try value.tensorData() as Data
        return data.withUnsafeBytes { Array($0.bindMemory(to: Int64.self)) }
    }

    private static func int64Value(_ values: [Int64], shape: [Int]) throws -> ORTValue {
        let data = values.withUnsafeBufferPointer { buffer in
            NSMutableData(bytes: buffer.baseAddress, length: buffer.count * MemoryLayout<Int64>.stride)
        }
        return try ORTValue(tensorData: data, elementType: .int64, shape: shape.map { NSNumber(value: $0) })
    }
}
