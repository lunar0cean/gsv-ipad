import XCTest
@testable import GSVPad

/// 在 iPad 模拟器上跑的冒烟测试。测试数据由 GitHub 上的编译流程现场生成，
/// 目录通过环境变量 GSV_FIXTURES 传进来；没有这个变量时跳过。
final class SmokeTests: XCTestCase {
    private struct ExpectedCase: Decodable {
        let text: String
        let lang: String
        let segments: [ExpectedSegment]
    }

    private struct ExpectedSegment: Decodable {
        let text: String
        let ids: [Int64]
        let pause: Double
        let bertIds: [Int64]?
        let bertRepeats: [Int]?
    }

    private func fixtures() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["GSV_FIXTURES"], !path.isEmpty else {
            throw XCTSkip("没有设置 GSV_FIXTURES")
        }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    private func elapsed(since start: CFAbsoluteTime) -> String {
        String(format: "%.0f 毫秒", (CFAbsoluteTimeGetCurrent() - start) * 1000)
    }

    /// 文本处理：JavaScriptCore 里的结果要和 Node 算出的标准答案逐个音素相同。
    func testFrontendMatchesNode() throws {
        let url = try fixtures().appendingPathComponent("expected.json")
        let cases = try JSONDecoder().decode([ExpectedCase].self, from: Data(contentsOf: url))

        var clock = CFAbsoluteTimeGetCurrent()
        let frontend = try TextFrontend()
        print("GSVTEST 载入文本处理脚本 \(elapsed(since: clock))")

        var mismatches = 0
        for language in ["ja", "zh"] {
            let subset = cases.filter { $0.lang == language }
            clock = CFAbsoluteTimeGetCurrent()
            _ = try frontend.prepare(subset[0].text, language: language)
            print("GSVTEST [\(language)] 第一句（含载入词典）\(elapsed(since: clock))")

            clock = CFAbsoluteTimeGetCurrent()
            for item in subset {
                let got = try frontend.prepare(item.text, language: language)
                let same = got.count == item.segments.count && zip(got, item.segments).allSatisfy { pair in
                    pair.0.ids == pair.1.ids && pair.0.text == pair.1.text && abs(pair.0.pause - pair.1.pause) < 1e-9
                        && pair.0.bertIds == pair.1.bertIds && pair.0.bertRepeats == pair.1.bertRepeats
                }
                if !same {
                    mismatches += 1
                    print("GSVTEST 不一致 [\(language)] \(item.text)")
                }
            }
            print("GSVTEST [\(language)] \(subset.count) 段文字 \(elapsed(since: clock))")
        }
        XCTAssertEqual(mismatches, 0, "JavaScriptCore 与 Node 的结果不一致")
    }

    /// 推理：用随机权重的同结构模型，从文字一路跑到写出音频文件。
    func testSynthesisRunsEndToEnd() throws {
        let directory = try fixtures()
        var clock = CFAbsoluteTimeGetCurrent()
        let engine = try SynthEngine(modelDirectory: directory.appendingPathComponent("models"), threads: 2,
                                     maxSteps: 40)
        print("GSVTEST 载入模型 \(elapsed(since: clock))")

        let voice = try TensorPack(url: directory.appendingPathComponent("voice.gsvpack"))
        XCTAssertEqual(voice.kind, "voice")
        let frontend = try TextFrontend()
        let segments = try frontend.prepare("今日はいい天気ですね。一緒に海まで散歩しませんか。", language: "ja")
        XCTAssertEqual(segments.count, 2)

        var all: [Float] = []
        for segment in segments {
            clock = CFAbsoluteTimeGetCurrent()
            let result = try XCTUnwrap(try engine.synthesize(voice: voice, phonemes: segment.ids,
                                                             isCancelled: { false }))
            XCTAssertGreaterThan(result.stats.tokens, 0)
            XCTAssertGreaterThan(result.samples.count, 0)
            XCTAssertTrue(result.samples.allSatisfy { $0.isFinite }, "声码器输出里有非有限数值")
            print("GSVTEST 合成一句：音素 \(result.stats.phones)，语义 \(result.stats.tokens)，"
                  + "采样 \(result.samples.count)，尝试 \(result.stats.attempts) 次，\(elapsed(since: clock))")
            all.append(contentsOf: AudioPost.trimAndFade(result.samples, sampleRate: SynthEngine.sampleRate))
        }

        let file = FileManager.default.temporaryDirectory.appendingPathComponent("gsv-smoke.wav")
        try WavWriter.write(samples: all, sampleRate: SynthEngine.sampleRate, to: file)
        let size = try XCTUnwrap(try FileManager.default.attributesOfItem(atPath: file.path)[.size] as? NSNumber)
        XCTAssertEqual(size.intValue, 44 + all.count * 2)

        // 取消：第一步之前就要求停止，应当返回 nil 而不是报错
        let cancelled = try engine.synthesize(voice: voice, phonemes: segments[0].ids, isCancelled: { true })
        XCTAssertNil(cancelled)

        // 中文语调模型：按字的特征要展开成按音素的特征，长度必须正好对上，再带着它合成一句
        clock = CFAbsoluteTimeGetCurrent()
        let bert = try BertEngine(modelURL: directory.appendingPathComponent(BertEngine.fileName), threads: 2)
        print("GSVTEST 载入语调模型 \(elapsed(since: clock))")
        let chinese = try XCTUnwrap(try frontend.prepare("今天的天气真不错，我们一起去海边走走吧。", language: "zh").first)
        let bertIds = try XCTUnwrap(chinese.bertIds, "中文片段应当带语调模型的输入")
        let bertRepeats = try XCTUnwrap(chinese.bertRepeats)
        XCTAssertEqual(bertIds.count, bertRepeats.count + 2)
        XCTAssertEqual(bertRepeats.reduce(0, +), chinese.ids.count)

        clock = CFAbsoluteTimeGetCurrent()
        let features = try bert.features(ids: bertIds, repeats: bertRepeats)
        print("GSVTEST 语调特征：\(bertIds.count - 2) 个字 -> \(chinese.ids.count) 个音素，\(elapsed(since: clock))")
        XCTAssertEqual(features.length, chinese.ids.count * BertEngine.width * MemoryLayout<Float>.stride)
        let values = UnsafeBufferPointer(start: features.bytes.assumingMemoryBound(to: Float.self),
                                         count: features.length / MemoryLayout<Float>.stride)
        XCTAssertTrue(values.allSatisfy { $0.isFinite }, "语调特征里有非有限数值")
        XCTAssertTrue(values.contains { $0 != 0 }, "语调特征不应当全是零")

        let withBert = try XCTUnwrap(try engine.synthesize(voice: voice, phonemes: chinese.ids, textBert: features,
                                                           isCancelled: { false }))
        XCTAssertGreaterThan(withBert.samples.count, 0)
        XCTAssertThrowsError(try bert.features(ids: bertIds, repeats: Array(bertRepeats.dropLast())), "长度对不上应当报错")
    }

    private func readFloats(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// 音频增强：连续处理三个片段，结果要与参考实现（tools/enhance_ref.py）一致。
    /// 参考实现在电脑上和网页界面用的 pedalboard + pyloudnorm 对照过。
    func testEnhancerMatchesReference() throws {
        let directory = try fixtures()
        let enhancer = Enhancer(sampleRate: 32_000)
        for index in 1...3 {
            let input = try readFloats(directory.appendingPathComponent("enhance_in_\(index).f32"))
            let expected = try readFloats(directory.appendingPathComponent("enhance_out_\(index).f32"))
            let clock = CFAbsoluteTimeGetCurrent()
            let output = enhancer.process(input)
            XCTAssertEqual(output.count, expected.count)

            var errorPower = 0.0
            var signalPower = 0.0
            for (got, want) in zip(output, expected) {
                errorPower += Double(got - want) * Double(got - want)
                signalPower += Double(want) * Double(want)
            }
            let relative = (errorPower / max(signalPower, 1e-12)).squareRoot()
            print("GSVTEST 音频增强片段 \(index)：\(input.count) 个采样，相对误差 "
                  + String(format: "%.2e", relative) + "，\(elapsed(since: clock))")
            XCTAssertLessThan(relative, 1e-3, "第 \(index) 个片段与参考实现不一致")
        }
        let loudness = try XCTUnwrap(enhancer.integratedLoudness)
        print("GSVTEST 音频增强前的累计响度 " + String(format: "%.2f LUFS", loudness))
    }

    private struct ExpectedReference: Decodable {
        let text: String
        let lang: String
        let ids: [Int64]
        let bertIds: [Int64]?
        let bertRepeats: [Int]?
    }

    /// 新建角色时参考文字的处理：与 Node 算出的标准答案相同。
    func testReferenceTextMatchesNode() throws {
        let url = try fixtures().appendingPathComponent("expected_reference.json")
        let cases = try JSONDecoder().decode([ExpectedReference].self, from: Data(contentsOf: url))
        let frontend = try TextFrontend()
        var mismatches = 0
        for item in cases {
            let got = try frontend.reference(item.text, language: item.lang)
            if got.ids != item.ids || got.bertIds != item.bertIds || got.bertRepeats != item.bertRepeats {
                mismatches += 1
                print("GSVTEST 参考文字不一致 [\(item.lang)] \(item.text)")
            }
        }
        print("GSVTEST 参考文字 \(cases.count) 句，不一致 \(mismatches) 句")
        XCTAssertEqual(mismatches, 0)
        XCTAssertThrowsError(try frontend.reference("「……」", language: "ja"), "只有标点时应当报错")
    }

    /// 声纹模型的 fbank 特征：与参考实现（tools/fbank_ref.py，和 torchaudio 对照过）一致。
    func testFbankMatchesReference() throws {
        let directory = try fixtures()
        let input = try readFloats(directory.appendingPathComponent("fbank_in.f32"))
        let expected = try readFloats(directory.appendingPathComponent("fbank_out.f32"))
        let clock = CFAbsoluteTimeGetCurrent()
        let result = Fbank.compute(input)
        XCTAssertEqual(result.features.count, expected.count)
        let worst = zip(result.features, expected).map { abs($0 - $1) }.max() ?? .infinity
        print("GSVTEST fbank：\(result.frames) 帧，最大误差 " + String(format: "%.2e", worst) + "，\(elapsed(since: clock))")
        XCTAssertLessThan(worst, 1e-2)
    }

    /// 结尾静音的长度：1 秒正弦波后面跟 0.2 秒静音，按原版的帧划分应当正好是 2816 个采样。
    func testTailOffset() {
        var audio = (0..<16_000).map { sin(Float($0) * 0.1) * 0.5 }
        audio.append(contentsOf: [Float](repeating: 0, count: 3_200))
        XCTAssertEqual(VoiceBuilder.tailOffset(audio), 2_816)
        XCTAssertEqual(VoiceBuilder.tailOffset([Float](repeating: 0, count: 10_000)), 0)
        XCTAssertEqual(VoiceBuilder.tailOffset([0.1, 0.2]), 0)
    }

    /// 在 iPad 上新建角色：读双声道 44.1kHz 的录音，跑三个（随机权重的）模型，写出角色包，再用它合成一句。
    func testVoiceBuiltOnDevice() throws {
        let directory = try fixtures()
        let wav = directory.appendingPathComponent("ref_stereo.wav")
        let (mono, rate) = try AudioLoader.loadMono(wav)
        XCTAssertEqual(rate, 44_100)
        XCTAssertEqual(mono.count, 132_300)
        let resampled = try AudioLoader.resample(mono, from: rate, to: 16_000)
        XCTAssertEqual(Double(resampled.count), Double(mono.count) * 16_000 / 44_100, accuracy: 64)
        func rms(_ values: [Float]) -> Double {
            (values.reduce(0.0) { $0 + Double($1) * Double($1) } / Double(max(1, values.count))).squareRoot()
        }
        print("GSVTEST 重采样 44100 -> 16000：\(mono.count) -> \(resampled.count) 个采样，音量比 "
              + String(format: "%.3f", rms(resampled) / rms(mono)))
        XCTAssertEqual(rms(resampled) / rms(mono), 1, accuracy: 0.1)
        XCTAssertEqual(AudioLoader.duration(of: wav) ?? 0, 3, accuracy: 0.01)
        XCTAssertThrowsError(try AudioLoader.loadMono(directory.appendingPathComponent("expected.json")))

        let voiceModels = directory.appendingPathComponent("voice_models")
        let models = Dictionary(uniqueKeysWithValues: ["hubert.onnx", "sv.onnx", "prompt_encoder_fp32.onnx"].map {
            ($0, voiceModels.appendingPathComponent($0))
        })
        let frontend = try TextFrontend()
        let reference = try frontend.reference("今天的天气真不错。", language: "zh")
        let bert = try BertEngine(modelURL: directory.appendingPathComponent(BertEngine.fileName), threads: 2)
        let refBert = try bert.features(ids: try XCTUnwrap(reference.bertIds), repeats: try XCTUnwrap(reference.bertRepeats))

        var steps: [String] = []
        let clock = CFAbsoluteTimeGetCurrent()
        let tensors = try VoiceBuilder.build(audio: wav, reference: reference, refBert: refBert, models: models,
                                             threads: 2) { steps.append($0) }
        print("GSVTEST 新建角色：\(steps.joined(separator: " ")) \(elapsed(since: clock))")
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("gsv-built.gsvpack")
        try TensorPack.write(to: file, kind: "voice", meta: ["name": "新角色", "lang": "zh"], tensors: tensors)

        let pack = try TensorPack(url: file)
        XCTAssertEqual(pack.kind, "voice")
        XCTAssertEqual(pack.meta["name"], "新角色")
        XCTAssertEqual(pack.shape(of: "ref_seq"), [1, reference.ids.count])
        XCTAssertEqual(pack.shape(of: "ref_bert"), [reference.ids.count, BertEngine.width])
        XCTAssertEqual(pack.shape(of: "ge"), [1, 1024, 1])
        XCTAssertEqual(pack.shape(of: "ge_advanced"), [1, 512, 1])
        let ssl = pack.shape(of: "ssl_content")
        print("GSVTEST 角色包：参考音素 \(reference.ids.count) 个，语气特征 \(ssl)")
        XCTAssertEqual(ssl.count, 3)
        XCTAssertEqual(ssl.first, 1)
        XCTAssertEqual(ssl.dropFirst().first, 768)
        XCTAssertGreaterThan(ssl.last ?? 0, 100)  // 3 秒加 0.3 秒静音，每 20 毫秒一帧

        // 读回来的内容与写进去的相同
        let ids = try pack.value("ref_seq")
        let storedIds: [Int64] = try withExtendedLifetime(ids) {
            let raw = try ids.tensorData()
            return Array(UnsafeBufferPointer(start: raw.bytes.assumingMemoryBound(to: Int64.self), count: raw.length / 8))
        }
        XCTAssertEqual(storedIds, reference.ids)
        let storedBert = try pack.value("ref_bert")
        withExtendedLifetime(storedBert) {
            XCTAssertEqual(try storedBert.tensorData() as Data, refBert as Data)
        }

        let engine = try SynthEngine(modelDirectory: directory.appendingPathComponent("models"), threads: 2, maxSteps: 40)
        let text = try XCTUnwrap(try frontend.prepare("你好，很高兴认识你。", language: "zh").first)
        let result = try XCTUnwrap(try engine.synthesize(voice: pack, phonemes: text.ids, isCancelled: { false }))
        XCTAssertGreaterThan(result.samples.count, 0)
        XCTAssertTrue(result.samples.allSatisfy { $0.isFinite })
    }

    /// 音频收尾处理的边界情况：空输入、很短的输入、含非有限数值的输入都不能崩溃。
    func testAudioPostHandlesEdgeCases() {
        XCTAssertEqual(AudioPost.trimAndFade([], sampleRate: 32_000).count, 0)
        XCTAssertEqual(AudioPost.trimAndFade([0.5], sampleRate: 32_000).count, 1)
        let noisy = (0..<20_000).map { index -> Float in index % 997 == 0 ? .nan : sin(Float(index) * 0.05) * 3 }
        let processed = AudioPost.trimAndFade(noisy, sampleRate: 32_000)
        XCTAssertFalse(processed.isEmpty)
        XCTAssertTrue(processed.allSatisfy { $0.isFinite && abs($0) <= 1.0001 })
    }
}
