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
