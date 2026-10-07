import Foundation

/// 音频增强，与电脑上网页界面的「启用音频增强」相同：
/// 高通 80Hz → 300Hz 提升 2.5dB → 7kHz 衰减 3dB → 压缩（-18dB，3.5:1）→ 很轻的混响 → +2dB → 响度统一到 -18 LUFS。
///
/// 这是 tools/enhance_ref.py 的逐行对应版本，那份参考实现和电脑上的 pedalboard + pyloudnorm 对照过。
/// 改这里的算法时两边要一起改，模拟器测试会拿参考实现的输出来比。
///
/// 一次合成用一个实例，每个片段调一次 `process`。滤波器、压缩器、混响的状态跨片段延续；
/// 响度按「到目前为止的全部音频」来量，再决定当前片段的增益。
final class Enhancer {
    static let targetLUFS = -18.0

    /// 转置直接 II 型，系数已按 a0 归一。
    private struct Biquad {
        let b0: Float
        let b1: Float
        let b2: Float
        let a1: Float
        let a2: Float
        var s1: Float = 0
        var s2: Float = 0

        init(_ b0: Double, _ b1: Double, _ b2: Double, _ a1: Double, _ a2: Double) {
            self.b0 = Float(b0)
            self.b1 = Float(b1)
            self.b2 = Float(b2)
            self.a1 = Float(a1)
            self.a2 = Float(a2)
        }

        mutating func process(_ samples: inout [Float]) {
            for index in samples.indices {
                let input = samples[index]
                let output = b0 * input + s1
                s1 = b1 * input - a1 * output + s2
                s2 = b2 * input - a2 * output
                samples[index] = output
            }
        }

        /// juce::dsp::IIR::Coefficients::makeFirstOrderHighPass
        static func firstOrderHighpass(rate: Double, frequency: Double) -> Biquad {
            let n = tan(Double.pi * frequency / rate)
            return Biquad(1 / (n + 1), -1 / (n + 1), 0, (n - 1) / (n + 1), 0)
        }

        /// juce::dsp::IIR::Coefficients::makePeakFilter
        static func peak(rate: Double, frequency: Double, q: Double, gainDB: Double) -> Biquad {
            let a = sqrt(pow(10, gainDB / 20))
            let omega = 2 * Double.pi * max(frequency, 2) / rate
            let alpha = sin(omega) / (q * 2)
            let c2 = -2 * cos(omega)
            let a0 = 1 + alpha / a
            return Biquad((1 + alpha * a) / a0, c2 / a0, (1 - alpha * a) / a0, c2 / a0, (1 - alpha / a) / a0)
        }

        /// 响度计的 K 加权第一级：高架
        static func weightingShelf(rate: Double) -> Biquad {
            let a = pow(10, 4.0 / 40.0)
            let w0 = 2 * Double.pi * (1500.0 / rate)
            let alpha = sin(w0) / (2 * (1 / sqrt(2.0)))
            let c = cos(w0)
            let b0 = a * ((a + 1) + (a - 1) * c + 2 * sqrt(a) * alpha)
            let b1 = -2 * a * ((a - 1) + (a + 1) * c)
            let b2 = a * ((a + 1) + (a - 1) * c - 2 * sqrt(a) * alpha)
            let a0 = (a + 1) - (a - 1) * c + 2 * sqrt(a) * alpha
            let a1 = 2 * ((a - 1) - (a + 1) * c)
            let a2 = (a + 1) - (a - 1) * c - 2 * sqrt(a) * alpha
            return Biquad(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
        }

        /// 响度计的 K 加权第二级：高通
        static func weightingHighpass(rate: Double) -> Biquad {
            let w0 = 2 * Double.pi * (38.0 / rate)
            let alpha = sin(w0) / (2 * 0.5)
            let c = cos(w0)
            let a0 = 1 + alpha
            return Biquad((1 + c) / 2 / a0, -(1 + c) / a0, (1 + c) / 2 / a0, -2 * c / a0, (1 - alpha) / a0)
        }
    }

    /// juce::dsp::Compressor：峰值包络（起音 1ms、释放 100ms），超过阈值的部分按比例压。
    private struct Compressor {
        let attack: Float
        let release: Float
        let threshold: Float
        let exponent: Float
        var envelope: Float = 0

        init(rate: Double, thresholdDB: Double, ratio: Double, attackMs: Double = 1, releaseMs: Double = 100) {
            let factor = -2 * Double.pi * 1000 / rate
            attack = Float(exp(factor / attackMs))
            release = Float(exp(factor / releaseMs))
            threshold = Float(pow(10, thresholdDB / 20))
            exponent = Float(1 / ratio - 1)
        }

        mutating func process(_ samples: inout [Float]) {
            for index in samples.indices {
                let level = abs(samples[index])
                let coefficient = level > envelope ? attack : release
                envelope = level + coefficient * (envelope - level)
                if envelope >= threshold {
                    samples[index] *= pow(envelope / threshold, exponent)
                }
            }
        }
    }

    /// juce::Reverb（Freeverb）的单声道处理。
    private struct Reverb {
        private static let combTunings = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
        private static let allpassTunings = [556, 441, 341, 225]

        var combs: [[Float]]
        var combIndex: [Int]
        var combLast: [Float]
        var allpasses: [[Float]]
        var allpassIndex: [Int]
        let damp: Float
        let feedback: Float
        let dry: Float
        let wet: Float
        let gain: Float = 0.015

        init(rate: Int, roomSize: Double, damping: Double, wet: Double, dry: Double, width: Double = 1) {
            combs = Reverb.combTunings.map { [Float](repeating: 0, count: rate * $0 / 44100) }
            combIndex = [Int](repeating: 0, count: combs.count)
            combLast = [Float](repeating: 0, count: combs.count)
            allpasses = Reverb.allpassTunings.map { [Float](repeating: 0, count: rate * $0 / 44100) }
            allpassIndex = [Int](repeating: 0, count: allpasses.count)
            damp = Float(damping * 0.4)
            feedback = Float(roomSize * 0.28 + 0.7)
            self.dry = Float(dry * 2)
            self.wet = Float(0.5 * (wet * 3) * (1 + width))
        }

        mutating func process(_ samples: inout [Float]) {
            for index in samples.indices {
                let value = samples[index] * gain
                var output: Float = 0
                for j in combs.indices {
                    let k = combIndex[j]
                    let delayed = combs[j][k]
                    let last = delayed * (1 - damp) + combLast[j] * damp
                    combLast[j] = last
                    combs[j][k] = value + last * feedback
                    combIndex[j] = (k + 1) % combs[j].count
                    output += delayed
                }
                for j in allpasses.indices {
                    let k = allpassIndex[j]
                    let delayed = allpasses[j][k]
                    allpasses[j][k] = output + delayed * 0.5
                    allpassIndex[j] = (k + 1) % allpasses[j].count
                    output = delayed - output
                }
                samples[index] = output * wet + samples[index] * dry
            }
        }
    }

    private static let blockSeconds = 0.4
    private static let hopSeconds = 0.1

    private let rate: Int
    private var highpass: Biquad
    private var warmth: Biquad
    private var deEss: Biquad
    private var compressor: Compressor
    private var reverb: Reverb
    private let gain = Float(pow(10, 2.0 / 20))
    private var weightingShelf: Biquad
    private var weightingHighpass: Biquad
    /// 到目前为止所有 0.4 秒块的均方值（K 加权后）
    private var blocks: [Double] = []
    private var lastGain: Float?

    init(sampleRate: Int) {
        rate = sampleRate
        let r = Double(sampleRate)
        highpass = .firstOrderHighpass(rate: r, frequency: 80)
        warmth = .peak(rate: r, frequency: 300, q: 1, gainDB: 2.5)
        deEss = .peak(rate: r, frequency: 7000, q: 2, gainDB: -3)
        compressor = Compressor(rate: r, thresholdDB: -18, ratio: 3.5)
        reverb = Reverb(rate: sampleRate, roomSize: 0.1, damping: 0.5, wet: 0.03, dry: 0.97)
        weightingShelf = .weightingShelf(rate: r)
        weightingHighpass = .weightingHighpass(rate: r)
    }

    /// 到目前为止全部音频的响度（BS.1770 的门限算法）：先去掉 -70 LUFS 以下的块，再去掉比平均低 10 dB 以上的块。
    var integratedLoudness: Double? {
        func loudness(_ meanSquare: Double) -> Double { -0.691 + 10 * log10(meanSquare) }
        let audible = blocks.filter { loudness($0) >= -70 }
        guard !audible.isEmpty else { return nil }
        let relative = loudness(audible.reduce(0, +) / Double(audible.count)) - 10
        let gated = blocks.filter { loudness($0) > relative && loudness($0) > -70 }
        guard !gated.isEmpty else { return nil }
        return loudness(gated.reduce(0, +) / Double(gated.count))
    }

    func process(_ input: [Float]) -> [Float] {
        var samples = input
        highpass.process(&samples)
        warmth.process(&samples)
        deEss.process(&samples)
        compressor.process(&samples)
        reverb.process(&samples)
        for index in samples.indices {
            samples[index] *= gain
        }

        measure(samples)
        if let loudness = integratedLoudness {
            lastGain = Float(pow(10, (Enhancer.targetLUFS - loudness) / 20))
        }
        let peak = samples.reduce(Float(0)) { max($0, abs($1)) }
        guard let lastGain else {
            // 开头就是不足 0.4 秒的片段，量不出响度：先按峰值 -6dB 处理
            return peak > 0 ? samples.map { $0 * (0.5 / peak) } : samples
        }
        // 响度统一后个别峰值可能超过 1，整段按比例压回去，避免削波
        let scaledPeak = peak * lastGain
        let factor = scaledPeak > 1 ? lastGain / scaledPeak : lastGain
        return samples.map { $0 * factor }
    }

    private func measure(_ shaped: [Float]) {
        var weighted = shaped
        weightingShelf.process(&weighted)
        weightingHighpass.process(&weighted)
        let block = Int(Enhancer.blockSeconds * Double(rate))
        let hop = Int(Enhancer.hopSeconds * Double(rate))
        guard weighted.count >= block else { return }
        // 与 pyloudnorm 相同的块数算法：最后一块可能超出数据末尾，仍按整块长度求均值
        let seconds = Double(weighted.count) / Double(rate)
        let count = Int(((seconds - Enhancer.blockSeconds) / Enhancer.hopSeconds).rounded(.toNearestOrEven)) + 1
        for j in 0..<count {
            let start = j * hop
            guard start < weighted.count else { break }
            let end = min(start + block, weighted.count)
            var sum = 0.0
            for index in start..<end {
                sum += Double(weighted[index]) * Double(weighted[index])
            }
            blocks.append(sum / Double(block))
        }
    }
}
