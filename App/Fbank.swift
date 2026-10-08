import Foundation

/// Kaldi 风格的 80 维 fbank 特征，声纹模型的输入。
///
/// 原版用的是 torchaudio.compliance.kaldi.fbank(wav, num_mel_bins=80, sample_frequency=16000, dither=0)。
/// 这是 tools/fbank_ref.py 的逐步对应版本，那份参考实现和 torchaudio 对照过；
/// 改这里时两边要一起改，模拟器测试会拿参考实现的输出来比。
enum Fbank {
    static let sampleRate = 16_000
    static let melBins = 80

    private static let window = 400      // 25 毫秒
    private static let shift = 160       // 10 毫秒
    private static let padded = 512      // 补到 2 的幂做 FFT
    private static let lowFrequency = 20.0
    private static let preemphasis = 0.97
    private static let epsilon = Double(Float.ulpOfOne)

    private static func mel(_ frequency: Double) -> Double {
        1127.0 * log(1.0 + frequency / 700.0)
    }

    /// 三角滤波器组，[80][257]。在 mel 刻度上等距，最后一列（奈奎斯特频率）是零。
    private static let banks: [[Double]] = {
        let binWidth = Double(sampleRate) / Double(padded)
        let melLow = mel(lowFrequency)
        let delta = (mel(0.5 * Double(sampleRate)) - melLow) / Double(melBins + 1)
        let points = (0..<(padded / 2)).map { mel(binWidth * Double($0)) }
        return (0..<melBins).map { index in
            let left = melLow + Double(index) * delta
            let center = melLow + Double(index + 1) * delta
            let right = melLow + Double(index + 2) * delta
            return points.map { max(0, min(($0 - left) / (center - left), (right - $0) / (right - center))) } + [0]
        }
    }()

    /// Povey 窗：汉宁窗的 0.85 次方。
    private static let poveyWindow: [Double] = (0..<window).map { n in
        pow(0.5 - 0.5 * cos(2 * Double.pi * Double(n) / Double(window - 1)), 0.85)
    }

    private static let bitReversed: [Int] = (0..<padded).map { value in
        var x = value
        var reversed = 0
        for _ in 0..<9 {
            reversed = (reversed << 1) | (x & 1)
            x >>= 1
        }
        return reversed
    }
    private static let cosTable: [Double] = (0..<(padded / 2)).map { cos(-2 * Double.pi * Double($0) / Double(padded)) }
    private static let sinTable: [Double] = (0..<(padded / 2)).map { sin(-2 * Double.pi * Double($0) / Double(padded)) }

    /// 512 点的复数 FFT，原地计算。
    private static func fft(_ re: inout [Double], _ im: inout [Double]) {
        for index in 0..<padded {
            let other = bitReversed[index]
            if other > index {
                re.swapAt(index, other)
                im.swapAt(index, other)
            }
        }
        var size = 2
        while size <= padded {
            let half = size / 2
            let step = padded / size
            var start = 0
            while start < padded {
                for k in 0..<half {
                    let wr = cosTable[k * step]
                    let wi = sinTable[k * step]
                    let a = start + k
                    let b = a + half
                    let tr = re[b] * wr - im[b] * wi
                    let ti = re[b] * wi + im[b] * wr
                    re[b] = re[a] - tr
                    im[b] = im[a] - ti
                    re[a] += tr
                    im[a] += ti
                }
                start += size
            }
            size *= 2
        }
    }

    /// `waveform` 是 16kHz 单声道。返回按行排列的 [帧数 × 80] 对数 mel 能量，以及帧数。
    static func compute(_ waveform: [Float]) -> (features: [Float], frames: Int) {
        guard waveform.count >= window else { return ([], 0) }
        let frames = 1 + (waveform.count - window) / shift
        var features = [Float](repeating: 0, count: frames * melBins)
        var re = [Double](repeating: 0, count: padded)
        var im = [Double](repeating: 0, count: padded)
        var power = [Double](repeating: 0, count: padded / 2 + 1)

        for frame in 0..<frames {
            let offset = frame * shift
            var mean = 0.0
            for j in 0..<window {
                mean += Double(waveform[offset + j])
            }
            mean /= Double(window)
            for j in 0..<window {
                re[j] = Double(waveform[offset + j]) - mean   // 每帧去直流
            }
            // 预加重：从后往前算，第一个采样用自己当前一个
            for j in stride(from: window - 1, through: 1, by: -1) {
                re[j] -= preemphasis * re[j - 1]
            }
            re[0] -= preemphasis * re[0]
            for j in 0..<window {
                re[j] *= poveyWindow[j]
            }
            for j in window..<padded {
                re[j] = 0
            }
            for j in 0..<padded {
                im[j] = 0
            }
            fft(&re, &im)
            for k in 0...(padded / 2) {
                power[k] = re[k] * re[k] + im[k] * im[k]
            }
            for bin in 0..<melBins {
                let weights = banks[bin]
                var energy = 0.0
                for k in 0...(padded / 2) {
                    energy += power[k] * weights[k]
                }
                features[frame * melBins + bin] = Float(log(max(energy, epsilon)))
            }
        }
        return (features, frames)
    }
}
