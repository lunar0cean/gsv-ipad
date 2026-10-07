import AVFoundation
import Foundation

/// 每个片段合成后的收尾处理，与 gsv_tts.TTS 的 _trim_audio、_fade 做法相同。
enum AudioPost {
    private static let frameLength = 512
    private static let hopLength = 256
    private static let searchLength = 6400
    private static let fadeLength = 3200

    static func trimAndFade(_ input: [Float], sampleRate: Int) -> [Float] {
        var audio = input.map { $0.isFinite ? $0 : 0 }
        guard !audio.isEmpty else { return audio }

        let peak = audio.reduce(Float(0)) { max($0, abs($1)) }
        if peak > 1 {
            for index in audio.indices {
                audio[index] /= peak
            }
        }

        // 开头从最安静的一帧起算，结尾去掉静音；裁完不足 0.2 秒就不裁
        let head = quietestOffset(audio)
        let end = audio.count - silentTailLength(audio, peak: min(peak, 1))
        if end - head >= min(audio.count, sampleRate / 5) {
            audio = Array(audio[head..<end])
        }

        let fade = min(fadeLength, audio.count / 2)
        if fade > 1 {
            for index in 0..<fade {
                let gain = Float(index) / Float(fade - 1)
                audio[index] *= gain
                audio[audio.count - fade + index] *= 1 - gain
            }
        }
        return audio
    }

    private static func frameRMS(_ audio: ArraySlice<Float>) -> [Float] {
        guard audio.count >= frameLength else { return [] }
        let samples = Array(audio)
        return stride(from: 0, through: samples.count - frameLength, by: hopLength).map { start in
            var sum: Float = 0
            for index in start..<(start + frameLength) {
                sum += samples[index] * samples[index]
            }
            return (sum / Float(frameLength)).squareRoot()
        }
    }

    private static func quietestOffset(_ audio: [Float]) -> Int {
        let rms = frameRMS(audio.prefix(searchLength))
        guard let quietest = rms.indices.min(by: { rms[$0] < rms[$1] }) else { return 0 }
        return quietest * hopLength
    }

    private static func silentTailLength(_ audio: [Float], peak: Float) -> Int {
        let tail = audio.suffix(searchLength)
        let rms = frameRMS(tail)
        guard let lastVoiced = rms.lastIndex(where: { $0 > 0.01 * peak }) else { return 0 }
        return tail.count - (lastVoiced * hopLength + frameLength)
    }
}

/// 边合成边播放：每个片段合成完就排进播放队列。
final class StreamPlayer {
    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private let format: AVAudioFormat

    init?(sampleRate: Int) {
        guard let format = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 1) else {
            return nil
        }
        self.format = format
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
    }

    func enqueue(_ samples: [Float]) throws {
        guard !samples.isEmpty,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)),
              let channel = buffer.floatChannelData?[0] else {
            return
        }
        samples.withUnsafeBufferPointer { source in
            if let base = source.baseAddress {
                channel.update(from: base, count: samples.count)
            }
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)

        if !engine.isRunning {
            try AVAudioSession.sharedInstance().setCategory(.playback)
            try AVAudioSession.sharedInstance().setActive(true)
            try engine.start()
        }
        node.scheduleBuffer(buffer, completionHandler: nil)
        if !node.isPlaying {
            node.play()
        }
    }

    func stop() {
        node.stop()
    }
}

/// 系统记在本 App 名下的内存，和 iPadOS 判断是否超限用的是同一个数。写进日志，方便判断模型装不装得下。
func memoryFootprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return result == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : 0
}

enum WavWriter {
    /// 单声道 16 位 PCM。
    static func write(samples: [Float], sampleRate: Int, to url: URL) throws {
        var pcm = [Int16](repeating: 0, count: samples.count)
        for index in samples.indices where samples[index].isFinite {
            pcm[index] = Int16(max(-1, min(1, samples[index])) * 32767)
        }

        let byteCount = pcm.count * MemoryLayout<Int16>.stride
        var data = Data(capacity: 44 + byteCount)
        func append<T: FixedWidthInteger>(_ value: T) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        data.append(contentsOf: "RIFF".utf8)
        append(UInt32(36 + byteCount))
        data.append(contentsOf: "WAVEfmt ".utf8)
        append(UInt32(16))
        append(UInt16(1))
        append(UInt16(1))
        append(UInt32(sampleRate))
        append(UInt32(sampleRate * 2))
        append(UInt16(2))
        append(UInt16(16))
        data.append(contentsOf: "data".utf8)
        append(UInt32(byteCount))
        pcm.withUnsafeBytes { data.append(contentsOf: $0) }
        try data.write(to: url, options: .atomic)
    }
}
