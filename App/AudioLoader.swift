import AVFoundation
import Foundation

enum AudioLoadError: LocalizedError {
    case unreadable(String)
    case empty
    case conversion(String)

    var errorDescription: String? {
        switch self {
        case .unreadable(let detail): return "读不了这个音频文件：\(detail)"
        case .empty: return "音频文件是空的"
        case .conversion(let detail): return "音频重采样失败：\(detail)"
        }
    }
}

/// 读参考音频：支持 wav、mp3、m4a、flac 等系统能解码的格式。
enum AudioLoader {
    /// 文件的时长（秒）；读不出来时返回 nil。
    static func duration(of url: URL) -> Double? {
        guard let file = try? AVAudioFile(forReading: url), file.fileFormat.sampleRate > 0 else { return nil }
        return Double(file.length) / file.fileFormat.sampleRate
    }

    /// 读整个文件并混成单声道，返回文件原始采样率下的波形。
    static func loadMono(_ url: URL) throws -> (samples: [Float], sampleRate: Double) {
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw AudioLoadError.unreadable(error.localizedDescription)
        }
        let format = file.processingFormat
        guard file.length > 0,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw AudioLoadError.empty
        }
        do {
            try file.read(into: buffer)
        } catch {
            throw AudioLoadError.unreadable(error.localizedDescription)
        }
        guard let channels = buffer.floatChannelData, buffer.frameLength > 0 else {
            throw AudioLoadError.empty
        }
        let count = Int(buffer.frameLength)
        let channelCount = Int(format.channelCount)
        var mono = [Float](repeating: 0, count: count)
        for channel in 0..<channelCount {
            let source = channels[channel]
            for index in 0..<count {
                mono[index] += source[index]
            }
        }
        if channelCount > 1 {
            let scale = 1 / Float(channelCount)
            for index in 0..<count {
                mono[index] *= scale
            }
        }
        return (mono, format.sampleRate)
    }

    /// 单声道重采样。
    static func resample(_ samples: [Float], from sourceRate: Double, to targetRate: Double) throws -> [Float] {
        if sourceRate == targetRate || samples.isEmpty {
            return samples
        }
        guard let sourceFormat = AVAudioFormat(standardFormatWithSampleRate: sourceRate, channels: 1),
              let targetFormat = AVAudioFormat(standardFormatWithSampleRate: targetRate, channels: 1),
              let converter = AVAudioConverter(from: sourceFormat, to: targetFormat),
              let input = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: AVAudioFrameCount(samples.count)),
              let inputChannel = input.floatChannelData?[0] else {
            throw AudioLoadError.conversion("无法建立转换器（\(Int(sourceRate)) → \(Int(targetRate))）")
        }
        samples.withUnsafeBufferPointer { source in
            if let base = source.baseAddress {
                inputChannel.update(from: base, count: samples.count)
            }
        }
        input.frameLength = AVAudioFrameCount(samples.count)
        // 默认的预读方式输出与输入在时间上对齐（没有滤波器延迟）
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue

        let chunk: AVAudioFrameCount = 65_536
        guard let output = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: chunk),
              let outputChannel = output.floatChannelData?[0] else {
            throw AudioLoadError.conversion("无法分配输出缓冲区")
        }
        var result: [Float] = []
        result.reserveCapacity(Int(Double(samples.count) * targetRate / sourceRate) + 1024)
        var supplied = false
        while true {
            var conversionError: NSError?
            let status = converter.convert(to: output, error: &conversionError) { _, inputStatus in
                if supplied {
                    inputStatus.pointee = .endOfStream
                    return nil
                }
                supplied = true
                inputStatus.pointee = .haveData
                return input
            }
            if status == .error {
                throw AudioLoadError.conversion(conversionError?.localizedDescription ?? "未知错误")
            }
            result.append(contentsOf: UnsafeBufferPointer(start: outputChannel, count: Int(output.frameLength)))
            if status == .endOfStream || output.frameLength == 0 {
                break
            }
        }
        return result
    }
}
