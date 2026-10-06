import Foundation

enum WavWriter {
    /// 单声道 16 位 PCM。峰值超过 1 时整体压回去，和电脑端的做法一致。
    static func write(samples: [Float], sampleRate: Int, to url: URL) throws {
        let peak = max(1, samples.reduce(Float(0)) { max($0, $1.isFinite ? abs($1) : 0) })
        var pcm = [Int16](repeating: 0, count: samples.count)
        for index in samples.indices where samples[index].isFinite {
            pcm[index] = Int16(max(-1, min(1, samples[index] / peak)) * 32767)
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

/// 系统记在本 App 名下的内存，和 iPadOS 判断是否超限用的是同一个数。
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
