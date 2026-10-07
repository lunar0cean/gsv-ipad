import Foundation
import OnnxRuntimeBindings

enum BertError: LocalizedError {
    case shape(String)

    var errorDescription: String? {
        switch self {
        case .shape(let detail): return "中文语调模型的输入输出对不上：\(detail)"
        }
    }
}

/// 中文语调模型（chinese-roberta-wwm-ext-large 的前 22 层，由 tools/export_bert.py 生成）。
/// 输入一段规范化文本里每个字的编号，输出按音素展开的特征，作为 T2S 编码器的 text_bert。
/// 这个文件是可选的：没有它时目标文本的特征用全零，中文能读但语气偏平。
final class BertEngine {
    static let fileName = "roberta_fp16.onnx"
    static let width = 1024

    private let env: ORTEnv
    private let session: ORTSession

    init(modelURL: URL, threads: Int) throws {
        let env = try ORTEnv(loggingLevel: .warning)
        let options = try ORTSessionOptions()
        try options.setIntraOpNumThreads(Int32(threads))
        try options.setGraphOptimizationLevel(.all)
        self.session = try ORTSession(env: env, modelPath: modelURL.path, sessionOptions: options)
        self.env = env
    }

    /// `ids` 含首尾的 [CLS]、[SEP]；`repeats` 是每个字对应的音素数。返回 [音素数 × 1024] 的单精度数据。
    func features(ids: [Int64], repeats: [Int]) throws -> NSMutableData {
        guard ids.count == repeats.count + 2 else {
            throw BertError.shape("\(ids.count) 个编号，\(repeats.count) 个字")
        }
        let input = ids.withUnsafeBufferPointer { buffer in
            NSMutableData(bytes: buffer.baseAddress, length: buffer.count * MemoryLayout<Int64>.stride)
        }
        let value = try ORTValue(tensorData: input, elementType: .int64, shape: [1, NSNumber(value: ids.count)])
        let outputs = try session.run(withInputs: ["input_ids": value], outputNames: ["char_features"], runOptions: nil)
        guard let output = outputs["char_features"] else {
            throw BertError.shape("模型没有给出 char_features")
        }

        // 模型按字给出特征，这里按每个字的音素数重复成按音素的特征。
        // tensorData 返回的数据直接指向 output 内部的内存，所以读的时候要保证 output 还活着
        let rowBytes = BertEngine.width * MemoryLayout<Float>.stride
        return try withExtendedLifetime(output) {
            let rows = try output.tensorData()
            guard rows.length == repeats.count * rowBytes else {
                throw BertError.shape("输出 \(rows.length) 字节，应为 \(repeats.count * rowBytes)")
            }
            let expanded = NSMutableData()
            for (index, count) in repeats.enumerated() where count > 0 {
                let row = rows.bytes.advanced(by: index * rowBytes)
                for _ in 0..<count {
                    expanded.append(row, length: rowBytes)
                }
            }
            return expanded
        }
    }
}
