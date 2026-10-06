import Foundation
import OnnxRuntimeBindings

/// tools/export_pack.py 写出的 .gsvpack 数据包，格式说明见那个脚本的开头。
struct TensorPack {
    struct Entry: Decodable {
        let name: String
        let dtype: String
        let shape: [Int]
        let offset: Int
        let length: Int
    }

    private struct Header: Decodable {
        let kind: String
        let meta: [String: String]
        let tensors: [Entry]
    }

    enum PackError: LocalizedError {
        case notAPack(String)
        case missingTensor(String)

        var errorDescription: String? {
            switch self {
            case .notAPack(let file): return "\(file) 不是有效的数据包"
            case .missingTensor(let name): return "数据包里缺少 \(name)"
            }
        }
    }

    let url: URL
    let kind: String
    let meta: [String: String]
    private let entries: [String: Entry]
    private let data: Data
    private let blobStart: Int

    init(url: URL) throws {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count >= 8, data.prefix(4).elementsEqual("GSVP".utf8) else {
            throw PackError.notAPack(url.lastPathComponent)
        }
        var headerLength = 0
        for index in 0..<4 {
            headerLength |= Int(data[4 + index]) << (8 * index)
        }
        guard data.count >= 8 + headerLength else {
            throw PackError.notAPack(url.lastPathComponent)
        }
        let header = try JSONDecoder().decode(Header.self, from: data.subdata(in: 8..<(8 + headerLength)))

        self.url = url
        self.kind = header.kind
        self.meta = header.meta
        self.entries = Dictionary(header.tensors.map { ($0.name, $0) }, uniquingKeysWith: { first, _ in first })
        self.data = data
        self.blobStart = 8 + headerLength
    }

    func shape(of name: String) -> [Int] {
        entries[name]?.shape ?? []
    }

    func value(_ name: String) throws -> ORTValue {
        guard let entry = entries[name], blobStart + entry.offset + entry.length <= data.count else {
            throw PackError.missingTensor(name)
        }
        let start = blobStart + entry.offset
        let bytes = NSMutableData(data: data.subdata(in: start..<(start + entry.length)))
        let elementType: ORTTensorElementDataType = entry.dtype == "i64" ? .int64 : .float
        return try ORTValue(tensorData: bytes, elementType: elementType, shape: entry.shape.map { NSNumber(value: $0) })
    }
}
