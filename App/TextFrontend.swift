import Foundation
import JavaScriptCore

struct TextSegment: Decodable {
    let text: String
    let ids: [Int64]
    let pause: Double
}

enum FrontendError: LocalizedError {
    case scriptMissing(String)
    case failed(String)

    var errorDescription: String? {
        switch self {
        case .scriptMissing(let name): return "App 里缺少文本处理脚本 \(name)"
        case .failed(let message): return "文本处理出错：\(message)"
        }
    }
}

/// 文本 -> 音素编号。逻辑在 Frontend/*.js 里（电脑上用 Node 对照原版测过），
/// 日文的分词、读音和重音由 native/ 里的 Rust 库提供。
final class TextFrontend {
    private final class ErrorBox {
        var message: String?
    }

    private struct Reply: Decodable {
        let segments: [TextSegment]?
        let error: String?
    }

    private static let scripts = ["data", "zh", "frontend"]
    private static let optionalScripts: Set<String> = ["zh"]

    private let context: JSContext
    private let prepareFunction: JSValue
    private let errors: ErrorBox

    init() throws {
        guard let context = JSContext() else {
            throw FrontendError.failed("无法创建 JavaScript 环境")
        }
        let errors = ErrorBox()
        context.exceptionHandler = { _, exception in
            errors.message = exception?.toString() ?? "未知错误"
        }

        let jaLabels: @convention(block) (String) -> String = { text in
            guard let pointer = gsv_ja_labels(text) else { return "" }
            defer { gsv_free(pointer) }
            return String(cString: pointer)
        }
        context.setObject(unsafeBitCast(jaLabels, to: AnyObject.self), forKeyedSubscript: "__jaLabels" as NSString)

        // 大块的词典数据是 JSON 文件，脚本用到时才读
        let loadText: @convention(block) (String) -> String = { name in
            guard let url = Bundle.main.url(forResource: name, withExtension: nil),
                  let text = try? String(contentsOf: url, encoding: .utf8) else {
                return ""
            }
            return text
        }
        context.setObject(unsafeBitCast(loadText, to: AnyObject.self), forKeyedSubscript: "__loadText" as NSString)

        for name in TextFrontend.scripts {
            guard let url = Bundle.main.url(forResource: name, withExtension: "js") else {
                if TextFrontend.optionalScripts.contains(name) {
                    continue
                }
                throw FrontendError.scriptMissing("\(name).js")
            }
            context.evaluateScript(try String(contentsOf: url, encoding: .utf8), withSourceURL: url)
            if let message = errors.message {
                throw FrontendError.failed(message)
            }
        }

        guard let function = context.objectForKeyedSubscript("GSV")?.objectForKeyedSubscript("prepareJSON"),
              !function.isUndefined else {
            throw FrontendError.failed("脚本里没有 GSV.prepareJSON")
        }
        self.context = context
        self.prepareFunction = function
        self.errors = errors
    }

    /// 整段文字 -> 可以逐个合成的片段。language 是 "ja" 或 "zh"。
    func prepare(_ text: String, language: String) throws -> [TextSegment] {
        errors.message = nil
        guard let json = prepareFunction.call(withArguments: [text, language])?.toString(),
              let data = json.data(using: .utf8) else {
            throw FrontendError.failed(errors.message ?? "没有返回结果")
        }
        let reply = try JSONDecoder().decode(Reply.self, from: data)
        if let error = reply.error {
            throw FrontendError.failed(error)
        }
        return reply.segments ?? []
    }
}
