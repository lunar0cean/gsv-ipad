import Foundation
import UIKit

struct VoicePreset: Identifiable {
    let url: URL
    let name: String
    let language: String
    var id: URL { url }
}

/// 界面状态。合成在后台队列上跑，界面状态只在主线程上改。
final class AppModel: ObservableObject {
    @Published var presets: [VoicePreset] = []
    @Published var selected: URL?
    @Published var language = "ja"
    @Published var text = ""
    @Published var busy = false
    @Published var status = ""
    @Published var missingFiles: [String] = []
    @Published var output: URL?
    /// 文档目录里有没有中文语调模型。它是可选的，没有时中文也能读，只是语气偏平
    @Published var bertInstalled = false
    /// 音频增强（均衡、压缩、统一响度），与电脑上网页界面的同名选项相同。默认打开，选择会记住
    @Published var enhance: Bool = UserDefaults.standard.object(forKey: "enhance") as? Bool ?? true {
        didSet { UserDefaults.standard.set(enhance, forKey: "enhance") }
    }

    private var modelDirectory: URL?
    private var bertURL: URL?
    private var engine: SynthEngine?
    private var bertEngine: BertEngine?
    private var frontend: TextFrontend?
    private var lastSamples: [Float] = []
    private let player = StreamPlayer(sampleRate: SynthEngine.sampleRate)
    private let worker = DispatchQueue(label: "gptsovits.synth", qos: .userInitiated)
    private let cancelLock = NSLock()
    private var cancelRequested = false

    // A16 有 2 个性能核，推理线程数与之对应
    private static let threads = 2

    var ready: Bool { modelDirectory != nil && !presets.isEmpty }
    var hasAudio: Bool { !lastSamples.isEmpty }
    var canGenerate: Bool {
        !busy && ready && selected != nil && !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    static var documents: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    /// 在本 App 的文件夹里找模型和角色包，放在哪一层子文件夹都可以。
    func refresh() {
        let fm = FileManager.default
        let docs = AppModel.documents
        AppModel.writeGuide(in: docs)

        var foundDirectory: URL?
        var foundBert: URL?
        var found: [VoicePreset] = []
        if let walker = fm.enumerator(at: docs, includingPropertiesForKeys: nil) {
            for case let url as URL in walker {
                if url.lastPathComponent == "vits_fp32.onnx" {
                    foundDirectory = url.deletingLastPathComponent()
                }
                if url.lastPathComponent == BertEngine.fileName {
                    foundBert = url
                }
                guard url.pathExtension == "gsvpack", let pack = try? TensorPack(url: url), pack.kind == "voice" else {
                    continue
                }
                found.append(VoicePreset(url: url,
                                         name: pack.meta["name"] ?? url.deletingPathExtension().lastPathComponent,
                                         language: pack.meta["lang"] ?? "ja"))
            }
        }

        let directory = foundDirectory ?? docs.appendingPathComponent("models", isDirectory: true)
        let missingModels = SynthEngine.modelFiles.filter {
            !fm.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
        missingFiles = missingModels + (found.isEmpty ? ["voices 文件夹里的角色包（.gsvpack）"] : [])

        let newDirectory = missingModels.isEmpty ? directory : nil
        if modelDirectory != newDirectory {
            engine = nil
            modelDirectory = newDirectory
        }
        if bertURL != foundBert {
            bertEngine = nil
            bertURL = foundBert
        }
        bertInstalled = foundBert != nil

        // 文件名前面的序号决定显示顺序
        presets = found.sorted { $0.url.lastPathComponent < $1.url.lastPathComponent }
        if !presets.contains(where: { $0.url == selected }), let first = presets.first {
            select(first)
        }
        if self.ready, engine == nil, !busy {
            warmUp()
        }
    }

    func select(_ preset: VoicePreset) {
        selected = preset.url
        if preset.language == "ja" || preset.language == "zh" {
            language = preset.language
        }
    }

    /// 提前把模型和文本处理载入内存，这样第一次点「生成」不用等。
    private func warmUp() {
        guard let directory = modelDirectory else { return }
        status = "正在载入模型…"
        let needsChinese = presets.contains { $0.language == "zh" }
        worker.async { [weak self] in
            guard let self else { return }
            do {
                let (frontend, _) = try self.loadIfNeeded(directory: directory)
                if needsChinese {
                    // 中文词典第一次用到时要花几秒载入，趁现在做掉
                    _ = try? frontend.prepare("你好。", language: "zh")
                }
                DispatchQueue.main.async {
                    if !self.busy {
                        self.status = "就绪"
                    }
                }
            } catch {
                DispatchQueue.main.async { self.status = "载入失败：\(error.localizedDescription)" }
            }
        }
    }

    /// 只在后台队列上调用。队列是串行的，所以不会同时载入两份模型。
    private func loadIfNeeded(directory: URL) throws -> (TextFrontend, SynthEngine) {
        let cached = DispatchQueue.main.sync { (self.frontend, self.engine) }
        let frontend = try cached.0 ?? TextFrontend()
        let engine = try cached.1 ?? SynthEngine(modelDirectory: directory, threads: AppModel.threads)
        DispatchQueue.main.sync {
            self.frontend = frontend
            self.engine = engine
        }
        return (frontend, engine)
    }

    /// 只在后台队列上调用。中文语调模型载入后占 1GB 以上内存，所以只在第一次合成中文时才载入。
    private func loadBertIfNeeded(url: URL) throws -> BertEngine {
        if let cached = DispatchQueue.main.sync(execute: { self.bertEngine }) {
            return cached
        }
        DispatchQueue.main.async { self.status = "正在载入中文语调模型…" }
        let start = CFAbsoluteTimeGetCurrent()
        let loaded = try BertEngine(modelURL: url, threads: AppModel.threads)
        AppModel.log(String(format: "中文语调模型载入用时 %.1f 秒，内存占用 %.0f MB",
                            CFAbsoluteTimeGetCurrent() - start, memoryFootprintMB()))
        DispatchQueue.main.sync { self.bertEngine = loaded }
        return loaded
    }

    func generate() {
        guard canGenerate, let directory = modelDirectory, let voiceURL = selected else { return }
        busy = true
        output = nil
        setCancelled(false)
        player?.stop()
        status = engine == nil ? "正在载入模型…" : "正在处理文字…"
        // 长文本要合成几分钟。期间不让屏幕自动锁定：锁屏后系统会把 App 挂起，合成和播放都会停下
        UIApplication.shared.isIdleTimerDisabled = true

        let text = self.text
        let language = self.language
        let bertURL = language == "zh" ? self.bertURL : nil
        let enhancer = enhance ? Enhancer(sampleRate: SynthEngine.sampleRate) : nil
        worker.async { [weak self] in
            guard let self else { return }
            var all: [Float] = []
            var seconds = 0.0
            var cancelled = false
            var skipped: [Int] = []
            var failure: String?
            do {
                let (frontend, engine) = try self.loadIfNeeded(directory: directory)
                let segments = try frontend.prepare(text, language: language)
                guard !segments.isEmpty else {
                    throw FrontendError.failed("没有可以合成的文字")
                }
                let voice = try TensorPack(url: voiceURL)

                // 语调模型载入失败不影响合成，退回全零特征
                var bert: BertEngine?
                if let bertURL {
                    do {
                        bert = try self.loadBertIfNeeded(url: bertURL)
                    } catch {
                        AppModel.log("中文语调模型载入失败：\(error.localizedDescription)")
                    }
                }
                AppModel.log("开始：\(segments.count) 句，语言 \(language)，角色包 \(voiceURL.lastPathComponent)，"
                    + "中文语调模型 \(bert == nil ? "未使用" : "已使用")，音频增强 \(enhancer == nil ? "关" : "开")")

                for (index, segment) in segments.enumerated() {
                    DispatchQueue.main.async { self.status = "正在合成第 \(index + 1) / \(segments.count) 句" }

                    var textBert: NSMutableData?
                    if let bert, let ids = segment.bertIds, let repeats = segment.bertRepeats {
                        do {
                            textBert = try bert.features(ids: ids, repeats: repeats)
                        } catch {
                            AppModel.log("第 \(index + 1) 句的语调特征没算出来：\(error.localizedDescription)")
                        }
                    }

                    // 一句出错不拖累后面的句子：重试一次，还不行就跳过这一句
                    var samples: [Float]?
                    for attempt in 1...2 {
                        do {
                            guard let result = try engine.synthesize(voice: voice, phonemes: segment.ids,
                                                                     textBert: textBert,
                                                                     isCancelled: { self.isCancelled }) else {
                                cancelled = true
                                break
                            }
                            seconds += result.stats.seconds
                            samples = AudioPost.trimAndFade(result.samples, sampleRate: SynthEngine.sampleRate)
                            AppModel.log("第 \(index + 1) 句：音素 \(result.stats.phones)，语义 \(result.stats.tokens)，"
                                + "内部重试 \(result.stats.attempts - 1) 次，"
                                + String(format: "用时 %.1f 秒，音频 %.1f 秒", result.stats.seconds, result.stats.audioSeconds))
                            break
                        } catch {
                            AppModel.log("第 \(index + 1) 句第 \(attempt) 次失败：\(error.localizedDescription)｜\(segment.text)")
                        }
                    }
                    if cancelled {
                        break
                    }
                    guard var samples else {
                        skipped.append(index + 1)
                        continue
                    }
                    let pause = Int(segment.pause * Double(SynthEngine.sampleRate))
                    samples.append(contentsOf: [Float](repeating: 0, count: max(0, pause)))
                    // 连同后面的停顿一起处理，混响的尾音才有地方落
                    if let enhancer {
                        samples = enhancer.process(samples)
                    }
                    all.append(contentsOf: samples)
                    do {
                        try self.player?.enqueue(samples)
                    } catch {
                        // 播放不了不影响合成，最后还能「再听一次」或导出
                        AppModel.log("第 \(index + 1) 句播放失败：\(error.localizedDescription)")
                    }
                }
            } catch {
                failure = error.localizedDescription
                AppModel.log("中断：\(error.localizedDescription)")
            }

            var file: URL?
            if !all.isEmpty {
                do {
                    let folder = AppModel.documents.appendingPathComponent("outputs", isDirectory: true)
                    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                    let formatter = DateFormatter()
                    formatter.dateFormat = "yyyyMMdd_HHmmss"
                    let url = folder.appendingPathComponent("\(formatter.string(from: Date())).wav")
                    try WavWriter.write(samples: all, sampleRate: SynthEngine.sampleRate, to: url)
                    file = url
                } catch {
                    AppModel.log("保存音频失败：\(error.localizedDescription)")
                }
            }

            let audioSeconds = Double(all.count) / Double(SynthEngine.sampleRate)
            var message: String
            if let failure {
                message = failure
            } else if cancelled {
                message = "已停止"
            } else {
                message = String(format: "完成　音频 %.1f 秒，合成用时 %.1f 秒", audioSeconds, seconds)
                if !skipped.isEmpty {
                    message += "；第 " + skipped.map { String($0) }.joined(separator: "、") + " 句没有合成出来"
                }
            }
            AppModel.log("结束：\(message)" + String(format: "　内存占用 %.0f MB", memoryFootprintMB()))

            let finalAudio = all
            let finalFile = file
            let finalMessage = message
            DispatchQueue.main.async {
                UIApplication.shared.isIdleTimerDisabled = false
                self.lastSamples = finalAudio
                self.output = finalFile
                self.status = finalMessage
                self.busy = false
            }
        }
    }

    /// 往 outputs/log.txt 追加一行。排查「中途停下」这类问题时看它，可以用「文件」App 或电脑上的「Apple 设备」取出来。
    private static func log(_ line: String) {
        let folder = documents.appendingPathComponent("outputs", isDirectory: true)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let url = folder.appendingPathComponent("log.txt")
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        guard let data = "\(formatter.string(from: Date()))  \(line)\n".data(using: .utf8) else { return }

        let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.intValue ?? 0
        if size > 512 * 1024 {
            try? FileManager.default.removeItem(at: url)  // 日志只留最近的一段
        }
        if let handle = try? FileHandle(forWritingTo: url) {
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: data)
        } else {
            try? data.write(to: url)
        }
    }

    func stop() {
        setCancelled(true)
        player?.stop()
        if busy {
            status = "正在停止…"
        }
    }

    func replay() {
        guard !busy, !lastSamples.isEmpty else { return }
        player?.stop()
        do {
            try player?.enqueue(lastSamples)
        } catch {
            status = "播放失败：\(error.localizedDescription)"
        }
    }

    private var isCancelled: Bool {
        cancelLock.lock()
        defer { cancelLock.unlock() }
        return cancelRequested
    }

    private func setCancelled(_ value: Bool) {
        cancelLock.lock()
        cancelRequested = value
        cancelLock.unlock()
    }

    /// 文件夹里有东西，它才会出现在「文件」App 和 iTunes 的文件共享里。
    private static func writeGuide(in docs: URL) {
        let guide = docs.appendingPathComponent("使用说明.txt")
        guard !FileManager.default.fileExists(atPath: guide.path) else { return }
        let text = """
        把电脑上 gsv-ipad\\work\\ipad 里这两个文件夹中的文件拷到这里（放不放在文件夹里都可以）：
          models  7 个模型文件（约 570MB）
          voices  角色包（.gsvpack）
        拷完回到 App 点「重新检查」。
        合成的音频保存在 outputs 文件夹；outputs\\log.txt 是合成日志，出问题时看它。
        """
        try? text.write(to: guide, atomically: true, encoding: .utf8)
    }
}
