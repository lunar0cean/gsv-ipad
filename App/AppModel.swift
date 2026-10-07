import Foundation

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

    private var modelDirectory: URL?
    private var engine: SynthEngine?
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
        var found: [VoicePreset] = []
        if let walker = fm.enumerator(at: docs, includingPropertiesForKeys: nil) {
            for case let url as URL in walker {
                if url.lastPathComponent == "vits_fp32.onnx" {
                    foundDirectory = url.deletingLastPathComponent()
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

    func generate() {
        guard canGenerate, let directory = modelDirectory, let voiceURL = selected else { return }
        busy = true
        output = nil
        setCancelled(false)
        player?.stop()
        status = engine == nil ? "正在载入模型…" : "正在处理文字…"

        let text = self.text
        let language = self.language
        worker.async { [weak self] in
            guard let self else { return }
            do {
                let (frontend, engine) = try self.loadIfNeeded(directory: directory)
                let segments = try frontend.prepare(text, language: language)
                guard !segments.isEmpty else {
                    throw FrontendError.failed("没有可以合成的文字")
                }
                let voice = try TensorPack(url: voiceURL)

                var all: [Float] = []
                var seconds = 0.0
                var cancelled = false
                for (index, segment) in segments.enumerated() {
                    DispatchQueue.main.async { self.status = "正在合成第 \(index + 1) / \(segments.count) 句" }
                    guard let result = try engine.synthesize(voice: voice, phonemes: segment.ids,
                                                             isCancelled: { self.isCancelled }) else {
                        cancelled = true
                        break
                    }
                    var samples = AudioPost.trimAndFade(result.samples, sampleRate: SynthEngine.sampleRate)
                    let pause = Int(segment.pause * Double(SynthEngine.sampleRate))
                    samples.append(contentsOf: [Float](repeating: 0, count: max(0, pause)))
                    seconds += result.stats.seconds
                    all.append(contentsOf: samples)
                    try self.player?.enqueue(samples)
                }

                var file: URL?
                if !all.isEmpty {
                    let folder = AppModel.documents.appendingPathComponent("outputs", isDirectory: true)
                    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                    let formatter = DateFormatter()
                    formatter.dateFormat = "yyyyMMdd_HHmmss"
                    let url = folder.appendingPathComponent("\(formatter.string(from: Date())).wav")
                    try WavWriter.write(samples: all, sampleRate: SynthEngine.sampleRate, to: url)
                    file = url
                }

                let audioSeconds = Double(all.count) / Double(SynthEngine.sampleRate)
                DispatchQueue.main.async {
                    self.lastSamples = all
                    self.output = file
                    self.status = cancelled
                        ? "已停止"
                        : String(format: "完成　音频 %.1f 秒，合成用时 %.1f 秒", audioSeconds, seconds)
                    self.busy = false
                }
            } catch {
                DispatchQueue.main.async {
                    self.status = error.localizedDescription
                    self.busy = false
                }
            }
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
        把电脑上 gsv-ipad\\work\\ipad 里的两个文件夹拷到这里：
          models  模型（约 570MB）
          voices  角色包
        拷完回到 App 点「重新检查」。合成的音频保存在 outputs 文件夹。
        """
        try? text.write(to: guide, atomically: true, encoding: .utf8)
    }
}
