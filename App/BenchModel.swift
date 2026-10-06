import AVFoundation
import Foundation
import OnnxRuntimeBindings

struct ModelFileStatus: Identifiable {
    let name: String
    let sizeMB: Double?
    var id: String { name }
}

struct PackItem: Identifiable {
    let url: URL
    let name: String
    var id: URL { url }
}

/// 基准测试界面的状态。推理在后台队列上跑，界面状态只在主线程上改。
final class BenchModel: ObservableObject {
    @Published var files: [ModelFileStatus] = []
    @Published var voices: [PackItem] = []
    @Published var texts: [PackItem] = []
    @Published var voice: URL?
    @Published var text: URL?
    @Published var threads = 2
    @Published var busy = false
    @Published var status = "还没有运行"
    @Published var loadSeconds: Double?
    @Published var stats: SynthStats?
    @Published var memoryMB = 0.0
    @Published var output: URL?

    let runtimeVersion = ORTVersion() ?? "未知"

    private var modelDirectory: URL?
    private var engine: SynthEngine?
    private var engineThreads = 0
    private var player: AVAudioPlayer?
    private let worker = DispatchQueue(label: "gsvpad.synth", qos: .userInitiated)

    var modelsReady: Bool { modelDirectory != nil }
    var canRun: Bool { !busy && modelsReady && voice != nil && text != nil }

    static var documents: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    /// 在本 App 的文件夹里找模型和数据包，放在哪一层子文件夹都可以。
    func refresh() {
        let fm = FileManager.default
        let docs = BenchModel.documents
        BenchModel.writeGuide(in: docs)

        var foundModelDirectory: URL?
        var foundVoices: [PackItem] = []
        var foundTexts: [PackItem] = []
        if let walker = fm.enumerator(at: docs, includingPropertiesForKeys: nil) {
            for case let url as URL in walker {
                if url.lastPathComponent == "vits_fp32.onnx" {
                    foundModelDirectory = url.deletingLastPathComponent()
                }
                guard url.pathExtension == "gsvpack", let pack = try? TensorPack(url: url) else {
                    continue
                }
                let item = PackItem(url: url, name: pack.meta["name"] ?? url.deletingPathExtension().lastPathComponent)
                if pack.kind == "voice" {
                    foundVoices.append(item)
                } else if pack.kind == "text" {
                    foundTexts.append(item)
                }
            }
        }

        let directory = foundModelDirectory ?? docs.appendingPathComponent("models", isDirectory: true)
        files = SynthEngine.modelFiles.map { name in
            let attributes = try? fm.attributesOfItem(atPath: directory.appendingPathComponent(name).path)
            let bytes = (attributes?[.size] as? NSNumber)?.doubleValue
            return ModelFileStatus(name: name, sizeMB: bytes.map { $0 / 1_048_576 })
        }
        let ready = files.allSatisfy { $0.sizeMB != nil }
        if modelDirectory != (ready ? directory : nil) {
            engine = nil
            modelDirectory = ready ? directory : nil
        }

        voices = foundVoices.sorted { $0.url.lastPathComponent < $1.url.lastPathComponent }
        texts = foundTexts.sorted { $0.url.lastPathComponent < $1.url.lastPathComponent }
        if !voices.contains(where: { $0.url == voice }) {
            voice = voices.first?.url
        }
        if !texts.contains(where: { $0.url == text }) {
            text = texts.first?.url
        }
    }

    func run() {
        guard canRun, let directory = modelDirectory, let voiceURL = voice, let textURL = text else {
            return
        }
        busy = true
        stats = nil
        output = nil
        player?.stop()

        // 换线程数要重建推理会话。先放掉旧的，避免两套模型同时占内存
        let threads = self.threads
        let cached = engineThreads == threads ? engine : nil
        if cached == nil {
            engine = nil
        }
        status = cached == nil ? "正在载入模型…" : "正在合成…"

        worker.async { [weak self] in
            do {
                var loadSeconds: Double?
                let engine: SynthEngine
                if let cached {
                    engine = cached
                } else {
                    let start = CFAbsoluteTimeGetCurrent()
                    engine = try SynthEngine(modelDirectory: directory, threads: threads)
                    loadSeconds = CFAbsoluteTimeGetCurrent() - start
                    DispatchQueue.main.async { self?.status = "正在合成…" }
                }

                let result = try engine.synthesize(voice: TensorPack(url: voiceURL), text: TensorPack(url: textURL))
                let memory = memoryFootprintMB()

                let folder = BenchModel.documents.appendingPathComponent("outputs", isDirectory: true)
                try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                let file = folder.appendingPathComponent("bench_\(Int(Date().timeIntervalSince1970)).wav")
                try WavWriter.write(samples: result.samples, sampleRate: SynthEngine.sampleRate, to: file)

                DispatchQueue.main.async {
                    guard let self else { return }
                    self.engine = engine
                    self.engineThreads = threads
                    if let loadSeconds {
                        self.loadSeconds = loadSeconds
                    }
                    self.stats = result.stats
                    self.memoryMB = memory
                    self.output = file
                    self.status = "完成"
                    self.busy = false
                    self.play()
                }
            } catch {
                DispatchQueue.main.async {
                    self?.status = "出错：\(error.localizedDescription)"
                    self?.busy = false
                }
            }
        }
    }

    func play() {
        guard let output else { return }
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback)
            try AVAudioSession.sharedInstance().setActive(true)
            let player = try AVAudioPlayer(contentsOf: output)
            self.player = player
            player.play()
        } catch {
            status = "播放失败：\(error.localizedDescription)"
        }
    }

    /// 文件夹里有东西，它才会出现在「文件」App 和 iTunes 的文件共享里。
    private static func writeGuide(in docs: URL) {
        let guide = docs.appendingPathComponent("使用说明.txt")
        guard !FileManager.default.fileExists(atPath: guide.path) else { return }
        let text = """
        把电脑上 gsv-ipad\\work\\ipad 里的三个文件夹拷到这里：
          models  模型（约 570MB）
          voices  角色包
          tests   测试文本包
        拷完回到 App 点「重新检查」。合成的音频保存在 outputs 文件夹。
        """
        try? text.write(to: guide, atomically: true, encoding: .utf8)
    }
}
