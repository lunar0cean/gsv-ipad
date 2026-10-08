import SwiftUI
import UIKit
import UniformTypeIdentifiers

/// 在 iPad 上加角色：选一段录音，填上录音里说的话，生成角色包。
struct AddVoiceView: View {
    @ObservedObject var model: AppModel
    /// 还没有任何角色时这里是第一屏，没有「取消」
    let canCancel: Bool
    let onDone: () -> Void

    @State private var name = ""
    @State private var language = "ja"
    @State private var audio: URL?
    @State private var audioLabel = ""
    @State private var seconds: Double?
    @State private var transcript = ""
    @State private var importing = false

    var body: some View {
        VStack(alignment: .leading, spacing: 30) {
            LabeledRow(title: "添加角色") {
                Text("准备一段 3～10 秒的录音：只有一个人在说话，没有背景音乐。新角色朗读时的音色和语气都照着这段录音。")
                    .font(Theme.serif(15))
                    .foregroundStyle(Theme.dim)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if model.voiceModelsMissing.isEmpty {
                form
            } else {
                missingModels
            }
        }
        .fileImporter(isPresented: $importing, allowedContentTypes: [.audio]) { result in
            switch result {
            case .success(let url):
                importFromFiles(url)
            case .failure(let error):
                model.status = "没有选上：\(error.localizedDescription)"
            }
        }
    }

    @ViewBuilder
    private var form: some View {
        LabeledRow(title: "录音") {
            VStack(alignment: .leading, spacing: 14) {
                if !model.recordings.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 28) {
                            ForEach(model.recordings, id: \.self) { url in
                                ChoiceText(title: url.lastPathComponent, selected: url == audio, disabled: model.busy) {
                                    choose(url, label: url.lastPathComponent, transcriptFrom: url)
                                }
                            }
                        }
                    }
                }
                Button("从「文件」App 里选…") { importing = true }
                    .buttonStyle(QuietButtonStyle())
                    .disabled(model.busy)
                Text(audioHint)
                    .font(Theme.serif(13))
                    .foregroundStyle(Theme.dim)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }

        LabeledRow(title: "名字") {
            TextField("", text: $name, prompt: Text("比如：小町鸫 平静").foregroundStyle(Theme.dim.opacity(0.6)))
                .font(Theme.serif(19))
                .foregroundStyle(Theme.ink)
                .padding(.bottom, 6)
                .overlay(alignment: .bottom) { Rectangle().fill(Theme.line).frame(height: 0.5) }
                .disabled(model.busy)
        }

        LabeledRow(title: "录音里说的是") {
            HStack(spacing: 28) {
                ChoiceText(title: "日文", selected: language == "ja", disabled: model.busy) { language = "ja" }
                ChoiceText(title: "中文", selected: language == "zh", disabled: model.busy) { language = "zh" }
            }
        }

        LabeledRow(title: "录音里说的话（要和录音一字不差）") {
            ZStack(alignment: .topLeading) {
                if transcript.isEmpty {
                    Text("把录音里说的话原样写在这里")
                        .font(Theme.serif(17))
                        .foregroundStyle(Theme.dim.opacity(0.6))
                        .padding(.top, 8)
                        .padding(.leading, 5)
                        .allowsHitTesting(false)
                }
                TextEditor(text: $transcript)
                    .font(Theme.serif(17))
                    .foregroundStyle(Theme.ink)
                    .scrollContentBackground(.hidden)
                    .frame(minHeight: 110)
                    .disabled(model.busy)
            }
            .padding(.vertical, 8)
            .overlay(alignment: .top) { Rectangle().fill(Theme.line).frame(height: 0.5) }
            .overlay(alignment: .bottom) { Rectangle().fill(Theme.line).frame(height: 0.5) }
        }

        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 30) {
                Button("生成角色") { create() }
                    .buttonStyle(LineButtonStyle())
                    .disabled(!canCreate)
                    .opacity(canCreate ? 1 : 0.35)
                if canCancel && !model.busy {
                    Button("取消") { onDone() }
                        .buttonStyle(LineButtonStyle())
                }
                if model.busy {
                    ProgressView()
                }
            }
            Text(model.status)
                .font(Theme.serif(15))
                .foregroundStyle(Theme.dim)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var missingModels: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("在 iPad 上加角色还差这些文件")
                .font(Theme.serif(17, weight: .semibold))
                .foregroundStyle(Theme.ink)
            ForEach(model.voiceModelsMissing, id: \.self) { name in
                HStack(spacing: 10) {
                    Text("◆")
                        .font(.system(size: 7))
                        .foregroundStyle(Theme.accent)
                    Text(name)
                        .font(Theme.mono(14))
                        .foregroundStyle(Theme.dim)
                }
            }
            Text("用数据线连上电脑，打开「Apple 设备」，在「文件」里选「GPT Sovits」，用「添加文件」把电脑上 work\\ipad\\voice 里的 4 个文件（约 350MB）加进来，再回到这里。也可以照旧把电脑上做好的角色包（.gsvpack）拷进来。")
                .font(Theme.serif(15))
                .foregroundStyle(Theme.dim)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 6)
            HStack(spacing: 30) {
                Button("重新检查") { model.refresh() }
                    .buttonStyle(LineButtonStyle())
                if canCancel {
                    Button("取消") { onDone() }
                        .buttonStyle(LineButtonStyle())
                }
            }
        }
    }

    private var audioHint: String {
        guard audio != nil else {
            return "用数据线拷进本 App 的录音会列在上面；录音旁边放一个同名的 .txt，写上录音里说的话，会自动填好。"
        }
        guard let seconds else {
            return "\(audioLabel) 读不出来，换一个文件试试。"
        }
        let length = String(format: "%.1f 秒", seconds)
        if seconds > VoiceBuilder.maxSeconds {
            return "\(audioLabel)：\(length)，太长了（最多 30 秒），请换一段 3～10 秒的。"
        }
        if seconds < VoiceBuilder.minSeconds {
            return "\(audioLabel)：\(length)，太短了，请换一段 3～10 秒的。"
        }
        if seconds < 3 || seconds > 10 {
            return "\(audioLabel)：\(length)。能用，不过 3～10 秒的效果最好。"
        }
        return "\(audioLabel)：\(length)"
    }

    private var canCreate: Bool {
        guard !model.busy, audio != nil, let seconds,
              seconds >= VoiceBuilder.minSeconds, seconds <= VoiceBuilder.maxSeconds else { return false }
        return !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// `transcriptFrom` 是去找同名 .txt 的位置；从「文件」App 选的录音旁边的文件读不到，传 nil。
    private func choose(_ url: URL, label: String, transcriptFrom source: URL?) {
        audio = url
        audioLabel = label
        seconds = AudioLoader.duration(of: url)
        if name.trimmingCharacters(in: .whitespaces).isEmpty {
            name = (label as NSString).deletingPathExtension
        }
        if let source, let text = model.transcript(for: source) {
            transcript = text
        }
    }

    /// 「文件」App 给的位置只在这一会儿能读，先拷一份到临时文件夹。
    private func importFromFiles(_ url: URL) {
        let accessed = url.startAccessingSecurityScopedResource()
        defer {
            if accessed {
                url.stopAccessingSecurityScopedResource()
            }
        }
        let copy = FileManager.default.temporaryDirectory
            .appendingPathComponent("reference-\(UUID().uuidString)")
            .appendingPathExtension(url.pathExtension)
        do {
            try FileManager.default.copyItem(at: url, to: copy)
            choose(copy, label: url.lastPathComponent, transcriptFrom: nil)
        } catch {
            model.status = "读不了这个文件：\(error.localizedDescription)"
        }
    }

    private func create() {
        guard canCreate, let audio else { return }
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
        model.addVoice(name: name.trimmingCharacters(in: .whitespacesAndNewlines), language: language, audio: audio,
                       transcript: transcript.trimmingCharacters(in: .whitespacesAndNewlines)) { ok in
            if ok {
                onDone()
            }
        }
    }
}
