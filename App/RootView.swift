import SwiftUI
import UIKit

struct RootView: View {
    @StateObject private var model = AppModel()
    @FocusState private var editing: Bool
    @Environment(\.scenePhase) private var scenePhase
    @State private var adding = false
    @State private var pendingDelete: VoicePreset?

    var body: some View {
        ZStack {
            Theme.background.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 30) {
                    header
                    if !model.modelsReady {
                        setupGuide
                    } else if adding || model.presets.isEmpty {
                        AddVoiceView(model: model, canCancel: !model.presets.isEmpty) { adding = false }
                    } else {
                        presetRow
                        languageRow
                        enhanceRow
                        editor
                        actions
                    }
                }
                .frame(maxWidth: 760, alignment: .leading)
                .padding(.horizontal, 44)
                .padding(.vertical, 40)
                .frame(maxWidth: .infinity)
            }
            .scrollDismissesKeyboard(.interactively)
        }
        .preferredColorScheme(.dark)
        .tint(Theme.accent)
        .onAppear { model.refresh() }
        // 用数据线拷完文件回到 App 时自动重新检查
        .onChange(of: scenePhase) { _, phase in
            if phase == .active && !model.busy {
                model.refresh()
            }
        }
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("完成") {
                    editing = false
                    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                }
            }
        }
        .confirmationDialog("删除角色", isPresented: Binding(get: { pendingDelete != nil },
                                                          set: { if !$0 { pendingDelete = nil } }),
                            titleVisibility: .visible, presenting: pendingDelete) { preset in
            Button("删除「\(preset.name)」", role: .destructive) { model.delete(preset) }
            Button("取消", role: .cancel) {}
        } message: { preset in
            Text("会删掉角色包 \(preset.url.lastPathComponent)，删了不能恢复。")
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("GPT Sovits")
                .font(.system(size: 40, weight: .semibold, design: .serif))
                .foregroundStyle(Theme.ink)
            Text("on-device speech")
                .font(.system(size: 16, design: .serif).italic())
                .foregroundStyle(Theme.accent)
        }
    }

    private var presetRow: some View {
        labeled("角色") {
            VStack(alignment: .leading, spacing: 14) {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 28) {
                        ForEach(model.presets) { preset in
                            choice(preset.name, selected: preset.url == model.selected) {
                                model.select(preset)
                            }
                        }
                    }
                }
                HStack(spacing: 28) {
                    Button("＋ 添加角色") { adding = true }
                        .buttonStyle(QuietButtonStyle())
                    if let current = model.presets.first(where: { $0.url == model.selected }) {
                        Button("删除这个角色") { pendingDelete = current }
                            .buttonStyle(QuietButtonStyle())
                    }
                }
                .disabled(model.busy)
            }
        }
    }

    private var languageRow: some View {
        labeled("输入的文字是") {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 28) {
                    choice("日文", selected: model.language == "ja") { model.language = "ja" }
                    choice("中文", selected: model.language == "zh") { model.language = "zh" }
                }
                if model.language == "zh" && !model.bertInstalled {
                    Text("没有找到中文语调模型（roberta_fp16.onnx）。中文可以照常合成，但停顿和语气会偏平。")
                        .font(Theme.serif(13))
                        .foregroundStyle(Theme.dim)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private var enhanceRow: some View {
        labeled("音频增强（均衡、压缩、统一响度）") {
            HStack(spacing: 28) {
                choice("开", selected: model.enhance) { model.enhance = true }
                choice("关", selected: !model.enhance) { model.enhance = false }
            }
        }
    }

    private var editor: some View {
        ZStack(alignment: .topLeading) {
            if model.text.isEmpty {
                Text("在这里输入要朗读的文字")
                    .font(Theme.serif(19))
                    .foregroundStyle(Theme.dim.opacity(0.6))
                    .padding(.top, 8)
                    .padding(.leading, 5)
                    .allowsHitTesting(false)
            }
            TextEditor(text: $model.text)
                .font(Theme.serif(19))
                .foregroundStyle(Theme.ink)
                .scrollContentBackground(.hidden)
                .focused($editing)
                .frame(minHeight: 240)
        }
        .padding(.vertical, 12)
        .overlay(alignment: .top) { Rectangle().fill(Theme.line).frame(height: 0.5) }
        .overlay(alignment: .bottom) { Rectangle().fill(Theme.line).frame(height: 0.5) }
    }

    private var actions: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 30) {
                if model.busy {
                    Button("停止") { model.stop() }
                        .buttonStyle(LineButtonStyle())
                    ProgressView()
                } else {
                    Button("生成") {
                        editing = false
                        model.generate()
                    }
                    .buttonStyle(LineButtonStyle())
                    .disabled(!model.canGenerate)
                    .opacity(model.canGenerate ? 1 : 0.35)
                    if model.hasAudio {
                        Button("再听一次") { model.replay() }
                            .buttonStyle(LineButtonStyle())
                    }
                    if let output = model.output {
                        ShareLink(item: output) {
                            Text("导出音频")
                        }
                        .buttonStyle(LineButtonStyle())
                    }
                }
            }
            Text(model.status)
                .font(Theme.serif(15))
                .foregroundStyle(Theme.dim)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var setupGuide: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("还差这些文件")
                .font(Theme.serif(17, weight: .semibold))
                .foregroundStyle(Theme.ink)
            ForEach(model.missingFiles, id: \.self) { name in
                HStack(spacing: 10) {
                    Text("◆")
                        .font(.system(size: 7))
                        .foregroundStyle(Theme.accent)
                    Text(name)
                        .font(Theme.mono(14))
                        .foregroundStyle(Theme.dim)
                }
            }
            Text("用数据线连上电脑，打开「Apple 设备」，在左边点「文件」，在 App 列表里选「GPT Sovits」，用「添加文件」把电脑上 work\\ipad\\models 里的 7 个文件加进来（不用保留文件夹），再回到这里。角色可以拷电脑上做好的角色包（voices 里的 .gsvpack），也可以之后在 iPad 上直接加。")
                .font(Theme.serif(15))
                .foregroundStyle(Theme.dim)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 6)
            Button("重新检查") { model.refresh() }
                .buttonStyle(LineButtonStyle())
        }
    }

    private func labeled<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        LabeledRow(title: title, content: content)
    }

    private func choice(_ title: String, selected: Bool, action: @escaping () -> Void) -> some View {
        ChoiceText(title: title, selected: selected, disabled: model.busy, action: action)
    }
}
