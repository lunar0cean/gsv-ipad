import SwiftUI

struct RootView: View {
    @StateObject private var model = AppModel()
    @FocusState private var editing: Bool

    var body: some View {
        ZStack {
            Theme.background.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 30) {
                    header
                    if model.ready {
                        presetRow
                        languageRow
                        editor
                        actions
                    } else {
                        setupGuide
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
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("完成") { editing = false }
            }
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
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 28) {
                    ForEach(model.presets) { preset in
                        choice(preset.name, selected: preset.url == model.selected) {
                            model.select(preset)
                        }
                    }
                }
            }
        }
    }

    private var languageRow: some View {
        labeled("输入的文字是") {
            HStack(spacing: 28) {
                choice("日文", selected: model.language == "ja") { model.language = "ja" }
                choice("中文", selected: model.language == "zh") { model.language = "zh" }
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
            Text("用 iTunes 的「文件共享」或 iPad 的「文件」App，把电脑上 work\\ipad 里的 models 和 voices 两个文件夹拷进本 App 的文件夹，再点「重新检查」。")
                .font(Theme.serif(15))
                .foregroundStyle(Theme.dim)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 6)
            Button("重新检查") { model.refresh() }
                .buttonStyle(LineButtonStyle())
        }
    }

    private func labeled<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                Text("◆")
                    .font(.system(size: 7))
                    .foregroundStyle(Theme.accent)
                Text(title)
                    .font(Theme.serif(14))
                    .foregroundStyle(Theme.dim)
            }
            content()
        }
    }

    private func choice(_ title: String, selected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(Theme.serif(19, weight: selected ? .semibold : .regular))
                .foregroundStyle(selected ? Theme.ink : Theme.dim)
                .padding(.bottom, 6)
                .overlay(alignment: .bottom) {
                    Rectangle()
                        .fill(selected ? Theme.accent : Color.clear)
                        .frame(height: 1)
                }
        }
        .buttonStyle(.plain)
        .disabled(model.busy)
    }
}
