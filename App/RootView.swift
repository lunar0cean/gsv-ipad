import SwiftUI

struct RootView: View {
    @StateObject private var model = BenchModel()

    var body: some View {
        ZStack {
            Theme.background.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 34) {
                    header
                    section("模型") { modelList }
                    section("角色与文本") { pickers }
                    section("运行") { runControls }
                    if let stats = model.stats {
                        section("结果") { results(stats) }
                    }
                }
                .frame(maxWidth: 720, alignment: .leading)
                .padding(.horizontal, 44)
                .padding(.vertical, 40)
                .frame(maxWidth: .infinity)
            }
        }
        .preferredColorScheme(.dark)
        .tint(Theme.accent)
        .onAppear { model.refresh() }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("声间")
                .font(Theme.serif(40, weight: .semibold))
                .foregroundStyle(Theme.ink)
            Text("on-device speech · benchmark")
                .font(.system(size: 16, design: .serif).italic())
                .foregroundStyle(Theme.accent)
            Text("本机推理基准测试　ONNX Runtime \(model.runtimeVersion)")
                .font(Theme.serif(14))
                .foregroundStyle(Theme.dim)
        }
    }

    private var modelList: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(model.files) { file in
                HStack(alignment: .firstTextBaseline) {
                    Text(file.name)
                        .font(Theme.mono(13))
                        .foregroundStyle(file.sizeMB == nil ? Theme.dim : Theme.ink)
                    Spacer(minLength: 16)
                    Text(file.sizeMB.map { String(format: "%.1f MB", $0) } ?? "缺少")
                        .font(Theme.mono(13))
                        .foregroundStyle(file.sizeMB == nil ? Theme.accent : Theme.dim)
                }
            }
            if !model.modelsReady {
                Text("用 iTunes 的「文件共享」或 iPad 的「文件」App，把电脑上 work\\ipad 里的 models、voices、tests 三个文件夹拷进本 App 的文件夹，再点「重新检查」。")
                    .font(Theme.serif(14))
                    .foregroundStyle(Theme.dim)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 4)
            }
            Button("重新检查") { model.refresh() }
                .buttonStyle(LineButtonStyle())
                .disabled(model.busy)
                .padding(.top, 4)
        }
    }

    private var pickers: some View {
        VStack(alignment: .leading, spacing: 12) {
            pickerRow("角色", items: model.voices, selection: $model.voice, empty: "没有找到角色包")
            pickerRow("文本", items: model.texts, selection: $model.text, empty: "没有找到文本包")
        }
    }

    private func pickerRow(_ label: String, items: [PackItem], selection: Binding<URL?>, empty: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .font(Theme.serif(15))
                .foregroundStyle(Theme.dim)
            Spacer(minLength: 16)
            if items.isEmpty {
                Text(empty)
                    .font(Theme.serif(15))
                    .foregroundStyle(Theme.dim)
            } else {
                Picker(label, selection: selection) {
                    ForEach(items) { item in
                        Text(item.name).tag(Optional(item.url))
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
            }
        }
    }

    private var runControls: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("线程数")
                    .font(Theme.serif(15))
                    .foregroundStyle(Theme.dim)
                Spacer(minLength: 16)
                Picker("线程数", selection: $model.threads) {
                    ForEach(1...5, id: \.self) { count in
                        Text("\(count)").tag(count)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 260)
                .disabled(model.busy)
            }
            HStack(spacing: 28) {
                Button("合成") { model.run() }
                    .buttonStyle(LineButtonStyle())
                    .disabled(!model.canRun)
                    .opacity(model.canRun ? 1 : 0.35)
                if model.busy {
                    ProgressView()
                }
                Text(model.status)
                    .font(Theme.serif(15))
                    .foregroundStyle(Theme.dim)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func results(_ stats: SynthStats) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            if let load = model.loadSeconds {
                row("载入模型", String(format: "%.2f 秒", load))
            }
            row("音素 / 语义", "\(stats.phones) / \(stats.tokens)")
            row("尝试次数", "\(stats.attempts)")
            row("编码器", String(format: "%.2f 秒", stats.encoderSeconds))
            row("解码循环", String(format: "%.2f 秒", stats.decoderSeconds))
            row("声码器", String(format: "%.2f 秒", stats.vocoderSeconds))
            row("音频时长", String(format: "%.2f 秒", stats.audioSeconds))
            row("实时率（越小越快）", String(format: "%.2f", stats.realTimeFactor))
            row("内存占用", String(format: "%.0f MB", model.memoryMB))
            HStack(spacing: 28) {
                Button("再听一次") { model.play() }
                    .buttonStyle(LineButtonStyle())
                if let output = model.output {
                    ShareLink(item: output) {
                        Text("导出音频")
                    }
                    .buttonStyle(LineButtonStyle())
                }
            }
            .padding(.top, 6)
        }
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .font(Theme.serif(15))
                .foregroundStyle(Theme.dim)
            Spacer(minLength: 16)
            Text(value)
                .font(Theme.mono(15))
                .foregroundStyle(Theme.ink)
        }
    }

    private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 10) {
                Text("◆")
                    .font(.system(size: 7))
                    .foregroundStyle(Theme.accent)
                Text(title)
                    .font(Theme.serif(17, weight: .semibold))
                    .foregroundStyle(Theme.ink)
                Rectangle()
                    .fill(Theme.line)
                    .frame(height: 0.5)
            }
            content()
        }
    }
}
