import SwiftUI

/// 冷色：夜蓝底、冷白字、一个淡蓝强调色。没有底板和卡片，字直接压在背景上。
enum Theme {
    static let ink = Color(red: 0.90, green: 0.93, blue: 0.98)
    static let dim = Color(red: 0.60, green: 0.68, blue: 0.80)
    static let accent = Color(red: 0.62, green: 0.80, blue: 1.00)
    static let line = Color.white.opacity(0.16)

    static let background = LinearGradient(
        colors: [
            Color(red: 0.03, green: 0.05, blue: 0.12),
            Color(red: 0.07, green: 0.11, blue: 0.22),
            Color(red: 0.04, green: 0.06, blue: 0.14),
        ],
        startPoint: .top,
        endPoint: .bottom
    )

    static func serif(_ size: CGFloat, weight: Font.Weight = .regular) -> Font {
        Font.custom("Songti SC", size: size).weight(weight)
    }

    static func mono(_ size: CGFloat) -> Font {
        Font.system(size: size, design: .monospaced)
    }
}

struct LineButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(Theme.serif(17, weight: .semibold))
            .foregroundStyle(Theme.accent)
            .padding(.vertical, 6)
            .overlay(alignment: .bottom) {
                Rectangle().fill(Theme.accent.opacity(0.7)).frame(height: 0.5)
            }
            .opacity(configuration.isPressed ? 0.5 : 1)
    }
}
