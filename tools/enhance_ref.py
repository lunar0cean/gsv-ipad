"""音频增强的参考实现。只依赖 numpy。

电脑上的网页界面（D:\\k\\web.py 的 enhance_audio）用 pedalboard 和 pyloudnorm 做这件事：
    高通 80Hz -> 300Hz 提升 2.5dB -> 7kHz 衰减 3dB -> 压缩（-18dB，3.5:1）-> 很轻的混响 -> +2dB -> 响度统一到 -18 LUFS
这里用 numpy 把同样的算法逐样本写一遍，作用有两个：
  - 在电脑上和 pedalboard 的输出对照，确认算法抄对了（python enhance_ref.py 文件.wav）
  - App\\Enhancer.swift 照着它写，GitHub 上的测试用它生成标准答案

各环节的公式来自 JUCE（pedalboard 的底层）和 pyloudnorm 的源码。

与网页界面的一处不同：那边是整段音频合成完再一起处理，iPad 上是合成一句播一句，所以这里是流式的。
滤波器、压缩器、混响的状态跨片段延续；响度按「到目前为止的全部音频」来量，再决定当前片段的增益，
这样第一句之后各句之间的相对响度不变，和整段处理的结果接近。
"""
import sys

import numpy as np

TARGET_LUFS = -18.0


class Biquad:
    """转置直接 II 型，系数已按 a0 归一。"""

    def __init__(self, b0, b1, b2, a1, a2):
        self.c = np.array([b0, b1, b2, a1, a2], dtype=np.float32)
        self.s1 = np.float32(0)
        self.s2 = np.float32(0)

    def process(self, x: np.ndarray) -> np.ndarray:
        b0, b1, b2, a1, a2 = self.c
        s1, s2 = self.s1, self.s2
        y = np.empty_like(x)
        for i in range(len(x)):
            v = x[i]
            out = b0 * v + s1
            s1 = b1 * v - a1 * out + s2
            s2 = b2 * v - a2 * out
            y[i] = out
        self.s1, self.s2 = s1, s2
        return y


def first_order_highpass(rate: float, frequency: float) -> Biquad:
    # juce::dsp::IIR::Coefficients::makeFirstOrderHighPass
    n = np.tan(np.pi * frequency / rate)
    return Biquad(1 / (n + 1), -1 / (n + 1), 0.0, (n - 1) / (n + 1), 0.0)


def peak_filter(rate: float, frequency: float, q: float, gain_db: float) -> Biquad:
    # juce::dsp::IIR::Coefficients::makePeakFilter
    a = np.sqrt(10 ** (gain_db / 20))
    omega = 2 * np.pi * max(frequency, 2.0) / rate
    alpha = np.sin(omega) / (q * 2)
    c2 = -2 * np.cos(omega)
    a0 = 1 + alpha / a
    return Biquad((1 + alpha * a) / a0, c2 / a0, (1 - alpha * a) / a0, c2 / a0, (1 - alpha / a) / a0)


def k_weighting(rate: float) -> list:
    # pyloudnorm.Meter 的 K-weighting：先高架后高通
    def shelf(g, q, fc):
        a = 10 ** (g / 40.0)
        w0 = 2.0 * np.pi * (fc / rate)
        alpha = np.sin(w0) / (2.0 * q)
        cos = np.cos(w0)
        b0 = a * ((a + 1) + (a - 1) * cos + 2 * np.sqrt(a) * alpha)
        b1 = -2 * a * ((a - 1) + (a + 1) * cos)
        b2 = a * ((a + 1) + (a - 1) * cos - 2 * np.sqrt(a) * alpha)
        a0 = (a + 1) - (a - 1) * cos + 2 * np.sqrt(a) * alpha
        a1 = 2 * ((a - 1) - (a + 1) * cos)
        a2 = (a + 1) - (a - 1) * cos - 2 * np.sqrt(a) * alpha
        return Biquad(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)

    def highpass(q, fc):
        w0 = 2.0 * np.pi * (fc / rate)
        alpha = np.sin(w0) / (2.0 * q)
        cos = np.cos(w0)
        a0 = 1 + alpha
        return Biquad((1 + cos) / 2 / a0, -(1 + cos) / a0, (1 + cos) / 2 / a0, -2 * cos / a0, (1 - alpha) / a0)

    return [shelf(4.0, 1 / np.sqrt(2), 1500.0), highpass(0.5, 38.0)]


class Compressor:
    """juce::dsp::Compressor：峰值包络（起音 1ms、释放 100ms），超过阈值的部分按比例压。"""

    def __init__(self, rate: float, threshold_db: float, ratio: float, attack_ms: float = 1.0, release_ms: float = 100.0):
        factor = -2.0 * np.pi * 1000.0 / rate
        self.attack = np.float32(np.exp(factor / attack_ms))
        self.release = np.float32(np.exp(factor / release_ms))
        self.threshold = np.float32(10 ** (threshold_db / 20))
        self.exponent = np.float32(1 / ratio - 1)
        self.envelope = np.float32(0)

    def process(self, x: np.ndarray) -> np.ndarray:
        env = self.envelope
        y = np.empty_like(x)
        for i in range(len(x)):
            level = abs(x[i])
            cte = self.attack if level > env else self.release
            env = level + cte * (env - level)
            gain = np.float32(1) if env < self.threshold else np.float32((env / self.threshold) ** self.exponent)
            y[i] = gain * x[i]
        self.envelope = env
        return y


class Reverb:
    """juce::Reverb（Freeverb）的单声道处理。"""

    COMBS = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
    ALLPASSES = [556, 441, 341, 225]

    def __init__(self, rate: int, room_size: float, damping: float, wet: float, dry: float, width: float = 1.0):
        self.combs = [np.zeros(rate * n // 44100, dtype=np.float32) for n in self.COMBS]
        self.comb_index = [0] * len(self.COMBS)
        self.comb_last = [np.float32(0)] * len(self.COMBS)
        self.allpasses = [np.zeros(rate * n // 44100, dtype=np.float32) for n in self.ALLPASSES]
        self.allpass_index = [0] * len(self.ALLPASSES)
        self.damp = np.float32(damping * 0.4)
        self.feedback = np.float32(room_size * 0.28 + 0.7)
        self.dry = np.float32(dry * 2.0)
        self.wet = np.float32(0.5 * (wet * 3.0) * (1.0 + width))
        self.gain = np.float32(0.015)

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        half = np.float32(0.5)
        one = np.float32(1)
        for i in range(len(x)):
            value = x[i] * self.gain
            out = np.float32(0)
            for j, buffer in enumerate(self.combs):
                k = self.comb_index[j]
                delayed = buffer[k]
                last = delayed * (one - self.damp) + self.comb_last[j] * self.damp
                self.comb_last[j] = last
                buffer[k] = value + last * self.feedback
                self.comb_index[j] = (k + 1) % len(buffer)
                out += delayed
            for j, buffer in enumerate(self.allpasses):
                k = self.allpass_index[j]
                delayed = buffer[k]
                buffer[k] = out + delayed * half
                self.allpass_index[j] = (k + 1) % len(buffer)
                out = delayed - out
            y[i] = out * self.wet + x[i] * self.dry
        return y


class Enhancer:
    """一次合成用一个。每个片段调一次 process，状态和响度统计跨片段延续。"""

    BLOCK_SECONDS = 0.4
    HOP_SECONDS = 0.1

    def __init__(self, rate: int = 32000):
        self.rate = rate
        self.stages = [
            first_order_highpass(rate, 80.0),
            peak_filter(rate, 300.0, 1.0, 2.5),
            peak_filter(rate, 7000.0, 2.0, -3.0),
            Compressor(rate, -18.0, 3.5),
            Reverb(rate, room_size=0.1, damping=0.5, wet=0.03, dry=0.97),
        ]
        self.gain = np.float32(10 ** (2.0 / 20))
        self.weighting = k_weighting(rate)
        self.blocks = []  # 到目前为止所有 0.4 秒块的均方值（K 加权后）
        self.last_gain = None

    def measure(self, shaped: np.ndarray) -> None:
        weighted = shaped
        for stage in self.weighting:
            weighted = stage.process(weighted)
        block = int(self.BLOCK_SECONDS * self.rate)
        hop = int(self.HOP_SECONDS * self.rate)
        if len(weighted) < block:
            return
        # 与 pyloudnorm 相同的块数算法：最后一块可能超出数据末尾，仍按整块长度求均值
        count = int(np.round((len(weighted) / self.rate - self.BLOCK_SECONDS) / self.HOP_SECONDS)) + 1
        squared = weighted.astype(np.float64) ** 2
        for j in range(count):
            self.blocks.append(float(squared[j * hop: j * hop + block].sum()) / block)

    def integrated_loudness(self):
        """BS.1770 的门限算法：先去掉 -70 LUFS 以下的块，再去掉比平均低 10 dB 以上的块。"""
        z = np.array(self.blocks, dtype=np.float64)
        if len(z) == 0:
            return None
        with np.errstate(divide="ignore"):
            loudness = -0.691 + 10 * np.log10(z)
        absolute = z[loudness >= -70.0]
        if len(absolute) == 0:
            return None
        relative = -0.691 + 10 * np.log10(absolute.mean()) - 10.0
        gated = z[(loudness > relative) & (loudness > -70.0)]
        if len(gated) == 0:
            return None
        return float(-0.691 + 10 * np.log10(gated.mean()))

    def process(self, samples: np.ndarray) -> np.ndarray:
        shaped = samples.astype(np.float32)
        for stage in self.stages:
            shaped = stage.process(shaped)
        shaped = shaped * self.gain
        self.measure(shaped)
        loudness = self.integrated_loudness()
        if loudness is not None:
            self.last_gain = np.float32(10 ** ((TARGET_LUFS - loudness) / 20))
        if self.last_gain is None:
            # 开头就是不足 0.4 秒的片段，量不出响度：先按峰值 -6dB 处理
            peak = float(np.abs(shaped).max()) if len(shaped) else 0.0
            return shaped * np.float32(0.5 / peak) if peak > 0 else shaped
        out = shaped * self.last_gain
        # 响度统一后个别峰值可能超过 1，整段按比例压回去，避免削波
        peak = float(np.abs(out).max()) if len(out) else 0.0
        return out / np.float32(peak) if peak > 1 else out


def main() -> None:
    """和 pedalboard + pyloudnorm（网页界面的做法）对照。"""
    import pyloudnorm as pyln
    import soundfile as sf
    from pedalboard import Compressor as PbCompressor
    from pedalboard import Gain, HighpassFilter, PeakFilter, Pedalboard
    from pedalboard import Reverb as PbReverb

    for path in sys.argv[1:]:
        audio, rate = sf.read(path, dtype="float32")
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=80),
            PeakFilter(cutoff_frequency_hz=300, gain_db=2.5, q=1.0),
            PeakFilter(cutoff_frequency_hz=7000, gain_db=-3.0, q=2.0),
            PbCompressor(threshold_db=-18, ratio=3.5),
            PbReverb(room_size=0.1, dry_level=0.97, wet_level=0.03, damping=0.5),
            Gain(gain_db=2),
        ])
        effected = board(audio, rate)
        column = effected.reshape(-1, 1)
        loudness = pyln.Meter(rate).integrated_loudness(column)
        want = pyln.normalize.loudness(column, loudness, TARGET_LUFS).flatten()

        enhancer = Enhancer(rate)
        got = enhancer.process(audio)
        mine = enhancer.integrated_loudness()
        skip = int(0.02 * rate)  # JUCE 的混响参数在开头 10 毫秒有个渐变，这里不比
        error = got[skip:] - want[skip:]
        relative = float(np.sqrt((error ** 2).mean()) / np.sqrt((want[skip:] ** 2).mean()))
        print(f"{path}\n    原版响度 {loudness:.3f} LUFS，本实现 {mine:.3f} LUFS；"
              f"波形相对误差 {relative:.2e}，最大绝对差 {float(np.abs(error).max()):.2e}（峰值 {float(np.abs(want).max()):.3f}）")


if __name__ == "__main__":
    main()
