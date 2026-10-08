"""Kaldi 风格的 fbank 特征的参考实现。只依赖 numpy。

声纹模型的输入是 80 维 fbank。原版用的是
    torchaudio.compliance.kaldi.fbank(wav, num_mel_bins=80, sample_frequency=16000, dither=0)
这里把它按 torchaudio 的源码逐步写一遍：App\\Fbank.swift 照着它写，GitHub 上的测试用它生成标准答案。

    python fbank_ref.py 文件.wav    # 和 torchaudio 的结果对照
"""
import sys

import numpy as np

SAMPLE_RATE = 16000
WINDOW = 400        # 25 毫秒
SHIFT = 160         # 10 毫秒
PADDED = 512        # 补到 2 的幂做 FFT
MEL_BINS = 80
LOW_FREQ = 20.0
PREEMPHASIS = 0.97
EPSILON = np.finfo(np.float32).eps


def mel(frequency):
    return 1127.0 * np.log(1.0 + frequency / 700.0)


def mel_banks() -> np.ndarray:
    """三角滤波器组，[80, 257]。在 mel 刻度上等距，最后一列（奈奎斯特频率）补零。"""
    high_freq = 0.5 * SAMPLE_RATE
    bin_width = SAMPLE_RATE / PADDED
    mel_low = mel(LOW_FREQ)
    delta = (mel(high_freq) - mel_low) / (MEL_BINS + 1)
    index = np.arange(MEL_BINS, dtype=np.float32)[:, None]
    left = mel_low + index * delta
    center = mel_low + (index + 1.0) * delta
    right = mel_low + (index + 2.0) * delta
    points = mel(bin_width * np.arange(PADDED // 2, dtype=np.float32))[None, :].astype(np.float32)
    up = (points - left) / (center - left)
    down = (right - points) / (right - center)
    banks = np.maximum(0.0, np.minimum(up, down)).astype(np.float32)
    return np.pad(banks, ((0, 0), (0, 1)))


def povey_window() -> np.ndarray:
    n = np.arange(WINDOW, dtype=np.float64)
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * n / (WINDOW - 1))
    return (hann ** 0.85).astype(np.float32)


def kaldi_fbank(waveform: np.ndarray) -> np.ndarray:
    """waveform：16kHz 单声道 float32。返回 [帧数, 80] 的对数 mel 能量。"""
    waveform = np.asarray(waveform, dtype=np.float32)
    if len(waveform) < WINDOW:
        return np.zeros((0, MEL_BINS), dtype=np.float32)
    count = 1 + (len(waveform) - WINDOW) // SHIFT
    frames = np.stack([waveform[i * SHIFT: i * SHIFT + WINDOW] for i in range(count)])
    frames = frames - frames.mean(axis=1, keepdims=True)                     # 每帧去直流
    previous = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)       # 预加重，第一个采样用自己当前一个
    frames = frames - PREEMPHASIS * previous
    frames = frames * povey_window()[None, :]
    frames = np.pad(frames, ((0, 0), (0, PADDED - WINDOW)))
    power = (np.abs(np.fft.rfft(frames, axis=1)) ** 2).astype(np.float32)    # [帧数, 257]
    energies = power @ mel_banks().T
    return np.log(np.maximum(energies, EPSILON)).astype(np.float32)


def main() -> None:
    import torch
    import torchaudio

    for path in sys.argv[1:]:
        wav, sr = torchaudio.load(path)
        wav16k = torchaudio.functional.resample(wav.mean(0, keepdim=True), sr, 16000)
        want = torchaudio.compliance.kaldi.fbank(wav16k, num_mel_bins=80, sample_frequency=16000, dither=0).numpy()
        got = kaldi_fbank(wav16k[0].numpy())
        print(f"{path}\n    形状 {got.shape} / {want.shape}，最大绝对差 {float(np.abs(got - want).max()):.2e}，"
              f"数值范围 {want.min():.1f} ~ {want.max():.1f}")


if __name__ == "__main__":
    main()
