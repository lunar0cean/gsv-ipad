"""生成模拟器冒烟测试用的假模型和假角色包。给 GitHub 上的自动测试用，也可以在电脑上跑。

真模型和角色音频不进仓库，所以测试用随机数填出结构完全相同的模型：合成出来的是噪声，
但「载入模型 -> 编码 -> 解码循环 -> 声码器 -> 写出音频」这条流程和真模型走的是同一份代码。

    python ci_fixtures.py <输出目录>          # 生成 models\\、voice_models\\、voice.gsvpack 和各项标准答案
    python ci_fixtures.py <输出目录> --run    # 另外用 onnxruntime 跑一遍，确认随机模型本身能跑通

只依赖 numpy、onnx 和 genie-tts 包里的模板（--run 另需 onnxruntime）。
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsvpack import write_pack  # noqa: E402
from onnx_pack import build_models, fill_random  # noqa: E402

REF_PHONES = 24
SSL_FRAMES = 150


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(20261007)

    def lookup(onnx_name: str, key: str, dims):
        if dims is None:
            sys.exit(f"{onnx_name} 的模板里没有 {key}")
        # 归一化层的缩放取 1 附近，其余取小随机数，数值不会发散
        if key.endswith(("norm1.weight", "norm2.weight", ".gamma", "alpha")):
            return (1 + rng.normal(0, 0.02, dims)).astype(np.float32)
        return rng.normal(0, 0.05, dims).astype(np.float32)

    build_models(os.path.join(args.out, "models"), lookup)
    # 「在 iPad 上加角色」用的三个模型放在另一个文件夹，与 App 里一样分开找
    voice_dir = os.path.join(args.out, "voice_models")
    os.makedirs(voice_dir, exist_ok=True)
    for name in ("prompt_encoder_fp32.onnx", "prompt_encoder_fp32.bin"):
        os.replace(os.path.join(args.out, "models", name), os.path.join(voice_dir, name))
    templates = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    for name in ("hubert.onnx", "sv.onnx"):
        fill_random(os.path.join(templates, name), os.path.join(voice_dir, name), rng)

    voice = {
        "ref_seq": rng.integers(1, 300, size=(1, REF_PHONES), dtype=np.int64),
        "ref_bert": np.zeros((REF_PHONES, 1024), dtype=np.float32),
        "ssl_content": rng.normal(0, 1, (1, 768, SSL_FRAMES)).astype(np.float32),
        "ge": rng.normal(0, 1, (1, 1024, 1)).astype(np.float32),
        "ge_advanced": rng.normal(0, 1, (1, 512, 1)).astype(np.float32),
    }
    write_pack(os.path.join(args.out, "voice.gsvpack"), "voice", {"name": "测试", "lang": "ja"}, voice)

    # 中文语调模型：结构相同，只留 1 层，权重随机
    from bert_onnx import build_bert_onnx

    def bert_lookup(key: str, shape):
        if key.endswith("LayerNorm.weight"):
            return (1 + rng.normal(0, 0.02, shape)).astype(np.float16)
        return rng.normal(0, 0.05, shape).astype(np.float16)

    build_bert_onnx(os.path.join(args.out, "roberta_fp16.onnx"), bert_lookup, layers=1)

    # 音频增强：造一段像说话的信号（带谐波的元音加上高频的擦音），连续三个片段，
    # 用参考实现算出标准答案。中间那个片段不足 0.4 秒，走「量不出响度就沿用上一次增益」的分支
    from enhance_ref import Enhancer
    from fbank_ref import kaldi_fbank

    def speech_like(seconds: float, rate: int = 32000) -> np.ndarray:
        t = np.arange(int(seconds * rate)) / rate
        envelope = np.clip(np.sin(2 * np.pi * 3.1 * t), 0, None) ** 0.7
        voiced = sum(np.sin(2 * np.pi * 180 * h * t + h) / h for h in range(1, 12) if 180 * h < rate / 2)
        hiss = rng.normal(0, 1, len(t)) * np.clip(np.sin(2 * np.pi * 1.7 * t + 1), 0, None) ** 4
        return ((0.25 * voiced * envelope + 0.05 * hiss) * 0.6).astype(np.float32)

    enhancer = Enhancer(32000)
    for index, seconds in enumerate([1.2, 0.2, 0.9], 1):
        segment = speech_like(seconds)
        segment.tofile(os.path.join(args.out, f"enhance_in_{index}.f32"))
        enhancer.process(segment).astype(np.float32).tofile(os.path.join(args.out, f"enhance_out_{index}.f32"))
    print(f"音频增强的标准答案：3 个片段，累计响度 {enhancer.integrated_loudness():.2f} LUFS")

    # 声纹模型的 fbank 特征：16kHz 的一段，加一点底噪，免得整帧都是数字静音
    wave16k = speech_like(1.5, 16000) + rng.normal(0, 1e-3, 24000).astype(np.float32)
    wave16k.tofile(os.path.join(args.out, "fbank_in.f32"))
    kaldi_fbank(wave16k).tofile(os.path.join(args.out, "fbank_out.f32"))

    # 参考音频：44.1kHz 双声道 16 位 wav，测读文件、混成单声道、重采样
    import wave
    left = speech_like(3.0, 44100)
    stereo = np.stack([left, 0.5 * np.roll(left, 300)], axis=1)
    with wave.open(os.path.join(args.out, "ref_stereo.wav"), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(44100)
        f.writeframes((np.clip(stereo, -1, 1) * 32767).astype("<i2").tobytes())
    print(f"参考音频：{len(left)} 个采样，fbank 标准答案 {len(kaldi_fbank(wave16k))} 帧")
    total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(args.out) for f in fs)
    print(f"测试数据共 {total / 1e6:.0f} MB，在 {args.out}")

    if args.run:
        import onnxruntime as ort

        def load(name):
            return ort.InferenceSession(os.path.join(args.out, "models", name), providers=["CPUExecutionProvider"])

        encoder, first, stage, vits = (load(n) for n in (
            "t2s_encoder_fp32.onnx", "t2s_first_stage_decoder_fp32.onnx", "t2s_stage_decoder_fp32.onnx",
            "vits_fp32.onnx"))
        text_seq = rng.integers(1, 300, size=(1, 30), dtype=np.int64)
        x, prompts = encoder.run(None, {
            "ref_seq": voice["ref_seq"], "text_seq": text_seq, "ref_bert": voice["ref_bert"],
            "text_bert": np.zeros((30, 1024), dtype=np.float32), "ssl_content": voice["ssl_content"]})
        y, y_emb, *kv = first.run(None, {"x": x, "prompts": prompts})
        names = [i.name for i in stage.get_inputs()]
        steps = 0
        for steps in range(1, 31):
            y, y_emb, stop, *kv = stage.run(None, dict(zip(names, [y, y_emb, *kv])))
            if stop:
                break
        tokens = y[0, prompts.shape[1]:]
        tokens = tokens[tokens < 1024]
        audio = vits.run(None, {"text_seq": text_seq, "pred_semantic": tokens[None, None, :],
                                "ge": voice["ge"], "ge_advanced": voice["ge_advanced"]})[0]
        print(f"随机模型跑通：解码 {steps} 步，语义 {len(tokens)} 个，音频 {len(audio)} 个采样，"
              f"全部是有限数值：{bool(np.isfinite(audio).all())}")

        bert = ort.InferenceSession(os.path.join(args.out, "roberta_fp16.onnx"), providers=["CPUExecutionProvider"])
        features = bert.run(None, {"input_ids": np.array([[101, 872, 1962, 119, 102]], dtype=np.int64)})[0]
        print(f"随机语调模型跑通：输出形状 {features.shape}，全部是有限数值：{bool(np.isfinite(features).all())}")

        def voice_model(name):
            return ort.InferenceSession(os.path.join(voice_dir, name), providers=["CPUExecutionProvider"])

        ssl = voice_model("hubert.onnx").run(None, {"waveform": wave16k[None]})[0]
        sv_emb = voice_model("sv.onnx").run(None, {"fbank": kaldi_fbank(wave16k)[None]})[0]
        ge, ge_advanced = voice_model("prompt_encoder_fp32.onnx").run(None, {
            "ref_audio": speech_like(1.5)[None], "sv_emb": sv_emb})
        print(f"随机建角色模型跑通：语义特征 {ssl.shape}，声纹 {sv_emb.shape}，音色 {ge.shape} {ge_advanced.shape}，"
              f"全部是有限数值：{all(np.isfinite(a).all() for a in (ssl, sv_emb, ge, ge_advanced))}")


if __name__ == "__main__":
    main()
