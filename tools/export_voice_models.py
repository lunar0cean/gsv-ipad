"""导出「在 iPad 上加角色」要用的两个模型，并生成 GitHub 上测试用的空模板。

处理一段参考音频需要三个模型：
    hubert.onnx          16kHz 波形 -> 语义特征（chinese-hubert-base）
    sv.onnx              80 维 fbank 特征 -> 声纹向量（ERes2NetV2）
    prompt_encoder_fp32  32kHz 波形 + 声纹向量 -> 音色向量（convert_models.py 已经生成）
前两个在这里从 D:\\k\\models 的 PyTorch 原版导出。这两个模型不大，用 torch.onnx.export 内存够用。

    python export_voice_models.py export     # 生成 work\\ipad\\voice\\ 下的模型，和 tools\\templates\\ 下的空模板
    python export_voice_models.py templates  # 只重新生成空模板
    python export_voice_models.py check      # 用 ONNX 重算一个角色包，和 export_pack.py 用 PyTorch 算出来的比

空模板是去掉权重、只留计算图的模型文件。真模型不进仓库，GitHub 上的测试往模板里填随机权重来跑流程。
"""
import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GSV_ROOT = r"D:\k"
MODELS_DIR = os.path.join(GSV_ROOT, "models")
OUT_DIR = os.path.join(ROOT, "work", "ipad", "voice")
TEMPLATE_DIR = os.path.join(ROOT, "tools", "templates")


def export() -> None:
    import onnx
    import torch
    from transformers import HubertModel

    from onnx_pack import float_initializers_to_fp16

    sys.path.insert(0, GSV_ROOT)
    from gsv_tts.GPT_SoVITS.SV.ERes2NetV2 import ERes2NetV2

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TEMPLATE_DIR, exist_ok=True)

    class Hubert(torch.nn.Module):
        """与 gsv_tts.TTS._get_prompt 相同：波形直接送进模型，取最后一层，转成 [1, 768, 帧数]。"""

        def __init__(self):
            super().__init__()
            self.model = HubertModel.from_pretrained(os.path.join(MODELS_DIR, "chinese-hubert-base"),
                                                     local_files_only=True).float().eval()

        def forward(self, waveform):
            return self.model(waveform)["last_hidden_state"].transpose(1, 2)

    class SpeakerEncoder(torch.nn.Module):
        """与 gsv_tts 的 ERes2Net.compute_embedding3 相同，fbank 特征在模型外面算。"""

        def __init__(self):
            super().__init__()
            self.model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
            state = torch.load(os.path.join(MODELS_DIR, "sv", "pretrained_eres2netv2w24s4ep4.ckpt"),
                               map_location="cpu", weights_only=False)
            self.model.load_state_dict(state)
            self.model.eval()

        def forward(self, fbank):
            return self.model.forward3(fbank)

    jobs = [
        ("hubert", Hubert(), torch.zeros(1, 32000), "waveform", {1: "samples"}, "ssl_content", {2: "frames"}, True),
        ("sv", SpeakerEncoder(), torch.zeros(1, 300, 80), "fbank", {1: "frames"}, "sv_emb", {}, True),
    ]
    for name, module, dummy, input_name, input_axes, output_name, output_axes, half in jobs:
        raw = os.path.join(ROOT, "work", "tmp", f"{name}_export.onnx")
        with torch.no_grad():
            torch.onnx.export(module, (dummy,), raw, input_names=[input_name], output_names=[output_name],
                              dynamic_axes={input_name: input_axes, output_name: output_axes},
                              opset_version=17, dynamo=False)
        del module
        model = onnx.load(raw)
        if half:
            # 原始权重本来就是半精度，按半精度存不损失；图里先转回单精度再算
            float_initializers_to_fp16(model)
        out = os.path.join(OUT_DIR, f"{name}.onnx")
        onnx.save(model, out)
        os.remove(raw)
        print(f"{name}：模型 {os.path.getsize(out) / 1e6:.0f} MB -> {out}")

    for file in ("prompt_encoder_fp32.onnx", "prompt_encoder_fp32.bin"):
        shutil.copy2(os.path.join(ROOT, "work", "onnx", file), os.path.join(OUT_DIR, file))
    print("音色编码器已复制到", OUT_DIR)
    templates()


def templates() -> None:
    """从导出的模型生成空模板：权重全部去掉，只留计算图。"""
    import onnx

    from onnx_pack import strip_weights

    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    for name in ("hubert", "sv"):
        model = onnx.load(os.path.join(OUT_DIR, f"{name}.onnx"))
        strip_weights(model)
        template = os.path.join(TEMPLATE_DIR, f"{name}.onnx")
        onnx.save(model, template)
        print(f"{name}：模板 {os.path.getsize(template) / 1e3:.0f} KB -> {template}")


def check() -> None:
    """用 ONNX 模型和 numpy 版 fbank 重算第一个角色的特征，和 PyTorch 算的角色包比。"""
    import json

    import onnxruntime as ort
    import torch
    import torchaudio

    from fbank_ref import kaldi_fbank
    from gsvpack import read_pack
    from ref_pipeline import tail_offset

    with open(os.path.join(ROOT, "work", "presets.json"), "r", encoding="utf-8") as f:
        preset = json.load(f)[0]
    _, _, want = read_pack(os.path.join(ROOT, "work", "ipad", "voices", preset["file"] + ".gsvpack"))

    # 预处理与 ref_pipeline.build_voice 相同
    wav, sr = torchaudio.load(preset["wav"])
    wav16k = torchaudio.functional.resample(wav, sr, 16000).mean(dim=0)
    off = tail_offset(wav16k)
    if off > 0:
        wav16k = wav16k[:-off]
    wav16k = torch.cat([wav16k, torch.zeros(int(16000 * 0.3))])
    audio32k = torchaudio.functional.resample(wav.mean(0, keepdim=True) if wav.shape[0] > 1 else wav, sr, 32000)
    peak = audio32k.abs().max()
    if peak > 1:
        audio32k = audio32k / min(2, peak)
    audio16k = torchaudio.functional.resample(audio32k, 32000, 16000)

    def run(name, feeds):
        session = ort.InferenceSession(os.path.join(OUT_DIR, name), providers=["CPUExecutionProvider"])
        return session.run(None, feeds)

    def report(label, got, expected):
        diff = float(np.abs(got - expected).max())
        cosine = float((got * expected).sum() / (np.linalg.norm(got) * np.linalg.norm(expected)))
        print(f"{label}：形状 {got.shape}，最大绝对差 {diff:.2e}，余弦相似度 {cosine:.7f}，数值范围 ±{np.abs(expected).max():.1f}")

    ssl = run("hubert.onnx", {"waveform": wav16k.unsqueeze(0).numpy()})[0]
    report("语义特征", ssl, want["ssl_content"])
    fbank = kaldi_fbank(audio16k[0].numpy())
    sv_emb = run("sv.onnx", {"fbank": fbank[None]})[0]
    ge, ge_advanced = run("prompt_encoder_fp32.onnx", {"ref_audio": audio32k.numpy().astype(np.float32), "sv_emb": sv_emb})
    report("音色向量 ge", ge, want["ge"])
    report("音色向量 ge_advanced", ge_advanced, want["ge_advanced"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["export", "templates", "check"])
    {"export": export, "templates": templates, "check": check}[ap.parse_args().command]()
