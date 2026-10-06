"""电脑上的参考流程：按 iPad 端将来的步骤，用转换后的 ONNX 模型合成一句话。

分两部分：
  1. 角色包（只在电脑上算）：参考音频 -> 语义特征、音色向量、参考文本的音素和 BERT 特征
  2. 合成（将来移植到 iPad）：目标文本的音素和 BERT -> T2S 编码器 -> 解码循环 -> 声码器

文本处理、HuBERT、声纹、RoBERTa 直接复用 D:\\k 里 GSV-TTS-Lite 的代码和模型，全部跑在 CPU 上，
不占用显卡，也不修改 D:\\k。
"""
import argparse
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GSV_ROOT = r"D:\k"
MODELS_DIR = os.path.join(GSV_ROOT, "models")
ONNX_DIR = os.path.join(ROOT, "work", "onnx")
LANG_CODE = {"中文": "zh", "日文": "ja", "英文": "en"}
PUNCTUATION = {".", "。", "?", "？", "!", "！", ",", "，", ":", "：", ";", "；", "、"}
MAX_STEPS = 1000
EOS = 1024


def load_gsv():
    sys.path.insert(0, GSV_ROOT)
    from gsv_tts.config import Config, global_config
    from gsv_tts.GPT_SoVITS.Featurizer import CNHubert, CNRoberta
    from gsv_tts.GPT_SoVITS.SV import ERes2Net
    from gsv_tts.TextProcessor import get_phones_and_bert

    global_config.models_dir = MODELS_DIR
    cfg = Config()
    cfg.device = torch.device("cpu")
    cfg.is_half = False
    cfg.dtype = torch.float32
    return cfg, CNHubert, CNRoberta, ERes2Net, get_phones_and_bert


class Frontend:
    """文本 -> 音素编号和 BERT 特征。iPad 端要用 Swift 重写这一层。"""

    def __init__(self, use_bert: bool = True):
        self.cfg, self.CNHubert, CNRoberta, self.ERes2Net, self._get = load_gsv()
        if use_bert:
            self.cfg.cnroberta = CNRoberta(os.path.join(MODELS_DIR, "chinese-roberta-wwm-ext-large"), self.cfg)

    def __call__(self, text: str, lang: str = None):
        self.cfg.force_lang = lang
        with torch.inference_mode():
            phones, _, bert, norm_text = self._get(text, self.cfg)
        seq = np.array([phones], dtype=np.int64)
        return seq, bert.float().cpu().numpy(), norm_text


def tail_offset(audio: torch.Tensor, threshold=0.01, frame_length=512, hop_length=256, search_len=6400) -> int:
    """与 gsv_tts.TTS._find_threshold_offsets 相同：找出参考音频结尾的静音长度。"""
    tail = audio[-search_len:]
    if tail.numel() < frame_length:
        return 0
    rms = torch.sqrt(torch.mean(tail.unfold(0, frame_length, hop_length) ** 2, dim=1))
    idx = torch.nonzero(rms > threshold * audio.abs().max())
    if idx.numel() == 0:
        return 0
    return tail.numel() - (idx[-1].item() * hop_length + frame_length)


def build_voice(frontend: Frontend, prompt_wav: str, prompt_text: str, spk_wav: str, lang: str = None) -> dict:
    """算出一个预设的角色包。prompt_wav 决定语气，spk_wav 决定音色。"""
    cfg = frontend.cfg
    with torch.inference_mode():
        # 语气：参考音频的 HuBERT 特征（与 gsv_tts.TTS._get_prompt 的预处理一致）
        wav, sr = torchaudio.load(prompt_wav)
        wav16k = torchaudio.functional.resample(wav, sr, 16000).mean(dim=0)
        off = tail_offset(wav16k)
        if off > 0:
            wav16k = wav16k[:-off]
        wav16k = torch.cat([wav16k, torch.zeros(int(16000 * 0.3))])
        hubert = frontend.CNHubert(os.path.join(MODELS_DIR, "chinese-hubert-base"), cfg)
        ssl = hubert.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        del hubert

        # 音色：32k 音频 + 声纹向量 -> 音色编码器（与 gsv_tts.TTS._get_spepc 的预处理一致）
        audio, sr0 = torchaudio.load(spk_wav)
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio32k = torchaudio.functional.resample(audio, sr0, 32000)
        peak = audio32k.abs().max()
        if peak > 1:
            audio32k = audio32k / min(2, peak)
        audio16k = torchaudio.functional.resample(audio32k, 32000, 16000)
        sv = frontend.ERes2Net(os.path.join(MODELS_DIR, "sv", "pretrained_eres2netv2w24s4ep4.ckpt"), cfg)
        sv_emb = sv.compute_embedding3(audio16k)
        del sv

    prompt_encoder = ort.InferenceSession(os.path.join(ONNX_DIR, "prompt_encoder_fp32.onnx"),
                                          providers=["CPUExecutionProvider"])
    ge, ge_advanced = prompt_encoder.run(None, {
        "ref_audio": audio32k.numpy().astype(np.float32),
        "sv_emb": sv_emb.float().numpy(),
    })
    ref_seq, ref_bert, _ = frontend(prompt_text, lang)
    return {
        "ref_seq": ref_seq,
        "ref_bert": ref_bert.astype(np.float32),
        "ssl_content": ssl.float().numpy(),
        "ge": ge.astype(np.float32),
        "ge_advanced": ge_advanced.astype(np.float32),
    }


class OnnxSynth:
    """文本特征 + 角色包 -> 波形。这一层就是要移植到 iPad 的全部推理。"""

    def __init__(self, onnx_dir: str = ONNX_DIR, threads: int = 0):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            opts.intra_op_num_threads = threads

        def load(name):
            return ort.InferenceSession(os.path.join(onnx_dir, name), sess_options=opts,
                                        providers=["CPUExecutionProvider"])

        self.encoder = load("t2s_encoder_fp32.onnx")
        self.first = load("t2s_first_stage_decoder_fp32.onnx")
        self.stage = load("t2s_stage_decoder_fp32.onnx")
        self.vits = load("vits_fp32.onnx")
        self.stage_inputs = [i.name for i in self.stage.get_inputs()]

    def semantic(self, voice: dict, text_seq: np.ndarray, text_bert: np.ndarray) -> np.ndarray:
        x, prompts = self.encoder.run(None, {
            "ref_seq": voice["ref_seq"],
            "text_seq": text_seq,
            "ref_bert": voice["ref_bert"],
            "text_bert": text_bert,
            "ssl_content": voice["ssl_content"],
        })
        y, y_emb, *kv = self.first.run(None, {"x": x, "prompts": prompts})
        prompt_len = prompts.shape[1]
        stopped = False
        for _ in range(MAX_STEPS):
            y, y_emb, stop, *kv = self.stage.run(None, dict(zip(self.stage_inputs, [y, y_emb, *kv])))
            if stop:
                stopped = True
                break
        tokens = y[0, prompt_len:]
        if stopped:
            tokens = tokens[:-1]  # 触发结束的那一个不算
        tokens = tokens[tokens < EOS]
        return tokens[None, None, :]

    def __call__(self, voice: dict, text_seq: np.ndarray, text_bert: np.ndarray):
        t0 = time.perf_counter()
        sem = self.semantic(voice, text_seq, text_bert)
        t1 = time.perf_counter()
        audio = self.vits.run(None, {
            "text_seq": text_seq,
            "pred_semantic": sem,
            "ge": voice["ge"],
            "ge_advanced": voice["ge_advanced"],
        })[0]
        t2 = time.perf_counter()
        return audio.astype(np.float32), {"tokens": int(sem.shape[-1]), "t2s_s": t1 - t0, "vits_s": t2 - t1}


def read_preset(preset_dir: str):
    """D:\\k\\presets\\角色\\情绪\\ 里的 ref.wav、ref.txt、spk\\ref.wav，以及上一级的 lang.txt。"""
    with open(os.path.join(preset_dir, "ref.txt"), "r", encoding="utf-8") as f:
        text = f.read().strip()
    lang = None
    lang_file = os.path.join(os.path.dirname(preset_dir), "lang.txt")
    if os.path.exists(lang_file):
        with open(lang_file, "r", encoding="utf-8") as f:
            lang = LANG_CODE.get(f.read().strip())
    prompt_wav = os.path.join(preset_dir, "ref.wav")
    spk_wav = os.path.join(preset_dir, "spk", "ref.wav")
    if not os.path.exists(spk_wav):
        spk_wav = prompt_wav
    return prompt_wav, text, spk_wav, lang


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, help=r"预设目录，例如 D:\k\presets\丹瑾\吃惊")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-lang", default=None, help="目标文本的语种 zh/ja/en，不填则自动判断")
    ap.add_argument("--no-bert", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    prompt_wav, prompt_text, spk_wav, lang = read_preset(args.preset)
    frontend = Frontend(use_bert=not args.no_bert)
    t0 = time.perf_counter()
    voice = build_voice(frontend, prompt_wav, prompt_text, spk_wav, lang)
    print(f"角色包：参考音素 {voice['ref_seq'].shape[1]} 个，语义特征 {voice['ssl_content'].shape[2]} 帧，"
          f"用时 {time.perf_counter() - t0:.1f}s")

    text = args.text if args.text[-1] in PUNCTUATION else args.text + "."
    text_seq, text_bert, norm_text = frontend(text, args.text_lang)
    print(f"目标文本：{norm_text}（{text_seq.shape[1]} 个音素）")

    synth = OnnxSynth(threads=args.threads)
    audio, stat = synth(voice, text_seq, text_bert)
    peak = float(np.abs(audio).max())
    if peak > 1:
        audio = audio / peak
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sf.write(args.out, audio, 32000)
    dur = len(audio) / 32000
    print(f"合成 {dur:.2f}s 音频：语义 {stat['tokens']} 个，T2S {stat['t2s_s']:.2f}s，声码器 {stat['vits_s']:.2f}s，"
          f"实时率 {(stat['t2s_s'] + stat['vits_s']) / dur:.2f}")
    print("已保存：", args.out)


if __name__ == "__main__":
    main()
