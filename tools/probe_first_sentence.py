"""检查「漏读第一句」：同一句话在加和不加前导句号两种情况下各合成几次，用语音识别看结果。"""
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_pipeline import GSV_ROOT, MODELS_DIR, ROOT, Frontend, OnnxSynth, build_voice, read_preset  # noqa: E402

CASES = [
    (r"神户小鸟\打招呼01", "おはようございます。昨日はよく眠れましたか。", "ja"),
    (r"丹瑾\生气", "你好。我今天有点累，想早点回去休息。", "zh"),
    (r"小町鸫\道谢道歉01", "ありがとう。本当に助かりました。", "ja"),
]
TRIALS = 4

out_dir = os.path.join(ROOT, "work", "out", "probe")
os.makedirs(out_dir, exist_ok=True)
frontend = Frontend(use_bert=True)
synth = OnnxSynth()
jobs = []
for ci, (preset, text, text_lang) in enumerate(CASES):
    voice = build_voice(frontend, *read_preset(os.path.join(GSV_ROOT, "presets", preset)))
    for prefix in ("", "。"):
        seq, bert, _ = frontend(prefix + text, text_lang)
        for t in range(TRIALS):
            audio, _ = synth(voice, seq, bert)
            path = os.path.join(out_dir, f"c{ci}_{'dot' if prefix else 'raw'}_{t}.wav")
            sf.write(path, audio / max(1.0, float(np.abs(audio).max())), 32000)
            jobs.append((ci, prefix, text, path, len(audio) / 32000))

from qwen_asr import Qwen3ASRModel  # noqa: E402

asr = Qwen3ASRModel.from_pretrained(os.path.join(MODELS_DIR, "qwen3_asr"), dtype=torch.bfloat16, device_map="cuda:0")
last = None
for ci, prefix, text, path, dur in jobs:
    if (ci, prefix) != last:
        print(f"--- {text}   前导句号：{'有' if prefix else '无'}")
        last = (ci, prefix)
    print(f"  {dur:5.2f}s  {asr.transcribe(path)[0].text}")
