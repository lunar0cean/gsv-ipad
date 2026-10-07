"""用 D:\\k\\models\\qwen3_asr 转写一个文件夹里的 wav，检查合成内容有没有读对。要用显卡。

    python asr_check.py <文件夹>
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_pipeline import GSV_ROOT, MODELS_DIR  # noqa: E402

sys.path.insert(0, GSV_ROOT)
from qwen_asr import Qwen3ASRModel  # noqa: E402

folder = sys.argv[1]
asr = Qwen3ASRModel.from_pretrained(os.path.join(MODELS_DIR, "qwen3_asr"), dtype=torch.bfloat16, device_map="cuda:0")
for name in sorted(os.listdir(folder)):
    if name.lower().endswith(".wav"):
        print(f"{name}\t{asr.transcribe(os.path.join(folder, name))[0].text}", flush=True)
