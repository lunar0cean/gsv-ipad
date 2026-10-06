"""导出 iPad 端用的数据包，并把要拷到 iPad 的文件集中到 work\\ipad\\。

数据包（.gsvpack）是一个很简单的容器，Swift 端不依赖第三方库就能读：
    4 字节  "GSVP"
    4 字节  小端 uint32，头部长度 N
    N 字节  UTF-8 JSON：{"kind", "meta", "tensors": [{"name", "dtype", "shape", "offset", "length"}]}
    之后    各张量的原始字节（小端），offset 从这里算起

两种包：
    voice  角色包：ref_seq、ref_bert、ssl_content、ge、ge_advanced
    text   文本包：text_seq、text_bert（第四阶段在 iPad 上实现文本处理后就不再需要）

用法：
    python export_pack.py bundle    # 生成基准测试用的一组角色包、文本包，并拷贝模型
    python export_pack.py verify    # 只用 work\\ipad\\ 里的文件合成一遍，确认这些文件是自足的
"""
import argparse
import json
import os
import shutil
import struct
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_pipeline import GSV_ROOT, ONNX_DIR, ROOT, OnnxSynth  # noqa: E402

IPAD_DIR = os.path.join(ROOT, "work", "ipad")
MODEL_FILES = [
    "t2s_encoder_fp32.onnx", "t2s_encoder_fp32.bin",
    "t2s_first_stage_decoder_fp32.onnx", "t2s_stage_decoder_fp32.onnx", "t2s_shared_fp32.bin",
    "vits_fp32.onnx", "vits_fp32.bin",
]
DTYPES = {"i64": np.int64, "f32": np.float32}

BENCH_VOICES = [r"丹瑾\吃惊", r"小町鸫\日常对话01", r"神户小鸟\打招呼01"]
BENCH_TEXTS = [
    ("zh", "今天的天气真不错，我们一起去海边走走吧。"),
    ("zh", "对不起，这件事是我没有考虑周全，下次一定会提前告诉你，不会再让你白白等这么久了。"),
    ("ja", "今日はいい天気ですね。一緒に海まで散歩しませんか。"),
    ("ja", "おはようございます。昨日はよく眠れましたか。"),
]


def write_pack(path: str, kind: str, meta: dict, tensors: dict) -> None:
    entries, blobs, offset = [], [], 0
    for name, arr in tensors.items():
        dtype = "i64" if arr.dtype == np.int64 else "f32"
        data = np.ascontiguousarray(arr.astype(DTYPES[dtype])).tobytes()
        entries.append({"name": name, "dtype": dtype, "shape": list(arr.shape), "offset": offset, "length": len(data)})
        blobs.append(data)
        offset += len(data)
    header = json.dumps({"kind": kind, "meta": {k: str(v) for k, v in meta.items()}, "tensors": entries},
                        ensure_ascii=False).encode("utf-8")
    header += b" " * ((-(8 + len(header))) % 8)  # 让数据区按 8 字节对齐
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"GSVP")
        f.write(struct.pack("<I", len(header)))
        f.write(header)
        for blob in blobs:
            f.write(blob)


def read_pack(path: str):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:4] != b"GSVP":
        raise ValueError(f"不是数据包：{path}")
    n = struct.unpack("<I", raw[4:8])[0]
    header = json.loads(raw[8:8 + n].decode("utf-8"))
    base = 8 + n
    tensors = {}
    for e in header["tensors"]:
        buf = raw[base + e["offset"]: base + e["offset"] + e["length"]]
        tensors[e["name"]] = np.frombuffer(buf, dtype=DTYPES[e["dtype"]]).reshape(e["shape"]).copy()
    return header["kind"], header["meta"], tensors


def bundle() -> None:
    from ref_pipeline import PUNCTUATION, Frontend, build_voice, read_preset

    frontend = Frontend(use_bert=True)
    for preset in BENCH_VOICES:
        prompt_wav, prompt_text, spk_wav, lang = read_preset(os.path.join(GSV_ROOT, "presets", preset))
        character, emotion = preset.split("\\")
        voice = build_voice(frontend, prompt_wav, prompt_text, spk_wav, lang)
        path = os.path.join(IPAD_DIR, "voices", f"{character}_{emotion}.gsvpack")
        write_pack(path, "voice", {"name": f"{character} · {emotion}", "lang": lang or "", "prompt_text": prompt_text},
                   voice)
        print(f"角色包 {os.path.getsize(path) / 1e6:5.2f} MB  {path}")

    for i, (lang, text) in enumerate(BENCH_TEXTS, 1):
        text = text if text[-1] in PUNCTUATION else text + "。"
        # 前导句号用来减少「漏读很短的第一句」，见 README
        seq, bert, norm = frontend("。" + text, lang)
        path = os.path.join(IPAD_DIR, "tests", f"{i:02d}_{lang}.gsvpack")
        write_pack(path, "text", {"name": text, "lang": lang, "norm_text": norm}, {"text_seq": seq, "text_bert": bert})
        print(f"文本包 {os.path.getsize(path) / 1e6:5.2f} MB  {path}")

    os.makedirs(os.path.join(IPAD_DIR, "models"), exist_ok=True)
    for name in MODEL_FILES:
        shutil.copy2(os.path.join(ONNX_DIR, name), os.path.join(IPAD_DIR, "models", name))
    total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(IPAD_DIR) for f in fs)
    print(f"要拷到 iPad 的文件共 {total / 1e6:.0f} MB，在 {IPAD_DIR}")


def verify() -> None:
    synth = OnnxSynth(onnx_dir=os.path.join(IPAD_DIR, "models"))
    voices = sorted(os.listdir(os.path.join(IPAD_DIR, "voices")))
    tests = sorted(os.listdir(os.path.join(IPAD_DIR, "tests")))
    out_dir = os.path.join(ROOT, "work", "out", "verify")
    os.makedirs(out_dir, exist_ok=True)
    for vi, test in enumerate(tests):
        _, vmeta, voice = read_pack(os.path.join(IPAD_DIR, "voices", voices[vi % len(voices)]))
        _, tmeta, text = read_pack(os.path.join(IPAD_DIR, "tests", test))
        audio, stat = synth(voice, text["text_seq"], text["text_bert"])
        dur = len(audio) / 32000
        sf.write(os.path.join(out_dir, test.replace(".gsvpack", ".wav")), audio / max(1.0, float(np.abs(audio).max())),
                 32000)
        print(f"{vmeta['name']} | {tmeta['name']}\n    音素 {text['text_seq'].shape[1]}，语义 {stat['tokens']}，"
              f"音频 {dur:.2f}s，实时率 {(stat['t2s_s'] + stat['vits_s']) / dur:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["bundle", "verify"])
    {"bundle": bundle, "verify": verify}[ap.parse_args().command]()
