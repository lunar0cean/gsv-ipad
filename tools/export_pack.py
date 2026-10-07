"""导出 iPad 端用的角色包，并把要拷到 iPad 的文件集中到 work\\ipad\\。

角色包（.gsvpack）是一个很简单的容器，Swift 端不依赖第三方库就能读：
    4 字节  "GSVP"
    4 字节  小端 uint32，头部长度 N
    N 字节  UTF-8 JSON：{"kind", "meta", "tensors": [{"name", "dtype", "shape", "offset", "length"}]}
    之后    各张量的原始字节（小端），offset 从这里算起
里面是参考音频算出来的 ref_seq、ref_bert、ssl_content、ge、ge_advanced。

要导出哪些角色写在 work\\presets.json 里（这个文件只在本机，不进仓库），每项：
    {"file": "1_名字", "name": "界面上显示的名字", "lang": "ja 或 zh", "wav": "参考音频", "text": "参考音频的文字"}
音色和语气用同一条参考音频。文件名前面的序号决定 App 里的显示顺序。

用法：
    python export_pack.py bundle    # 生成角色包并拷贝模型到 work\\ipad\\
    python export_pack.py verify    # 只用 work\\ipad\\ 里的文件、按 iPad 的做法各合成一句，存到 work\\out\\verify\\
"""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_pipeline import ONNX_DIR, ROOT, OnnxSynth  # noqa: E402

IPAD_DIR = os.path.join(ROOT, "work", "ipad")
PRESETS_FILE = os.path.join(ROOT, "work", "presets.json")
MODEL_FILES = [
    "t2s_encoder_fp32.onnx", "t2s_encoder_fp32.bin",
    "t2s_first_stage_decoder_fp32.onnx", "t2s_stage_decoder_fp32.onnx", "t2s_shared_fp32.bin",
    "vits_fp32.onnx", "vits_fp32.bin",
]
DTYPES = {"i64": np.int64, "f32": np.float32}
VERIFY_TEXTS = {
    "ja": "今日はいい天気ですね。一緒に海まで散歩しませんか。",
    "zh": "今天的天气真不错，我们一起去海边走走吧。",
}


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


def run_isolated(args: list, attempts: int = 3) -> None:
    """在单独的进程里跑一步，跑完内存全部还给系统。电脑内存紧张时特征提取会因为分配不到内存而崩溃
    （报「not enough memory」或访问冲突），所以失败时自动重试；-X faulthandler 会把崩溃位置打出来。"""
    command = [sys.executable, "-X", "faulthandler", os.path.abspath(__file__), *args]
    for attempt in range(1, attempts + 1):
        code = subprocess.run(command).returncode
        if code == 0:
            return
        print(f"第 {attempt} 次运行 {' '.join(args)} 失败，退出码 {code}", flush=True)
    raise SystemExit(f"{' '.join(args)} 连续 {attempts} 次失败")


def load_presets() -> list:
    with open(PRESETS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def export_voice(index: int) -> None:
    from ref_pipeline import Frontend, build_voice

    preset = load_presets()[index]
    # 中文 RoBERTa 载入后占 1.3GB 内存，只有中文参考文本才用得到
    frontend = Frontend(use_bert=preset["lang"] == "zh")
    voice = build_voice(frontend, preset["wav"], preset["text"], preset["wav"], preset["lang"])
    path = os.path.join(IPAD_DIR, "voices", preset["file"] + ".gsvpack")
    write_pack(path, "voice", {"name": preset["name"], "lang": preset["lang"]}, voice)
    seconds = voice["ssl_content"].shape[2] / 50
    print(f"角色包 {os.path.getsize(path) / 1e6:5.2f} MB  参考音频约 {seconds:.1f}s  {preset['name']}", flush=True)


def bundle() -> None:
    shutil.rmtree(os.path.join(IPAD_DIR, "voices"), ignore_errors=True)
    shutil.rmtree(os.path.join(IPAD_DIR, "tests"), ignore_errors=True)

    for index in range(len(load_presets())):
        run_isolated(["voice", str(index)])

    os.makedirs(os.path.join(IPAD_DIR, "models"), exist_ok=True)
    for name in MODEL_FILES:
        shutil.copy2(os.path.join(ONNX_DIR, name), os.path.join(IPAD_DIR, "models", name))
    total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(IPAD_DIR) for f in fs)
    print(f"要拷到 iPad 的文件共 {total / 1e6:.0f} MB，在 {IPAD_DIR}")


def verify(lang: str) -> None:
    """iPad 上目标文本的 BERT 特征是全零，这里也用全零，合成结果就是 iPad 上该有的声音。"""
    from ref_pipeline import Frontend

    frontend = Frontend(use_bert=False)
    synth = OnnxSynth(onnx_dir=os.path.join(IPAD_DIR, "models"))
    out_dir = os.path.join(ROOT, "work", "out", "verify")
    os.makedirs(out_dir, exist_ok=True)
    voices_dir = os.path.join(IPAD_DIR, "voices")
    for file in sorted(os.listdir(voices_dir)):
        _, meta, voice = read_pack(os.path.join(voices_dir, file))
        if meta["lang"] != lang:
            continue
        text = VERIFY_TEXTS[meta["lang"]]
        seq, _, _ = frontend("。" + text, meta["lang"])
        bert = np.zeros((seq.shape[1], 1024), dtype=np.float32)
        audio, stat = synth(voice, seq, bert)
        dur = len(audio) / 32000
        sf.write(os.path.join(out_dir, file.replace(".gsvpack", ".wav")), audio / max(1.0, float(np.abs(audio).max())),
                 32000)
        print(f"{meta['name']} | {text}\n    音素 {seq.shape[1]}，语义 {stat['tokens']}，音频 {dur:.2f}s，"
              f"实时率 {(stat['t2s_s'] + stat['vits_s']) / dur:.2f}")
    print("已保存到", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["bundle", "verify", "voice"])
    ap.add_argument("arg", nargs="?", help="voice：角色序号；verify：ja 或 zh，不填则两种都跑")
    args = ap.parse_args()
    if args.command == "bundle":
        bundle()
    elif args.command == "voice":
        export_voice(int(args.arg))
    elif args.arg:
        verify(args.arg)
    else:
        for lang in VERIFY_TEXTS:
            run_isolated(["verify", lang])
