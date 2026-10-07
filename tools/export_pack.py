"""导出 iPad 端用的角色包，并把要拷到 iPad 的文件集中到 work\\ipad\\。

角色包（.gsvpack，格式见 gsvpack.py）里是参考音频算出来的 ref_seq、ref_bert、ssl_content、ge、ge_advanced。

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
import subprocess
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsvpack import read_pack, write_pack  # noqa: E402
from ref_pipeline import ONNX_DIR, ROOT, OnnxSynth  # noqa: E402

IPAD_DIR = os.path.join(ROOT, "work", "ipad")
PRESETS_FILE = os.path.join(ROOT, "work", "presets.json")
MODEL_FILES = [
    "t2s_encoder_fp32.onnx", "t2s_encoder_fp32.bin",
    "t2s_first_stage_decoder_fp32.onnx", "t2s_stage_decoder_fp32.onnx", "t2s_shared_fp32.bin",
    "vits_fp32.onnx", "vits_fp32.bin",
]
BERT_MODEL = os.path.join(IPAD_DIR, "bert", "roberta_fp16.onnx")
VERIFY_TEXTS = {
    "ja": ["今日はいい天気ですね。一緒に海まで散歩しませんか。"],
    # 中文多放几句、带上不同的语气，方便听语调模型有没有起作用
    "zh": [
        "今天的天气真不错，我们一起去海边走走吧。",
        "你怎么这么晚才回来？我等了你好久，饭菜都凉了。",
        "说实话，这件事我一开始并不同意，可是后来想了想，觉得他说的也有道理。",
    ],
}

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
    """按 iPad 的做法合成，结果就是 iPad 上该有的声音。

    日文的目标文本特征是全零。中文每句出两份：装了语调模型时的（按字算特征再展开），和没装时的（全零）。
    两份之间除了语调模型，还有每次采样本身的随机差别，听的时候多比几句。
    """
    from ref_pipeline import Frontend

    frontend = Frontend(use_bert=False)
    synth = OnnxSynth(onnx_dir=os.path.join(IPAD_DIR, "models"))
    out_dir = os.path.join(ROOT, "work", "out", "verify")
    os.makedirs(out_dir, exist_ok=True)

    bert_session, vocab = None, None
    if lang == "zh" and os.path.exists(BERT_MODEL):
        import onnxruntime as ort

        from export_bert import char_ids, load_vocab

        bert_session = ort.InferenceSession(BERT_MODEL, providers=["CPUExecutionProvider"])
        vocab = load_vocab()

    def save(name: str, voice: dict, seq, bert, label: str) -> None:
        audio, stat = synth(voice, seq, bert)
        dur = len(audio) / 32000
        sf.write(os.path.join(out_dir, name), audio / max(1.0, float(np.abs(audio).max())), 32000)
        print(f"    {label}：语义 {stat['tokens']}，音频 {dur:.2f}s，实时率 {(stat['t2s_s'] + stat['vits_s']) / dur:.2f}  {name}",
              flush=True)

    voices_dir = os.path.join(IPAD_DIR, "voices")
    for file in sorted(os.listdir(voices_dir)):
        _, meta, voice = read_pack(os.path.join(voices_dir, file))
        if meta["lang"] != lang:
            continue
        stem = file.replace(".gsvpack", "")
        for number, text in enumerate(VERIFY_TEXTS[lang], 1):
            seq, word2ph, _, norm = frontend.detail("。" + text, lang)
            zeros = np.zeros((seq.shape[1], 1024), dtype=np.float32)
            print(f"{meta['name']} | {text}（{seq.shape[1]} 个音素）", flush=True)
            if bert_session is None:
                save(f"{stem}_{number}.wav", voice, seq, zeros, "合成")
                continue
            chars = bert_session.run(None, {"input_ids": np.array([char_ids(vocab, norm)], dtype=np.int64)})[0]
            features = np.repeat(chars, word2ph, axis=0).astype(np.float32)
            assert features.shape[0] == seq.shape[1], (features.shape, seq.shape)
            save(f"{stem}_{number}_有语调模型.wav", voice, seq, features, "有语调模型")
            save(f"{stem}_{number}_无语调模型.wav", voice, seq, zeros, "无语调模型")
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
