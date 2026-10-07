"""把本机的中文 RoBERTa（D:\\k\\models\\chinese-roberta-wwm-ext-large）做成 iPad 用的 ONNX。

    python export_bert.py build      # 生成 work\\ipad\\bert\\roberta_fp16.onnx（约 600MB）和 App\\Frontend\\zh_bert_vocab.json
    python export_bert.py reference  # 用原版 PyTorch 模型算几句的特征，存到 work\\tmp\\bert_ref.npz
    python export_bert.py check      # 用 ONNX 算同样的句子，和上一步的结果比

reference 和 check 分开两个进程跑，各占 2GB 左右内存，跑之前确认电脑可用内存在 4GB 以上。
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BERT_DIR = r"D:\k\models\chinese-roberta-wwm-ext-large"
OUT = os.path.join(ROOT, "work", "ipad", "bert", "roberta_fp16.onnx")
VOCAB_OUT = os.path.join(ROOT, "App", "Frontend", "zh_bert_vocab.json")
REFERENCE = os.path.join(ROOT, "work", "tmp", "bert_ref.npz")
SENTENCES = [
    ".今天的天气真不错,我们一起去海边走走吧.",
    ".对不起,这件事是我没有考虑周全,下次一定会提前告诉你.",
    ".你好.",
    ".啊?真的吗!太好了…我简直不敢相信.",
]
CLS, SEP, UNK = 101, 102, 100


def load_vocab() -> dict:
    with open(os.path.join(BERT_DIR, "tokenizer.json"), "r", encoding="utf-8") as f:
        return json.load(f)["model"]["vocab"]


def char_ids(vocab: dict, text: str) -> list:
    """规范化后的中文只有汉字和几种标点，BERT 的分词结果就是一字一个，查不到的记作 [UNK]。"""
    return [CLS] + [vocab.get(ch, UNK) for ch in text] + [SEP]


def build() -> None:
    import torch

    from bert_onnx import build_bert_onnx

    vocab = load_vocab()
    single = {token: index for token, index in vocab.items()
              if len(token) == 1 and ("\u4e00" <= token <= "\u9fa5" or token in "!?…,.-")}
    os.makedirs(os.path.dirname(VOCAB_OUT), exist_ok=True)
    with open(VOCAB_OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"cls": CLS, "sep": SEP, "unk": UNK, "chars": single}, f, ensure_ascii=False, separators=(",", ":"))
    print(f"字表 {len(single)} 个 -> {VOCAB_OUT}")

    state = torch.load(os.path.join(BERT_DIR, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    build_bert_onnx(OUT, lambda key, shape: state["bert." + key].numpy())
    print(f"{os.path.getsize(OUT) / 1e6:.0f} MB -> {OUT}")


def reference() -> None:
    """与 gsv_tts 的 CNRoberta.forward 相同：hidden_states[-3]，去掉首尾特殊符号。在 CPU 上用单精度算。"""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
    model = AutoModelForMaskedLM.from_pretrained(BERT_DIR).float().eval()
    vocab = load_vocab()
    out = {}
    with torch.no_grad():
        for i, text in enumerate(SENTENCES):
            inputs = tokenizer(text, return_tensors="pt")
            ids = inputs["input_ids"][0].tolist()
            assert ids == char_ids(vocab, text), (text, ids)  # 分词确实是一字一个
            hidden = model(**inputs, output_hidden_states=True)["hidden_states"][-3]
            out[f"s{i}"] = hidden[0, 1:-1, :].numpy().astype(np.float32)
    os.makedirs(os.path.dirname(REFERENCE), exist_ok=True)
    np.savez(REFERENCE, **out)
    print(f"原版特征 {len(out)} 句 -> {REFERENCE}")


def check() -> None:
    import onnxruntime as ort

    vocab = load_vocab()
    ref = np.load(REFERENCE)
    session = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    worst = 0.0
    for i, text in enumerate(SENTENCES):
        ids = np.array([char_ids(vocab, text)], dtype=np.int64)
        got = session.run(None, {"input_ids": ids})[0]
        want = ref[f"s{i}"]
        diff = float(np.abs(got - want).max())
        cosine = float((got * want).sum() / (np.linalg.norm(got) * np.linalg.norm(want)))
        worst = max(worst, diff)
        print(f"{text}\n    形状 {got.shape}，最大绝对差 {diff:.2e}，余弦相似度 {cosine:.8f}，原版数值范围 ±{np.abs(want).max():.1f}")
    print("通过" if worst < 1e-2 else "差异过大")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "reference", "check"])
    {"build": build, "reference": reference, "check": check}[ap.parse_args().command]()
