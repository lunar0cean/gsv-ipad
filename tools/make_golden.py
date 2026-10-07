"""生成 iPad 端文本处理要用的符号表和对照样本。

    python make_golden.py ja    App\\Frontend\\data.js（音素符号表）和 frontend\\test\\ja_golden.json
    python make_golden.py zh    frontend\\test\\zh_golden.json

日文样本记下原版 GSV-TTS-Lite 对每句给出的音素，以及 OpenJTalk 对每个片段给出的全上下文标签：
  - 电脑上：把标签喂给 frontend.js，结果应与原版音素完全一致，用来验证 JS 的移植
  - GitHub 上：把 jpreprocess 的标签喂给 frontend.js，看它和原版 OpenJTalk 差多少
中文样本记下原版的规范化文本、分词（带词性）和音素，zh.js 的结果应与之完全一致。

两种语言要分开两次运行：pyopenjtalk 和 jieba_fast 在同一个进程里先后初始化会崩溃。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_pipeline import MODELS_DIR, ROOT, load_gsv  # noqa: E402

TEST_DIR = os.path.join(ROOT, "frontend", "test")


def read_sentences(name: str) -> list:
    with open(os.path.join(TEST_DIR, name), "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_json(name: str, data) -> None:
    path = os.path.join(TEST_DIR, name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=0)
    print(f"{len(data)} 句 -> {path}（{os.path.getsize(path) / 1e3:.0f} KB）")


def japanese() -> None:
    # 必须在导入 pyopenjtalk 之前指定词典目录，否则它找不到词典会自己联网下载
    os.environ["OPEN_JTALK_DICT_DIR"] = os.path.join(MODELS_DIR, "g2p", "ja", "open_jtalk_dic_utf_8-1.11")
    import pyopenjtalk
    from gsv_tts.GPT_SoVITS.G2P import Symbols, text_to_phonemes

    data_path = os.path.join(ROOT, "App", "Frontend", "data.js")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("// 由 tools/make_golden.py 生成，不要手改\n")
        f.write("var GSV_DATA = " + json.dumps({"symbols": Symbols.symbols}, ensure_ascii=False) + ";\n")
    print(f"符号表 {len(Symbols.symbols)} 个 -> {data_path}")

    # 记下 OpenJTalk 对每个片段给出的标签
    segments = {}
    state = {"text": None}
    run_frontend, make_label = pyopenjtalk.run_frontend, pyopenjtalk.make_label

    def traced_run_frontend(text, *args, **kwargs):
        state["text"] = text
        return run_frontend(text, *args, **kwargs)

    def traced_make_label(features, *args, **kwargs):
        labels = make_label(features, *args, **kwargs)
        segments[state["text"]] = labels
        return labels

    pyopenjtalk.run_frontend, pyopenjtalk.make_label = traced_run_frontend, traced_make_label

    cases = []
    for text in read_sentences("ja_sentences.txt"):
        segments.clear()
        phones, _, norm_text = text_to_phonemes(text, "ja")
        cases.append({"text": text, "norm": norm_text, "phones": phones, "segments": dict(segments)})
    write_json("ja_golden.json", cases)


def chinese(extra: str = None) -> None:
    import jieba_fast.posseg as psg
    from gsv_tts.GPT_SoVITS.G2P import text_to_phonemes

    def case(text: str) -> dict:
        try:
            phones, _, norm_text = text_to_phonemes(text, "zh")
        except Exception as error:  # 原版遇到读不出来的字会直接报错，这类句子不进样本
            return {"text": text, "error": repr(error)}
        return {"text": text, "norm": norm_text, "phones": phones,
                "words": [[word, flag] for word, flag in psg.lcut(norm_text)]}

    if extra:
        with open(extra, "r", encoding="utf-8") as f:
            cases = [case(line.strip()) for line in f if line.strip()]
        out = os.path.splitext(extra)[0] + "_golden.json"
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(cases, f, ensure_ascii=False)
        print(f"{len(cases)} 句 -> {out}")
        return
    write_json("zh_golden.json", [c for c in (case(t) for t in read_sentences("zh_sentences.txt"))])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lang", choices=["ja", "zh"])
    ap.add_argument("--extra", help="zh：另外给一个每行一句的文本文件，结果写到同名的 _golden.json（本机压力测试用）")
    args = ap.parse_args()
    load_gsv()
    if args.lang == "ja":
        japanese()
    else:
        chinese(args.extra)
