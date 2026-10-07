"""生成 iPad 端文本处理要用的数据和对照样本。

    App\\Frontend\\data.js           音素符号表（frontend.js 运行时要用）
    frontend\\test\\ja_golden.json   日文对照样本：原版 GSV-TTS-Lite 对每句给出的音素，
                                   以及 OpenJTalk 对每个片段给出的全上下文标签

对照样本有两个用途：
  - 电脑上：把标签喂给 frontend.js，结果应与原版音素完全一致，用来验证 JS 的移植
  - GitHub 上：把 jpreprocess 的标签喂给 frontend.js，看它和原版 OpenJTalk 差多少
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_pipeline import MODELS_DIR, ROOT, load_gsv  # noqa: E402


def main() -> None:
    load_gsv()
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

    with open(os.path.join(ROOT, "frontend", "test", "ja_sentences.txt"), "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    cases = []
    for text in sentences:
        segments.clear()
        phones, _, norm_text = text_to_phonemes(text, "ja")
        cases.append({"text": text, "norm": norm_text, "phones": phones, "segments": dict(segments)})

    out = os.path.join(ROOT, "frontend", "test", "ja_golden.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cases, f, ensure_ascii=False, indent=0)
    print(f"日文对照样本 {len(cases)} 句 -> {out}（{os.path.getsize(out) / 1e3:.0f} KB）")


if __name__ == "__main__":
    main()
