"""导出 iPad 端中文文本处理要用的词典数据（App\\Frontend\\zh_*）。

数据全部来自本机 Python 环境里的开源包，原样转换格式，不改内容：
    zh_dict.txt     jieba 的词典（词 词频 词性），原文件
    zh_hmm.json     jieba 的两套 HMM 参数：finalseg（分词）和 posseg（分词 + 词性）
    zh_pinyin.json  pypinyin 的单字和词语读音，已按 GSV 用的风格转成「声母|韵母+声调」
    zh_misc.json    拼音 -> 音素的对照表（opencpop-strict.txt）、Python isnumeric() 为真的汉字

来源和许可证见 App\\Frontend\\NOTICE.md。
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_pipeline import MODELS_DIR, ROOT  # noqa: E402

OUT_DIR = os.path.join(ROOT, "App", "Frontend")
HAN_FIRST, HAN_LAST = 0x4E00, 0x9FA5  # GSV 的中文处理只保留这个范围内的汉字


def in_range(text: str) -> bool:
    return all(HAN_FIRST <= ord(ch) <= HAN_LAST for ch in text)


def dump(name: str, data) -> None:
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"{os.path.getsize(path) / 1e6:6.2f} MB  {name}")


def state_key(state) -> str:
    return f"{state[0]}|{state[1]}"


def export_jieba() -> None:
    import jieba_fast
    from jieba_fast import finalseg, posseg

    src = os.path.join(os.path.dirname(jieba_fast.__file__), "dict.txt")
    dst = os.path.join(OUT_DIR, "zh_dict.txt")
    shutil.copyfile(src, dst)
    print(f"{os.path.getsize(dst) / 1e6:6.2f} MB  zh_dict.txt")

    dump("zh_hmm.json", {
        "final": {
            "start": finalseg.start_P,
            "trans": finalseg.trans_P,
            "emit": finalseg.emit_P,
        },
        "pos": {
            "start": {state_key(k): v for k, v in posseg.start_P.items()},
            "trans": {state_key(k): {state_key(k2): v2 for k2, v2 in v.items()} for k, v in posseg.trans_P.items()},
            "emit": {state_key(k): v for k, v in posseg.emit_P.items()},
            "char_state": {ch: [state_key(s) for s in states] for ch, states in posseg.char_state_tab_P.items()},
        },
    })


def export_pinyin() -> None:
    from pypinyin import Style, lazy_pinyin
    from pypinyin.constants import PHRASES_DICT

    def reading(text: str):
        initials = lazy_pinyin(text, neutral_tone_with_five=True, style=Style.INITIALS)
        finals = lazy_pinyin(text, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
        return initials, finals

    chars = {}
    for code in range(HAN_FIRST, HAN_LAST + 1):
        ch = chr(code)
        initials, finals = reading(ch)
        assert len(initials) == len(finals) == 1, ch
        if initials[0] == ch and finals[0] == ch:
            continue  # 没有读音的字，脚本里按「原样返回」处理
        chars[ch] = f"{initials[0]}|{finals[0]}"

    phrases, prefix_only, skipped = {}, set(), 0
    for phrase in PHRASES_DICT:
        if in_range(phrase):
            initials, finals = reading(phrase)
            if len(initials) != len(phrase) or len(finals) != len(phrase):
                skipped += 1
                continue
            phrases[phrase] = " ".join(f"{i}|{f}" for i, f in zip(initials, finals))
        else:
            # 含范围外字符的词语不可能整体匹配，但它在范围内的那段前缀会影响 pypinyin 的最大正向匹配
            prefix = ""
            for ch in phrase:
                if not in_range(ch):
                    break
                prefix += ch
            if prefix:
                prefix_only.add(prefix)
    print(f"单字 {len(chars)}，词语 {len(phrases)}，只作前缀 {len(prefix_only)}，跳过 {skipped}")
    dump("zh_pinyin.json", {"chars": chars, "phrases": phrases, "prefix_only": sorted(prefix_only)})


def export_misc() -> None:
    opencpop = {}
    with open(os.path.join(MODELS_DIR, "g2p", "zh", "opencpop-strict.txt"), "r", encoding="utf-8") as f:
        for line in f.readlines():
            opencpop[line.split("\t")[0]] = line.strip().split("\t")[1]
    numeric = "".join(chr(c) for c in range(HAN_FIRST, HAN_LAST + 1) if chr(c).isnumeric())
    dump("zh_misc.json", {"opencpop": opencpop, "numeric": numeric})


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    export_jieba()
    export_pinyin()
    export_misc()
