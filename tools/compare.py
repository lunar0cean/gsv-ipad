"""对比原版 GSV-TTS-Lite（PyTorch，显卡）和转换后的 ONNX 模型（CPU）。

同一个预设、同一句话各合成一次，然后用两个客观指标打分：
  - 语音识别（D:\\k\\models\\qwen3_asr）转写，看内容有没有读对
  - 声纹相似度（ERes2NetV2），看音色和参考音频像不像
两份音频都保存到 work\\out\\compare\\，可以直接听。
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_pipeline import (GSV_ROOT, MODELS_DIR, PUNCTUATION, ROOT, Frontend, OnnxSynth, build_voice,  # noqa: E402
                          read_preset)

CASES = [
    (r"丹瑾\吃惊", "今天的天气真不错，我们一起去海边走走吧。", None),
    (r"丹瑾\难过", "对不起，这件事是我没有考虑周全，下次一定会提前告诉你。", None),
    (r"小町鸫\日常对话01", "今日はいい天気ですね。一緒に海まで散歩しませんか。", "ja"),
    (r"小町鸫\叙述01", "明天早上八点，我在图书馆门口等你。", "zh"),
    (r"神户小鸟\打招呼01", "おはようございます。昨日はよく眠れましたか。", "ja"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "work", "out", "compare"))
    ap.add_argument("--no-asr", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    frontend = Frontend(use_bert=True)
    synth = OnnxSynth()

    from gsv_tts import TTS
    os.chdir(GSV_ROOT)
    tts = TTS(gpt_cache=[(1, 512)], sovits_cache=[], use_bert=True, models_dir=MODELS_DIR, always_load_sv=True)

    rows = []
    for i, (preset, text, text_lang) in enumerate(CASES, 1):
        prompt_wav, prompt_text, spk_wav, lang = read_preset(os.path.join(GSV_ROOT, "presets", preset))
        text = text if text[-1] in PUNCTUATION else text + "."
        tag = f"{i:02d}_{preset.replace(os.sep, '_')}"

        # ONNX（CPU）
        voice = build_voice(frontend, prompt_wav, prompt_text, spk_wav, lang)
        seq, bert, _ = frontend(text, text_lang)
        audio, stat = synth(voice, seq, bert)
        peak = float(np.abs(audio).max())
        onnx_wav = os.path.join(args.out, f"{tag}_onnx.wav")
        sf.write(onnx_wav, audio / peak if peak > 1 else audio, 32000)

        # 原版（显卡）。参考文本按角色语种处理，目标文本按指定语种处理，和网页界面的做法一致
        tts.tts_config.force_lang = lang
        tts.cache_prompt_audio(prompt_wav, prompt_text)
        tts.tts_config.force_lang = text_lang
        clip = tts.infer(spk_audio_path=spk_wav, prompt_audio_path=prompt_wav, prompt_audio_text=prompt_text,
                         text=text)
        torch_wav = os.path.join(args.out, f"{tag}_torch.wav")
        sf.write(torch_wav, clip.audio_data, clip.samplerate)

        rows.append({
            "case": tag, "text": text, "onnx_wav": onnx_wav, "torch_wav": torch_wav, "spk_wav": spk_wav,
            "onnx_s": round(len(audio) / 32000, 2), "torch_s": round(clip.audio_len_s, 2),
            "onnx_rtf": round((stat["t2s_s"] + stat["vits_s"]) / (len(audio) / 32000), 2),
            "sim_onnx": round(tts.verify_speaker(spk_wav, onnx_wav), 3),
            "sim_torch": round(tts.verify_speaker(spk_wav, torch_wav), 3),
        })
        print(f"[{tag}] 合成完成", flush=True)

    if not args.no_asr:
        del tts
        torch.cuda.empty_cache()
        from qwen_asr import Qwen3ASRModel
        asr = Qwen3ASRModel.from_pretrained(os.path.join(MODELS_DIR, "qwen3_asr"), dtype=torch.bfloat16,
                                            device_map="cuda:0")
        for row in rows:
            row["asr_onnx"] = asr.transcribe(row["onnx_wav"])[0].text
            row["asr_torch"] = asr.transcribe(row["torch_wav"])[0].text

    with open(os.path.join(args.out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    for row in rows:
        print("=" * 60)
        print("预设    ", row["case"])
        print("原文    ", row["text"])
        if "asr_onnx" in row:
            print("ONNX 识别", row["asr_onnx"])
            print("原版识别", row["asr_torch"])
        print(f"时长     ONNX {row['onnx_s']}s / 原版 {row['torch_s']}s   ONNX 实时率 {row['onnx_rtf']}")
        print(f"音色相似 ONNX {row['sim_onnx']} / 原版 {row['sim_torch']}")


if __name__ == "__main__":
    main()
