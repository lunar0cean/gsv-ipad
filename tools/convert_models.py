"""把 GSV-TTS-Lite 的本地底模（safetensors）转成 Genie-TTS 布局的 ONNX 模型。

输入：  D:\\k\\models\\s1v3、D:\\k\\models\\s2Gv2ProPlus（只读，不修改）
输出：  work\\onnx\\ 下的 5 个 .onnx 和对应的 .bin 权重（全精度）
"""
import argparse
import os
import re
import sys

from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onnx_pack import build_models  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# GSV-TTS-Lite 加载官方 GPT 权重时做的改名（见 gsv_tts/Loader.py），这里反过来用
GPT_LAYER_KEY_MAP = {
    "self_attn.in_proj_weight": "qkv.weight",
    "self_attn.in_proj_bias": "qkv.bias",
    "self_attn.out_proj.weight": "out_proj.weight",
    "self_attn.out_proj.bias": "out_proj.bias",
    "linear1.weight": "mlp.0.weight",
    "linear1.bias": "mlp.0.bias",
    "linear2.weight": "mlp.2.weight",
    "linear2.bias": "mlp.2.bias",
    "norm1.weight": "norm1.weight",
    "norm1.bias": "norm1.bias",
    "norm2.weight": "norm2.weight",
    "norm2.bias": "norm2.bias",
}
_LAYER_RE = re.compile(r"^transformer_encoder\.layers\.(\d+)\.(.+)$")


def gpt_key(onnx_key: str) -> str:
    m = _LAYER_RE.match(onnx_key)
    return f"t2s_transformer.blocks.{m.group(1)}.{GPT_LAYER_KEY_MAP[m.group(2)]}" if m else onnx_key


def restore_weight_norm(sd: dict) -> int:
    """GSV-TTS-Lite 对 dec 做了 remove_weight_norm，把 weight_g、weight_v 合并成了 weight。
    模板图里仍按 g * v / ||v|| 计算，所以取 v = weight、g = ||weight||，结果与合并后的权重相同。"""
    restored = 0
    for key in [k for k in sd if k.startswith("dec.") and k.endswith(".weight")]:
        stem = key[: -len(".weight")]
        if stem + ".weight_g" in sd:
            continue
        w = sd[key].float()
        if w.dim() != 3:
            continue
        sd[stem + ".weight_v"] = w
        sd[stem + ".weight_g"] = w.norm(dim=(1, 2), keepdim=True)
        restored += 1
    return restored


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt", default=r"D:\k\models\s1v3\model.safetensors")
    ap.add_argument("--sovits", default=r"D:\k\models\s2Gv2ProPlus\model.safetensors")
    ap.add_argument("--out", default=os.path.join(ROOT, "work", "onnx"))
    args = ap.parse_args()

    gpt_sd = load_file(args.gpt)
    sovits_sd = load_file(args.sovits)
    print(f"GPT 权重 {len(gpt_sd)} 个，SoVITS 权重 {len(sovits_sd)} 个")
    print(f"还原 SoVITS 解码器的权重归一化：{restore_weight_norm(sovits_sd)} 层")

    def lookup(onnx_name: str, key: str, dims):
        if onnx_name.startswith("t2s_encoder"):
            # 编码器的权重一部分来自 GPT，一部分来自 SoVITS
            tensor = gpt_sd[key[len("encoder."):]] if key.startswith("encoder.") else sovits_sd[key[len("vits."):]]
        elif onnx_name.startswith("t2s_"):
            tensor = gpt_sd[gpt_key(key)]
        else:
            tensor = sovits_sd[key[len("vq_model."):] if key.startswith("vq_model.") else key]
        return tensor.float().cpu().numpy()

    build_models(args.out, lookup)
    print("完成，输出目录：", args.out)
    for name in sorted(os.listdir(args.out)):
        print(f"  {os.path.getsize(os.path.join(args.out, name)) / 1e6:8.1f} MB  {name}")


if __name__ == "__main__":
    main()
