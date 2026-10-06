"""把 GSV-TTS-Lite 的本地底模（safetensors）转成 Genie-TTS 布局的 ONNX 模型。

输入：  D:\\k\\models\\s1v3、D:\\k\\models\\s2Gv2ProPlus（只读，不修改）
输出：  work\\onnx\\ 下的 5 个 .onnx 和对应的 .bin 权重（全精度）

ONNX 计算图用 genie-tts 包里自带的模板，这里只负责把权重按模板要求的名字和顺序写进去。
不 import genie_tts 本身，因为它在导入时会检查并提示下载 GenieData。
"""
import argparse
import importlib.util
import os
import re
import sys

import numpy as np
import onnx
import torch
from safetensors.torch import load_file

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

ENCODER_KEYS = [
    "encoder.ar_text_embedding.word_embeddings.weight",
    "encoder.bert_proj.weight",
    "encoder.bert_proj.bias",
    "encoder.ar_text_position.alpha",
    "vits.ssl_proj.weight",
    "vits.ssl_proj.bias",
    "vits.quantizer.vq.layers.0._codebook.embed",
]


def genie_data_dir() -> str:
    spec = importlib.util.find_spec("genie_tts")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("找不到 genie_tts 包，请先在这个 venv 里安装 genie-tts")
    return os.path.join(list(spec.submodule_search_locations)[0], "Data")


def read_keys(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def gpt_tensor(sd: dict, onnx_key: str) -> torch.Tensor:
    m = _LAYER_RE.match(onnx_key)
    if m:
        key = f"t2s_transformer.blocks.{m.group(1)}.{GPT_LAYER_KEY_MAP[m.group(2)]}"
    else:
        key = onnx_key
    return sd[key]


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


def sovits_tensor(sd: dict, onnx_key: str) -> torch.Tensor:
    key = onnx_key[len("vq_model."):] if onnx_key.startswith("vq_model.") else onnx_key
    return sd[key]


def write_bin(path: str, items: list) -> dict:
    """按顺序把张量写成全精度 .bin，返回 {名字: (偏移, 长度, 形状)}。"""
    index = {}
    offset = 0
    with open(path, "wb") as f:
        for name, tensor in items:
            data = tensor.to(torch.float32).cpu().numpy().tobytes()
            f.write(data)
            index[name] = (offset, len(data), tuple(tensor.shape))
            offset += len(data)
    return index


def cast_bool_outputs(model: onnx.ModelProto) -> None:
    """onnxruntime 的 Objective-C 接口没有 bool 张量类型，iPad 上读不了 bool 输出，这里统一转成 int64。"""
    for out in model.graph.output:
        if out.type.tensor_type.elem_type != onnx.TensorProto.BOOL:
            continue
        src = out.name
        new = "stop_flag" if src == "stop_condition_tensor" else src + "_i64"
        model.graph.node.append(
            onnx.helper.make_node("Cast", [src], [new], to=onnx.TensorProto.INT64, name=f"Cast_{new}"))
        out.name = new
        out.type.tensor_type.elem_type = onnx.TensorProto.INT64


def relink(template: str, out_path: str, bin_name: str, index: dict) -> None:
    model = onnx.load_model(template, load_external_data=False)
    cast_bool_outputs(model)
    linked = 0
    for init in model.graph.initializer:
        if init.name not in index:
            continue
        offset, length, shape = index[init.name]
        expected = int(np.prod(init.dims)) * 4 if len(init.dims) else 4
        if expected != length:
            sys.exit(f"形状对不上：{init.name} 模板要 {tuple(init.dims)}，本地权重是 {shape}")
        if tuple(init.dims) != shape:
            print(f"  注意：{init.name} 元素数相同但形状不同，模板 {tuple(init.dims)}，本地 {shape}")
        init.ClearField("raw_data")
        init.data_location = onnx.TensorProto.EXTERNAL
        del init.external_data[:]
        for k, v in (("location", bin_name), ("offset", str(offset)), ("length", str(length))):
            entry = init.external_data.add()
            entry.key = k
            entry.value = v
        linked += 1
    missing = [k for k in index if k not in {i.name for i in model.graph.initializer}]
    if missing:
        sys.exit(f"{os.path.basename(template)} 里找不到这些权重：{missing[:5]}")
    onnx.save(model, out_path)
    print(f"  {os.path.basename(out_path)}：接入 {linked} 个权重 -> {bin_name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt", default=r"D:\k\models\s1v3\model.safetensors")
    ap.add_argument("--sovits", default=r"D:\k\models\s2Gv2ProPlus\model.safetensors")
    ap.add_argument("--out", default=os.path.join(ROOT, "work", "onnx"))
    args = ap.parse_args()

    data = genie_data_dir()
    v2, v2pp = os.path.join(data, "v2"), os.path.join(data, "v2ProPlus")
    os.makedirs(args.out, exist_ok=True)

    gpt_sd = load_file(args.gpt)
    sovits_sd = load_file(args.sovits)
    print(f"GPT 权重 {len(gpt_sd)} 个，SoVITS 权重 {len(sovits_sd)} 个")
    print(f"还原 SoVITS 解码器的权重归一化：{restore_weight_norm(sovits_sd)} 层")

    print("文本到语义（T2S）解码器")
    keys = read_keys(os.path.join(v2, "Keys", "t2s_onnx_keys.txt"))
    index = write_bin(os.path.join(args.out, "t2s_shared_fp32.bin"), [(k, gpt_tensor(gpt_sd, k)) for k in keys])
    for name in ("t2s_first_stage_decoder_fp32.onnx", "t2s_stage_decoder_fp32.onnx"):
        relink(os.path.join(v2, "Models", name), os.path.join(args.out, name), "t2s_shared_fp32.bin", index)

    print("T2S 编码器")
    items = []
    for k in ENCODER_KEYS:
        if k.startswith("encoder."):
            items.append((k, gpt_sd[k[len("encoder."):]]))
        else:
            items.append((k, sovits_sd[k[len("vits."):]]))
    index = write_bin(os.path.join(args.out, "t2s_encoder_fp32.bin"), items)
    relink(os.path.join(v2, "Models", "t2s_encoder_fp32.onnx"), os.path.join(args.out, "t2s_encoder_fp32.onnx"),
           "t2s_encoder_fp32.bin", index)

    print("声码器（VITS）")
    keys = read_keys(os.path.join(v2pp, "Keys", "vits_weights.txt"))
    index = write_bin(os.path.join(args.out, "vits_fp32.bin"), [(k, sovits_tensor(sovits_sd, k)) for k in keys])
    relink(os.path.join(v2pp, "Models", "vits_fp32.onnx"), os.path.join(args.out, "vits_fp32.onnx"),
           "vits_fp32.bin", index)

    print("音色编码器（只在电脑上用，用来预先算角色包）")
    keys = read_keys(os.path.join(v2pp, "Keys", "prompt_encoder_weights.txt"))
    index = write_bin(os.path.join(args.out, "prompt_encoder_fp32.bin"),
                      [(k, sovits_tensor(sovits_sd, k)) for k in keys])
    relink(os.path.join(v2pp, "Models", "prompt_encoder_fp32.onnx"),
           os.path.join(args.out, "prompt_encoder_fp32.onnx"), "prompt_encoder_fp32.bin", index)

    print("完成，输出目录：", args.out)
    for name in sorted(os.listdir(args.out)):
        print(f"  {os.path.getsize(os.path.join(args.out, name)) / 1e6:8.1f} MB  {name}")


if __name__ == "__main__":
    main()
