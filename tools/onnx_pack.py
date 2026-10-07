"""把权重写进 Genie-TTS 的 ONNX 模板。只依赖 numpy 和 onnx，电脑上的模型转换和 GitHub 上的测试都用它。

计算图用 genie-tts 包里自带的模板，这里只负责把权重按模板要求的名字和顺序写成 .bin，并让模板指向它。
不 import genie_tts 本身，因为它在导入时会检查并提示下载 GenieData。
"""
import importlib.util
import os
import sys

import numpy as np
import onnx

ENCODER_KEYS = [
    "encoder.ar_text_embedding.word_embeddings.weight",
    "encoder.bert_proj.weight",
    "encoder.bert_proj.bias",
    "encoder.ar_text_position.alpha",
    "vits.ssl_proj.weight",
    "vits.ssl_proj.bias",
    "vits.quantizer.vq.layers.0._codebook.embed",
]

# (模板, 权重名清单, 输出的 .onnx, 输出的 .bin)。清单为 None 表示用上面的 ENCODER_KEYS
MODELS = [
    ("v2/Models/t2s_first_stage_decoder_fp32.onnx", "v2/Keys/t2s_onnx_keys.txt",
     "t2s_first_stage_decoder_fp32.onnx", "t2s_shared_fp32.bin"),
    ("v2/Models/t2s_stage_decoder_fp32.onnx", "v2/Keys/t2s_onnx_keys.txt",
     "t2s_stage_decoder_fp32.onnx", "t2s_shared_fp32.bin"),
    ("v2/Models/t2s_encoder_fp32.onnx", None, "t2s_encoder_fp32.onnx", "t2s_encoder_fp32.bin"),
    ("v2ProPlus/Models/vits_fp32.onnx", "v2ProPlus/Keys/vits_weights.txt", "vits_fp32.onnx", "vits_fp32.bin"),
    ("v2ProPlus/Models/prompt_encoder_fp32.onnx", "v2ProPlus/Keys/prompt_encoder_weights.txt",
     "prompt_encoder_fp32.onnx", "prompt_encoder_fp32.bin"),
]


def genie_data_dir() -> str:
    spec = importlib.util.find_spec("genie_tts")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("找不到 genie_tts 包，请先安装 genie-tts（可以不带依赖）")
    return os.path.join(list(spec.submodule_search_locations)[0], "Data")


def read_keys(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def model_keys(data_dir: str, key_file) -> list:
    return ENCODER_KEYS if key_file is None else read_keys(os.path.join(data_dir, key_file))


def template_dims(template: str) -> dict:
    model = onnx.load_model(template, load_external_data=False)
    return {init.name: tuple(init.dims) for init in model.graph.initializer}


def write_bin(path: str, items: list) -> dict:
    """按顺序把数组写成全精度 .bin，返回 {名字: (偏移, 长度, 形状)}。"""
    index = {}
    offset = 0
    with open(path, "wb") as f:
        for name, array in items:
            data = np.ascontiguousarray(array, dtype=np.float32).tobytes()
            f.write(data)
            index[name] = (offset, len(data), tuple(array.shape))
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
            sys.exit(f"形状对不上：{init.name} 模板要 {tuple(init.dims)}，给的权重是 {shape}")
        if tuple(init.dims) != shape:
            print(f"  注意：{init.name} 元素数相同但形状不同，模板 {tuple(init.dims)}，权重 {shape}")
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


def build_models(out_dir: str, lookup) -> None:
    """lookup(模板文件名, 权重名, 模板里的形状) -> numpy 数组。"""
    data_dir = genie_data_dir()
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for template, key_file, onnx_name, bin_name in MODELS:
        template_path = os.path.join(data_dir, template)
        if bin_name not in written:
            dims = template_dims(template_path)
            keys = model_keys(data_dir, key_file)
            written[bin_name] = write_bin(os.path.join(out_dir, bin_name),
                                          [(k, lookup(onnx_name, k, dims.get(k))) for k in keys])
        relink(template_path, os.path.join(out_dir, onnx_name), bin_name, written[bin_name])
