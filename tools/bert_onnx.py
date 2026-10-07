"""按 BERT 的结构直接拼出 ONNX 计算图，把权重写进去。只依赖 numpy 和 onnx。

不用 torch.onnx.export：那样要把整个模型在内存里展开好几份，这台电脑吃不消。BERT 的结构很规整，
直接拼图既省内存，又能让同一份代码在 GitHub 上用随机权重生成测试模型。

对应 GSV-TTS-Lite 的做法（gsv_tts/GPT_SoVITS/Featurizer/cnroberta.py）：取 hidden_states[-3]，
也就是 24 层里前 22 层的输出，再去掉首尾的 [CLS]、[SEP]。

    输入  input_ids      int64   [1, L]      含 [CLS] 和 [SEP] 的字符编号
    输出  char_features  float32 [L-2, 1024]  每个字一行

权重按半精度存进文件（原始权重本来就是半精度，没有损失），图里先转成单精度再计算。
ONNX Runtime 载入时会把这一步预先算好，所以运行时内存按单精度算。
"""
import numpy as np
import onnx
from onnx import TensorProto, helper

HIDDEN = 1024
HEADS = 16
INTERMEDIATE = 4096
VOCAB = 21128
MAX_POSITION = 512
LAYERS = 22  # hidden_states[-3]
EPSILON = 1e-12


def weight_shapes(layers: int = LAYERS) -> dict:
    """权重名（Hugging Face 的 BertModel 命名，不带 bert. 前缀）-> 形状。"""
    shapes = {
        "embeddings.word_embeddings.weight": (VOCAB, HIDDEN),
        "embeddings.position_embeddings.weight": (MAX_POSITION, HIDDEN),
        "embeddings.token_type_embeddings.weight": (2, HIDDEN),
        "embeddings.LayerNorm.weight": (HIDDEN,),
        "embeddings.LayerNorm.bias": (HIDDEN,),
    }
    for i in range(layers):
        p = f"encoder.layer.{i}."
        for name in ("attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense"):
            shapes[p + name + ".weight"] = (HIDDEN, HIDDEN)
            shapes[p + name + ".bias"] = (HIDDEN,)
        shapes[p + "intermediate.dense.weight"] = (INTERMEDIATE, HIDDEN)
        shapes[p + "intermediate.dense.bias"] = (INTERMEDIATE,)
        shapes[p + "output.dense.weight"] = (HIDDEN, INTERMEDIATE)
        shapes[p + "output.dense.bias"] = (HIDDEN,)
        for name in ("attention.output.LayerNorm", "output.LayerNorm"):
            shapes[p + name + ".weight"] = (HIDDEN,)
            shapes[p + name + ".bias"] = (HIDDEN,)
    return shapes


def build_bert_onnx(path: str, lookup, layers: int = LAYERS) -> None:
    """lookup(权重名, 形状) -> numpy 数组。"""
    shapes = weight_shapes(layers)
    nodes, initializers = [], []

    def const(name: str, array: np.ndarray) -> str:
        initializers.append(helper.make_tensor(name, TensorProto.INT64 if array.dtype == np.int64 else TensorProto.FLOAT,
                                               array.shape, array.tobytes(), raw=True))
        return name

    def weight(key: str, transpose: bool = False) -> str:
        """半精度存、单精度用。线性层的权重转置后存，图里直接 MatMul。"""
        array = np.asarray(lookup(key, shapes[key]))
        if tuple(array.shape) != shapes[key]:
            raise ValueError(f"{key} 的形状是 {tuple(array.shape)}，应为 {shapes[key]}")
        if transpose:
            array = array.T
        array = np.ascontiguousarray(array, dtype=np.float16)
        initializers.append(helper.make_tensor(key + ".h", TensorProto.FLOAT16, array.shape, array.tobytes(), raw=True))
        nodes.append(helper.make_node("Cast", [key + ".h"], [key], to=TensorProto.FLOAT, name="cast/" + key))
        return key

    def node(op: str, inputs: list, name: str, **attrs) -> str:
        nodes.append(helper.make_node(op, inputs, [name], name=name, **attrs))
        return name

    def linear(x: str, key: str, name: str) -> str:
        return node("Add", [node("MatMul", [x, weight(key + ".weight", transpose=True)], name + "/matmul"),
                            weight(key + ".bias")], name)

    def layer_norm(x: str, key: str, name: str) -> str:
        return node("LayerNormalization", [x, weight(key + ".weight"), weight(key + ".bias")], name,
                    axis=-1, epsilon=EPSILON)

    split_shape = const("split_shape", np.array([0, 0, HEADS, HIDDEN // HEADS], dtype=np.int64))
    merge_shape = const("merge_shape", np.array([0, 0, HIDDEN], dtype=np.int64))
    scale = const("scale", np.array(np.sqrt(HIDDEN // HEADS), dtype=np.float32))
    sqrt2 = const("sqrt2", np.array(np.sqrt(2.0), dtype=np.float32))
    half = const("half", np.array(0.5, dtype=np.float32))
    one = const("one", np.array(1.0, dtype=np.float32))
    zero_i = const("zero_i", np.array([0], dtype=np.int64))
    one_i = const("one_i", np.array([1], dtype=np.int64))
    minus_one_i = const("minus_one_i", np.array([-1], dtype=np.int64))
    index_one = const("index_one", np.array(1, dtype=np.int64))

    # 词向量 + 位置向量 + 句子类型向量（全是第 0 类）
    words = node("Gather", [weight("embeddings.word_embeddings.weight"), "input_ids"], "emb/words", axis=0)
    length = node("Gather", [node("Shape", ["input_ids"], "emb/shape"), index_one], "emb/length", axis=0)
    length_1d = node("Unsqueeze", [length, zero_i], "emb/length_1d")
    positions = node("Slice", [weight("embeddings.position_embeddings.weight"), zero_i, length_1d, zero_i],
                     "emb/positions")
    token_type = node("Gather", [weight("embeddings.token_type_embeddings.weight"), const("type_index", np.array(0, dtype=np.int64))],
                      "emb/type", axis=0)
    x = node("Add", [node("Add", [words, positions], "emb/add_pos"), token_type], "emb/add_type")
    x = layer_norm(x, "embeddings.LayerNorm", "emb/norm")

    for i in range(layers):
        p = f"encoder.layer.{i}."
        n = f"layer{i}/"

        def heads(tensor: str, name: str, perm: list) -> str:
            return node("Transpose", [node("Reshape", [tensor, split_shape], name + "/split")], name, perm=perm)

        q = heads(linear(x, p + "attention.self.query", n + "q"), n + "q_heads", [0, 2, 1, 3])
        k = heads(linear(x, p + "attention.self.key", n + "k"), n + "k_heads", [0, 2, 3, 1])
        v = heads(linear(x, p + "attention.self.value", n + "v"), n + "v_heads", [0, 2, 1, 3])
        scores = node("Div", [node("MatMul", [q, k], n + "scores_raw"), scale], n + "scores")
        context = node("MatMul", [node("Softmax", [scores], n + "probs", axis=-1), v], n + "context")
        context = node("Reshape", [node("Transpose", [context], n + "context_t", perm=[0, 2, 1, 3]), merge_shape],
                       n + "context_merged")
        attended = linear(context, p + "attention.output.dense", n + "attn_out")
        x = layer_norm(node("Add", [attended, x], n + "attn_res"), p + "attention.output.LayerNorm", n + "attn_norm")

        inner = linear(x, p + "intermediate.dense", n + "inner")
        # GELU 的精确形式：x * 0.5 * (1 + erf(x / sqrt(2)))
        erf = node("Erf", [node("Div", [inner, sqrt2], n + "gelu_div")], n + "gelu_erf")
        gelu = node("Mul", [inner, node("Mul", [node("Add", [erf, one], n + "gelu_add"), half], n + "gelu_half")],
                    n + "gelu")
        x = layer_norm(node("Add", [linear(gelu, p + "output.dense", n + "out"), x], n + "out_res"),
                       p + "output.LayerNorm", n + "out_norm")

    rows = node("Squeeze", [x, zero_i], "rows")
    nodes.append(helper.make_node("Slice", [rows, one_i, minus_one_i, zero_i], ["char_features"], name="drop_special"))

    graph = helper.make_graph(
        nodes, "chinese_roberta_hidden_minus_3",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "length"])],
        [helper.make_tensor_value_info("char_features", TensorProto.FLOAT, ["chars", HIDDEN])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="gsv-ipad")
    model.ir_version = 9
    onnx.save(model, path)
