"""角色包（.gsvpack）的读写。只依赖 numpy。

容器很简单，Swift 端不依赖第三方库就能读：
    4 字节  "GSVP"
    4 字节  小端 uint32，头部长度 N
    N 字节  UTF-8 JSON：{"kind", "meta", "tensors": [{"name", "dtype", "shape", "offset", "length"}]}
    之后    各张量的原始字节（小端），offset 从这里算起
"""
import json
import os
import struct

import numpy as np

DTYPES = {"i64": np.int64, "f32": np.float32}


def write_pack(path: str, kind: str, meta: dict, tensors: dict) -> None:
    entries, blobs, offset = [], [], 0
    for name, arr in tensors.items():
        dtype = "i64" if arr.dtype == np.int64 else "f32"
        data = np.ascontiguousarray(arr.astype(DTYPES[dtype])).tobytes()
        entries.append({"name": name, "dtype": dtype, "shape": list(arr.shape), "offset": offset, "length": len(data)})
        blobs.append(data)
        offset += len(data)
    header = json.dumps({"kind": kind, "meta": {k: str(v) for k, v in meta.items()}, "tensors": entries},
                        ensure_ascii=False).encode("utf-8")
    header += b" " * ((-(8 + len(header))) % 8)  # 让数据区按 8 字节对齐
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"GSVP")
        f.write(struct.pack("<I", len(header)))
        f.write(header)
        for blob in blobs:
            f.write(blob)


def read_pack(path: str):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:4] != b"GSVP":
        raise ValueError(f"不是数据包：{path}")
    n = struct.unpack("<I", raw[4:8])[0]
    header = json.loads(raw[8:8 + n].decode("utf-8"))
    base = 8 + n
    tensors = {}
    for e in header["tensors"]:
        buf = raw[base + e["offset"]: base + e["offset"] + e["length"]]
        tensors[e["name"]] = np.frombuffer(buf, dtype=DTYPES[e["dtype"]]).reshape(e["shape"]).copy()
    return header["kind"], header["meta"], tensors
