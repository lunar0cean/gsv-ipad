# GPT Sovits：在 iPad 上离线运行的 GPT-SoVITS

把电脑上的 GSV-TTS-Lite（GPT v3 + SoVITS v2ProPlus 底模，靠参考音频定音色）做成一个 iPad 原生 App：选角色、输入文字、生成语音，全程不连电脑。

目标设备：A16 芯片的 iPad，iPadOS 26。没有 Mac：用 GitHub Actions 的 macOS 机器编译未签名 IPA，再在 Windows 上用 Sideloadly 免费签名安装。

仓库里只有代码和开源词典数据。模型、角色音频、合成结果都在本机的 `work\` 里，不进仓库。

## 做法

| 环节 | 在哪里跑 | 怎么做 |
| --- | --- | --- |
| 参考音频 → 角色包 | 电脑 | HuBERT 语义特征、声纹、音色向量、参考文本特征预先算好，存成 `.gsvpack` |
| 文字 → 音素 | iPad | `App\Frontend\*.js`，由系统自带的 JavaScriptCore 运行。日文的分词、读音、重音来自 `native\` 里的 Rust 库（[jpreprocess](https://github.com/jpreprocess/jpreprocess)，OpenJTalk 的 Rust 重写） |
| 音素 → 语义 → 波形 | iPad | Swift + ONNX Runtime。计算图用 [Genie-TTS](https://github.com/High-Logic/Genie-TTS) 的模板，权重取自本机底模，共约 570MB |

文本处理用 JavaScript 写，是因为它能在电脑上用 Node 对照原版 Python 逐句验证，iPad 上又不需要额外的运行环境。

## 进度

| 内容 | 状态 |
| --- | --- |
| 模型转成 ONNX，并对照原版验证声音 | 完成 |
| 日文文本处理 | 完成。JS 移植与原版逐音素一致（40 句样本）；jpreprocess 与原版 OpenJTalk 的差异由 GitHub 上的对照测试给出 |
| 中文文本处理 | 进行中 |
| App：选角色、输入文字、边合成边播放、导出音频 | 已写好，等真机验证 |
| 中文语调模型（RoBERTa） | 未接入，目标文本的 BERT 特征暂用全零 |

已知情况：

- 很短的第一句偶尔会被漏读。对策是按标点切句、每句前加一个句号、语义数量明显偏少时自动重试。
- 速度：电脑 CPU（i7-14650HX）上 2 线程的实时率约 1.0。A16 只有 2 个性能核，预计比实时慢，以真机为准。

## 目录

| 位置 | 内容 |
| --- | --- |
| `App\` | iPad App 的 Swift 源码。`SynthEngine` 推理，`TextFrontend` 文本处理，`AppModel` 流程，`RootView` 界面 |
| `App\Frontend\` | 文本处理脚本和数据，打进 App 当资源 |
| `native\` | Rust 库：日文文本 → OpenJTalk 全上下文标签 |
| `frontend\test\` | 文本处理的对照样本和测试 |
| `tools\` | 电脑端的 Python 脚本 |
| `project.yml` | XcodeGen 工程描述 |
| `.github\workflows\build.yml` | 自动编译。产物挂在名为 `ci` 的 Release 上 |
| `work\`（不进仓库） | `onnx\` 转换后的模型，`ipad\` 要拷到 iPad 的文件，`presets.json` 要导出的角色，`out\` 合成结果 |

## 电脑端脚本

Python 环境在 `tools\.venv`，由 `D:\k\env` 创建并共用它的包，另外只装了 genie-tts（不带依赖）、onnx、onnxruntime。脚本不会改动 `D:\k`。

| 脚本 | 作用 |
| --- | --- |
| `convert_models.py` | 本机底模（safetensors）→ `work\onnx\` 的 ONNX 模型 |
| `export_pack.py bundle` | 按 `work\presets.json` 导出角色包，并把模型一起集中到 `work\ipad\` |
| `export_pack.py verify` | 只用 `work\ipad\` 里的文件、按 iPad 的做法各合成一句，存到 `work\out\verify\` |
| `make_golden.py` | 生成符号表和文本处理的对照样本 |
| `ref_pipeline.py` | 电脑上的参考流程，`OnnxSynth` 与 App 里的 `SynthEngine` 一一对应 |
| `compare.py`、`probe_first_sentence.py` | 原版与 ONNX 的对比；漏读第一句的检查 |

文本处理的测试不需要 Python：

```powershell
node D:\ios\gsv-ipad\frontend\test\test_ja.js
```

实现上要留意的几点：

- 不要 `import genie_tts`：它导入时会检查 GenieData 并提示从 Hugging Face 下载。脚本只用它包里的模板文件。
- 导入 `pyopenjtalk` 之前必须先设好 `OPEN_JTALK_DICT_DIR`，否则它会自己联网下载词典。
- 特征提取很占内存（中文 RoBERTa 一个就 1.3GB）。电脑内存紧张时会报「not enough memory」或直接崩溃，所以 `export_pack.py` 每个角色单独起一个进程，失败自动重试。
- GSV-TTS-Lite 对 SoVITS 解码器做了 `remove_weight_norm`，转换脚本用 `v = weight`、`g = ||weight||` 还原。
- ONNX Runtime 的 Objective-C 接口没有 bool 张量类型，转换脚本把解码器的结束信号改成了 int64 输出 `stop_flag`。

## 装到 iPad

1. 每次推送后 GitHub 自动编译。安装包在 [ci Release](https://github.com/lunar0cean/gsv-ipad/releases/tag/ci) 的 `GSVPad-unsigned.ipa`；同一处的 `build-info.txt` 记着对应的提交，`errors.txt` 是报错摘要，`parity.txt` 是文本处理的对照结果。
2. 用 Sideloadly 和自己的 Apple ID 签名安装。
3. 用 iTunes 的「文件共享」或 iPad 的「文件」App，把 `work\ipad\` 里的 `models` 和 `voices` 两个文件夹拷进 App 的文件夹。
4. 打开 App，选角色、输入文字、点「生成」。
