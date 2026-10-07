# GPT Sovits：在 iPad 上离线运行的 GPT-SoVITS

把电脑上的 GSV-TTS-Lite（GPT v3 + SoVITS v2ProPlus 底模，靠参考音频定音色）做成一个 iPad 原生 App：选角色、输入文字、生成语音，全程不连电脑。

目标设备：A16 芯片的 iPad，iPadOS 26。没有 Mac：用 GitHub Actions 的 macOS 机器编译未签名 IPA，再在 Windows 上用 Sideloadly 免费签名安装。

仓库里只有代码和开源词典数据。模型、角色音频、合成结果都在本机的 `work\` 里，不进仓库。

接手这个项目，或者要把它并入别的 iOS 程序，先读 [HANDOFF.md](HANDOFF.md)。

## 做法

| 环节 | 在哪里跑 | 怎么做 |
| --- | --- | --- |
| 参考音频 → 角色包 | 电脑 | HuBERT 语义特征、声纹、音色向量、参考文本特征预先算好，存成 `.gsvpack` |
| 文字 → 音素 | iPad | `App\Frontend\*.js`，由系统自带的 JavaScriptCore 运行。日文的分词、读音、重音来自 `native\` 里的 Rust 库（[jpreprocess](https://github.com/jpreprocess/jpreprocess)，OpenJTalk 的 Rust 重写） |
| 中文语调特征 | iPad | 可选。`chinese-roberta-wwm-ext-large` 的前 22 层，571MB，只在输入中文时用 |
| 音素 → 语义 → 波形 | iPad | Swift + ONNX Runtime。计算图用 [Genie-TTS](https://github.com/High-Logic/Genie-TTS) 的模板，权重取自本机底模，共约 570MB |

文本处理用 JavaScript 写，是因为它能在电脑上用 Node 对照原版 Python 逐句验证，iPad 上又不需要额外的运行环境。

## 进度

| 内容 | 状态 |
| --- | --- |
| 模型转成 ONNX，并对照原版验证声音 | 完成 |
| 日文文本处理 | 完成。JS 移植与原版逐音素一致（40 句样本）；jpreprocess 给出的标签在这些样本上也与电脑上的 OpenJTalk 完全一致 |
| 中文文本处理 | 完成。JS 移植与原版逐音素一致（121 句样本，另有 2998 句随机生成的压力测试）。英文单词暂时会被跳过 |
| App：选角色、输入文字、边合成边播放、导出音频 | 已在 A16 iPad 上跑通，能合成中文和日文，速度约为实时的 1.2 倍。每次推送都在 iPad 模拟器上跑端到端冒烟测试（见下） |
| 中文语调模型（RoBERTa） | 0.2.0 接入。ONNX 版与原版 PyTorch 数值一致；模型文件可选，没有时退回全零特征。真机上的内存占用和听感还没有验证 |

每次推送后 GitHub 会在 iPad 模拟器上跑 `Tests\SmokeTests.swift`：

- 文本处理：样本里的每段中日文在 JavaScriptCore 里的结果，要与 Node 算出的逐个音素相同。
- 推理：真模型不在仓库里，所以用随机权重填出结构相同的模型（`tools\ci_fixtures.py`），把「文字 → 音素 → 编码 → 解码循环 → 声码器 → 音频文件」整条流程跑一遍。合成出来的是噪声，验证的是流程，不是音质，也不代表真机速度。

已知情况：

- 很短的第一句偶尔会被漏读。对策是按标点切句、每句前加一个句号、语义数量明显偏少时自动重试。
- 速度：A16 iPad 上实测，合成用时约为语音时长的 1.2 倍（一分钟的日文对话用了 68 秒）。边合成边播放时，等完第一段之后基本能接上。电脑 CPU（i7-14650HX）上 2 线程约 1.0 倍。

## 目录

| 位置 | 内容 |
| --- | --- |
| `App\` | iPad App 的 Swift 源码。`SynthEngine` 推理，`TextFrontend` 文本处理，`AppModel` 流程，`RootView` 界面 |
| `App\Frontend\` | 文本处理脚本和数据，打进 App 当资源 |
| `native\` | Rust 库：日文文本 → OpenJTalk 全上下文标签 |
| `frontend\test\` | 文本处理的对照样本和测试 |
| `Tests\` | 在 iPad 模拟器上跑的冒烟测试 |
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
| `make_golden.py ja`、`zh` | 生成符号表和文本处理的对照样本（两种语言要分开运行） |
| `make_zh_data.py` | 从本机的 jieba_fast、pypinyin 导出中文词典数据到 `App\Frontend\` |
| `ref_pipeline.py` | 电脑上的参考流程，`OnnxSynth` 与 App 里的 `SynthEngine` 一一对应 |
| `compare.py`、`probe_first_sentence.py` | 原版与 ONNX 的对比；漏读第一句的检查 |
| `asr_check.py <文件夹>` | 用本机的语音识别模型转写合成结果，检查内容有没有读对 |
| `export_bert.py`、`bert_onnx.py` | 生成中文语调模型的 ONNX（直接拼计算图，不用 `torch.onnx.export`）和字表，并与原版对照 |
| `onnx_pack.py`、`gsvpack.py` | 写 ONNX 权重和角色包的公共代码，只依赖 numpy 和 onnx |
| `ci_fixtures.py` | 生成模拟器测试用的随机权重模型和假角色包 |

文本处理的测试不需要 Python：

```powershell
node D:\ios\gsv-ipad\frontend\test\test_ja.js
```

```powershell
node D:\ios\gsv-ipad\frontend\test\test_zh.js
```

实现上要留意的几点：

- 不要 `import genie_tts`：它导入时会检查 GenieData 并提示从 Hugging Face 下载。脚本只用它包里的模板文件。
- 导入 `pyopenjtalk` 之前必须先设好 `OPEN_JTALK_DICT_DIR`，否则它会自己联网下载词典。
- 特征提取很占内存（中文 RoBERTa 一个就 1.3GB）。电脑内存紧张时会报「not enough memory」或直接崩溃，所以 `export_pack.py` 每个角色单独起一个进程，失败自动重试。
- GSV-TTS-Lite 对 SoVITS 解码器做了 `remove_weight_norm`，转换脚本用 `v = weight`、`g = ||weight||` 还原。
- ONNX Runtime 的 Objective-C 接口没有 bool 张量类型，转换脚本把解码器的结束信号改成了 int64 输出 `stop_flag`。

## 装到 iPad

1. 每次推送后 GitHub 自动编译。安装包在 [ci Release](https://github.com/lunar0cean/gsv-ipad/releases/tag/ci) 的 `GSVPad-unsigned.ipa`；同一处的 `build-info.txt` 记着对应的提交，`errors.txt` 是报错摘要，`parity.txt` 是文本处理的对照结果，`sim-test.txt` 是模拟器测试的结果。
2. 用 Sideloadly 和自己的 Apple ID 签名安装。
3. 在电脑的「Apple 设备」里点「文件」，选「GPT Sovits」，把 `work\ipad\` 里 `models` 和 `voices` 两个文件夹中的文件加进去：用「添加文件」按钮，或者拖到 App 的名字上。往文档列表里拖是拖不进去的。中文语调模型 `work\ipad\bert\roberta_fp16.onnx` 也这样加进去，它是可选的。
4. 打开 App，选角色、输入文字、点「生成」。
