# 声间：在 iPad 上离线运行的 GPT-SoVITS

目标：把电脑上的 GSV-TTS-Lite（GPT v3 + SoVITS v2ProPlus 底模，靠参考音频定音色）做成一个 iPad 原生 App，不连电脑也能合成。

目标设备：A16 芯片的 iPad，iPadOS 26。没有 Mac：用 GitHub Actions 的 macOS 机器编译未签名 IPA，再在 Windows 上用 Sideloadly 免费签名安装。

仓库里只有代码。模型、角色音频、合成结果都在本机的 `work\` 里，不进仓库。

## 做法

- 模型：计算图用 [Genie-TTS](https://github.com/High-Logic/Genie-TTS) 的 ONNX 模板，权重取自本机 `D:\k\models` 的底模。
- 分工：参考音频相关的计算（HuBERT 语义特征、声纹、音色向量、参考文本特征）在电脑上预先算成「角色包」；iPad 只做「文字 → 音素 → 语义 → 波形」。
- iPad 端用 Swift + ONNX Runtime 推理，模型共约 570MB（全精度）。

## 进度

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 1 | 电脑上转换模型并验证声音 | 完成 |
| 2 | GitHub 自动编译，装到 iPad | 代码已写好，等第一次编译和安装 |
| 3 | iPad 上跑通模型推理 | 基准测试版已写好，等真机结果 |
| 4 | 在 iPad 上实现中文、日文文本处理 | 未开始 |
| 5 | 正式界面：选角色和情绪、输入、播放、保存 | 未开始 |

### 第一阶段结果

`tools\compare.py` 用 5 组预设各合成一句，对比原版 GSV-TTS-Lite（显卡）和 ONNX（CPU）：

- 4 组的语音识别结果与原文完全一致，音色相似度与原版持平（ONNX 0.756–0.843，原版 0.749–0.832）。
- 1 组漏读了很短的第一句。`tools\probe_first_sentence.py` 的结果：文本前加一个「。」后，日文 4 次全对，中文 4 次对 3 次，剩下 1 次提前结束。对策：按句切分、加前导句号、语义数量明显偏少时重试。
- 速度（i7-14650HX）：4 线程实时率约 0.75，2 线程约 1.04。A16 只有 2 个性能核，预计比实时慢，以真机测出的数字为准。

## 目录

| 位置 | 内容 |
| --- | --- |
| `App\` | iPad App 的 Swift 源码。`SynthEngine.swift` 是推理，`TensorPack.swift` 读数据包 |
| `project.yml` | XcodeGen 工程描述，Xcode 工程由它生成 |
| `.github\workflows\build.yml` | 自动编译。产物挂在名为 `ci` 的 Release 上 |
| `tools\` | 电脑端的 Python 脚本 |
| `work\`（不进仓库） | `onnx\` 转换后的模型，`ipad\` 要拷到 iPad 的文件，`out\` 合成结果 |

## 电脑端脚本

Python 环境在 `tools\.venv`，由 `D:\k\env` 创建并共用它的包（PyTorch、transformers、pyopenjtalk 等），另外只装了 genie-tts（不带依赖）、onnx、onnxruntime。脚本不会改动 `D:\k`。

| 脚本 | 作用 |
| --- | --- |
| `convert_models.py` | 本机底模（safetensors）→ `work\onnx\` 的 ONNX 模型 |
| `ref_pipeline.py` | 电脑上的参考流程：算角色包并用 ONNX 合成。其中的 `OnnxSynth` 与 App 里的 `SynthEngine` 一一对应 |
| `export_pack.py` | `bundle` 生成基准测试用的角色包、文本包并集中到 `work\ipad\`；`verify` 只用这些文件合成一遍 |
| `compare.py` | 原版与 ONNX 的对比 |
| `probe_first_sentence.py` | 检查漏读第一句的问题 |

```powershell
& "D:\ios\gsv-ipad\tools\.venv\Scripts\python.exe" "D:\ios\gsv-ipad\tools\export_pack.py" bundle
```

实现上要留意的三点：

- 不要 `import genie_tts`：它导入时会检查 GenieData 并提示从 Hugging Face 下载。脚本只用它包里的模板文件。
- GSV-TTS-Lite 对 SoVITS 解码器做了 `remove_weight_norm`，转换脚本用 `v = weight`、`g = ||weight||` 还原，结果与合并后的权重相同。
- ONNX Runtime 的 Objective-C 接口没有 bool 张量类型，转换脚本把解码器的结束信号改成了 int64 输出 `stop_flag`。

## 装到 iPad

1. 每次推送后 GitHub 自动编译。安装包在 [ci Release](https://github.com/lunar0cean/gsv-ipad/releases/tag/ci) 的 `GSVPad-unsigned.ipa`，同一处的 `build-info.txt` 记着对应的提交，`errors.txt` 是编译报错摘要。
2. 用 Sideloadly 和自己的 Apple ID 签名安装，步骤和「页间」相同。
3. 用 iTunes 的「文件共享」或 iPad 的「文件」App，把 `work\ipad\` 里的 `models`、`voices`、`tests` 三个文件夹拷进 App 的文件夹。
4. 打开 App 点「重新检查」，选角色和文本，点「合成」。界面会显示各步用时、实时率和内存占用。
