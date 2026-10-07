# GPT Sovits（iPad 离线语音合成）：交接说明

写于 2026-10-07，给接手的 AI 或开发者用。这份说明自成一体，不依赖上一位 AI 的会话记忆。

一句话现状：**试验成功**。App 已经装在用户的 A16 iPad 上，能离线把中文和日文合成语音。用户下一步打算把这套能力并入另一个 iOS 程序。

## 0. 动手之前必须知道的

### 用户定下的规则

| 规则 | 具体怎么做 |
| --- | --- |
| 要原生 App、在 iPad 本机推理 | 「iPad 用浏览器连电脑」的方案被明确否决过，不要再提 |
| 仓库公开，只放代码 | 模型权重、角色音频、参考文本、合成结果一律不进仓库，都在本机的 `work\`（已被 `.gitignore` 排除）。提交前检查一遍 |
| 不改动 `D:\k` | 那是用户正在用的 GSV-TTS-Lite。只读、只调用 |
| 不要做测试界面 | 用户要的是「输入文字 → 出声音」，基准测试、调试面板之类不要放进界面 |
| 下载的东西放 D 盘 | C 盘空间紧。pip 缓存、临时文件也指到 D 盘 |
| 界面风格 | 冷色（夜蓝底、冷白字、一个淡蓝强调色），宋体，没有底板和卡片，细线加小菱形。暖黄和褐色被否决过。详见 `D:\ios\AI_HANDOFF.md` 第 0 节 |
| 沟通 | 用户说中文，不是程序员。回复用简体中文、少用术语。给命令时一条一个代码块 |

`D:\ios\AI_HANDOFF.md` 第 0 节还有用户给上一个项目定的规则（不碰用户的浏览器、缺资源直接说等），在 `D:\ios` 下做事同样适用。

### 这台电脑

| 项目 | 情况 |
| --- | --- |
| 系统 | Windows 11，终端是 PowerShell 5.1，不支持 `&&` |
| 内存 | 15.7GB，用户常同时开着别的大程序。**跑占内存的任务会把内存撑爆**：报「not enough memory」或访问冲突，桌面端的 AI 应用也跟着退出过两次。重活要一个一个跑、放在独立进程里、跑之前看一眼可用内存 |
| 显卡 | RTX 4060 Laptop 8GB。只在跑原版对照和语音识别时用 |
| 网络 | 直连 GitHub 和 PyPI 很慢（20–60KB/s）。pip 用阿里云镜像；GitHub Release 的文件在地址前加 `https://gh-proxy.com/`，实测约 1MB/s，下完用 GitHub API 给的 sha256 校验 |
| 没有 Mac | 编译全靠 GitHub Actions。本机不能编译 Swift |
| 没装的东西 | GitHub 命令行工具 `gh`；Rust 工具链（有 rustup，没装任何 toolchain） |
| git | 推送走 Git Credential Manager。仓库内提交身份设为 `lunar0cean <lunar0cean@users.noreply.github.com>`。**提交信息用 `git commit -F 文件`**，PowerShell 5.1 会弄坏带引号的 `-m` |
| 安装 | Sideloadly + 免费 Apple ID（7 天重签）。传文件用微软商店的「Apple 设备」 |

## 1. 这是什么

| 项目 | 内容 |
| --- | --- |
| 来源 | 用户电脑上的 GSV-TTS-Lite（`D:\k`）：GPT v3（`s1v3`）+ SoVITS v2ProPlus（`s2Gv2ProPlus`）官方底模，不微调，靠参考音频定音色和语气 |
| 目标 | A16 芯片的 iPad，iPadOS 26.7。离线、不依赖电脑 |
| 仓库 | https://github.com/lunar0cean/gsv-ipad （公开，`main` 分支），本机在 `D:\ios\gsv-ipad` |
| App | 显示名「GPT Sovits」，包名 `com.lunar0cean.gsvpad`，最低 iOS 17，只支持 iPad |
| 功能 | 选角色 → 选输入语言（日文或中文）→ 输入文字 → 生成。按标点切句，合成一句播一句，结束后可重听、导出 |
| 角色 | 4 个：黎瑟、森の神、小町鸫·傲慢（日文），示例·认真（中文）。音色和语气用同一条参考音频。来源写在本机的 `work\presets.json` |

## 2. 整体流程

```
电脑（一次性）                                   iPad（每次合成）
参考音频 + 参考文本                               输入文字
   │ HuBERT、声纹、音色编码器、文本处理             │ frontend.js / zh.js（JavaScriptCore）
   ▼                                              │   日文的分词和重音来自 Rust 库
角色包 .gsvpack ───── 拷进 App ─────┐              ▼
                                   │           音素编号（按标点切成若干片段）
本机底模 safetensors                │              │
   │ convert_models.py              ▼              ▼
   ▼                            T2S 编码器 → 首步解码 → 逐步解码循环 → 声码器
ONNX 模型 ──────── 拷进 App ────────┘              │（ONNX Runtime，CPU，全精度）
                                                  ▼
                                           32kHz 波形 → 裁剪淡入淡出 → 播放、存 WAV
```

为什么这样分工：HuBERT（189MB）和声纹模型（185MB）只在处理参考音频时用，预先算好就不用装进 iPad。文本处理用 JavaScript 写，是因为它能在电脑上用 Node 对照原版 Python 逐句验证，iPad 上又不需要额外的运行环境。

## 3. 目录

| 位置 | 内容 |
| --- | --- |
| `App\` | Swift 源码，见第 7 节 |
| `App\Frontend\` | 文本处理脚本和词典数据，作为资源打进 App |
| `App\Native\gsv_native.h` | Rust 库的 C 接口，用作桥接头文件 |
| `Config\Info.plist` | 只声明文件共享的两项，其余由 Xcode 生成后合并 |
| `native\` | Rust 库（日文文本 → OpenJTalk 全上下文标签） |
| `Tests\SmokeTests.swift` | 在 iPad 模拟器上跑的冒烟测试 |
| `frontend\test\` | 文本处理的对照样本和 Node 测试 |
| `tools\` | 电脑端的 Python 脚本，见第 8 节 |
| `project.yml` | XcodeGen 工程描述。Xcode 工程由它生成，不进仓库 |
| `.github\workflows\build.yml` | 自动编译和测试 |
| `README.md`、`NOTICE.md` | 项目说明；第三方代码和数据的来源 |
| `work\`（不进仓库） | `onnx\` 转换后的模型；`ipad\models`、`ipad\voices` 要拷到 iPad 的文件；`presets.json` 角色清单；`out\` 合成结果；`tmp\`、`cache\` |
| `D:\ios\builds\GPTSovits-v*\` | 下载好的安装包和安装说明（仓库之外） |

## 4. 模型

### 4.1 转换

`tools\convert_models.py` 把 `D:\k\models\s1v3` 和 `s2Gv2ProPlus` 的 safetensors 转成 ONNX。计算图不是自己导出的，用的是 [Genie-TTS](https://github.com/High-Logic/Genie-TTS) 2.0.2 包里自带的模板（`genie_tts\Data\`），脚本只把权重按模板要求的名字和顺序写成 `.bin`，再让模板指向它。

三处要留意：

- **不要 `import genie_tts`**。它导入时会检查 GenieData 并提示从 Hugging Face 下载。脚本用 `importlib.util.find_spec` 只取包目录。
- GSV-TTS-Lite 加载官方权重时改过名（见 `D:\k\gsv_tts\Loader.py`），还对 SoVITS 解码器做了 `remove_weight_norm`。转换脚本反向改名，并用 `v = weight`、`g = ||weight||` 还原，数学上与合并后的权重相同。
- ONNX Runtime 的 Objective-C 接口**没有 bool 张量类型**。解码器的结束信号 `stop_condition_tensor` 是 bool，脚本加了一个 Cast 节点，改成 int64 输出 `stop_flag`。

### 4.2 文件和接口

都是 opset 20、全精度。`work\onnx\` 里是全套，`work\ipad\models\` 里是 iPad 要用的 7 个文件（共约 570MB）。

| 模型 | 权重文件 | 输入 | 输出 |
| --- | --- | --- | --- |
| `t2s_encoder_fp32.onnx` | `t2s_encoder_fp32.bin` 11.5MB | `ref_seq` i64 [1,R]，`text_seq` i64 [1,T]，`ref_bert` f32 [R,1024]，`text_bert` f32 [T,1024]，`ssl_content` f32 [1,768,S] | `x` f32，`prompts` i64 [1,P] |
| `t2s_first_stage_decoder_fp32.onnx` | `t2s_shared_fp32.bin` 307MB | `x`，`prompts` | `y`，`y_emb`，24 层的 `present_k_layer_i`、`present_v_layer_i` |
| `t2s_stage_decoder_fp32.onnx` | 同上（两个解码器共用） | `iy`，`iy_emb`，24 层的 `past_k_layer_i`、`past_v_layer_i` | `y`，`y_emb`，`stop_flag` i64 标量，24 层的 `present_*` |
| `vits_fp32.onnx` | `vits_fp32.bin` 249MB | `text_seq` i64 [1,T]，`pred_semantic` i64 [1,1,N]，`ge` f32 [1,1024,1]，`ge_advanced` f32 [1,512,1] | `audio` f32，32kHz |
| `prompt_encoder_fp32.onnx`（只在电脑上用） | `prompt_encoder_fp32.bin` 88.5MB | `ref_audio` f32 [1,L]（32kHz），`sv_emb` f32 [1,20480] | `ge`，`ge_advanced` |

推理步骤（`App\SynthEngine.swift`，与 `tools\ref_pipeline.py` 的 `OnnxSynth` 一一对应）：

1. 编码器 → `x`、`prompts`。
2. 首步解码 → 状态（`y`、`y_emb`、48 个缓存张量）。
3. 循环跑逐步解码，上一步的输出按**位置**喂给下一步的输入（靠 `outputNames()` 和 `inputNames()` 的顺序对应），直到 `stop_flag` 非零或到达上限 1000 步。一步是 0.04 秒语音。
4. 取 `y` 里 `prompts` 之后的部分，去掉触发结束的最后一个，再去掉编号 ≥ 1024 的（结束符），得到语义。
5. 声码器 → 波形。

采样（top-k 等）写死在计算图里，每次结果不同。模板里采样参数的具体数值**没有核对过**，原版网页界面的默认值是 top_k 15、temperature 1.0、repetition_penalty 1.35、noise_scale 0.5。

### 4.3 角色包

`.gsvpack` 是自定义的简单容器（格式见 `tools\gsvpack.py`，Swift 端是 `App\TensorPack.swift`）：4 字节 `GSVP`，4 字节小端头部长度，UTF-8 JSON 头（`kind`、`meta`、张量列表），然后是原始小端数据。

| 张量 | 形状 | 来历 |
| --- | --- | --- |
| `ref_seq` | i64 [1,R] | 参考文本的音素编号 |
| `ref_bert` | f32 [R,1024] | 参考文本的 BERT 特征。中文是真特征，日文是全零 |
| `ssl_content` | f32 [1,768,S] | 参考音频的 HuBERT 特征，50 帧每秒 |
| `ge`、`ge_advanced` | f32 [1,1024,1]、[1,512,1] | 音色编码器的输出 |

`meta` 里有 `name`（界面上显示的名字）和 `lang`（`ja` 或 `zh`）。文件名前面的序号决定显示顺序。

导出：编辑 `work\presets.json`，然后运行下面这条。每个角色在独立进程里算，失败自动重试。

```powershell
& "D:\ios\gsv-ipad\tools\.venv\Scripts\python.exe" "D:\ios\gsv-ipad\tools\export_pack.py" bundle
```

预处理与原版一致（`ref_pipeline.py` 的 `build_voice`）：参考音频去掉结尾静音、补 0.3 秒静音后送 HuBERT；32kHz 音频加声纹向量送音色编码器。

## 5. 文本处理

### 5.1 结构

| 文件 | 内容 |
| --- | --- |
| `App\Frontend\data.js` | 音素符号表，732 个。由 `tools\make_golden.py ja` 生成 |
| `App\Frontend\frontend.js` | 入口、日文处理、切句 |
| `App\Frontend\zh.js` | 中文处理（1281 行）：文本规范化、jieba 分词和词性、pypinyin 词语切分、变调、儿化 |
| `App\Frontend\zh_dict.txt`、`zh_hmm.json`、`zh_pinyin.json`、`zh_misc.json` | 中文词典数据共约 11MB，由 `tools\make_zh_data.py` 从本机的 jieba_fast、pypinyin 导出。第一次处理中文时才载入 |

宿主（iPad 上是 `App\TextFrontend.swift`，电脑上是 `frontend\test\harness.js`）要提供两个函数：

- `__jaLabels(text)`：日文片段 → OpenJTalk 全上下文标签，每行一个。
- `__loadText(name)`：读同目录下的数据文件。

入口：

- `GSV.g2p(text, lang)` → `{norm, phones, ids}`。与原版 `text_to_phonemes` 逐音素一致。
- `GSV.prepare(text, lang)` → `[{text, ids, pause}]`。整段文字切成可以逐个合成的片段。`GSV.prepareJSON` 是给 Swift 用的版本，出错时返回 `{"error": ...}`。

语种由调用方指定（`ja` 或 `zh`），没有做自动判断。App 里默认跟随角色的语种，可以手动切换。

### 5.2 `prepare` 做了什么

这一层是自己写的，不是移植，改它不影响与原版的一致性：

1. 按换行拆行。**先去掉引号**（`「」『』“”‘’"'`）。
2. 行尾不是标点就补一个 `.`。
3. 按原版 `cut_text` 的规则切句：标点集合和最小长度 10 取自原版网页界面的默认值。
4. 每个片段前面加一个 `。` 再转音素。这是为了减少「漏读很短的第一句」（见第 10 节）。
5. **丢掉读不出声音的片段**（音素全是标点）。
6. 片段后的停顿：0.2 秒乘以末尾标点对应的系数。

第 1 步和第 5 步是 2026-10-07 加的，起因见第 10.1 节。

### 5.3 与原版的对应和精度

| 部分 | 移植自 | 验证 |
| --- | --- | --- |
| 日文 | `D:\k\gsv_tts\GPT_SoVITS\G2P\Japanese\japanese.py` | 40 句样本逐音素一致 |
| 中文规范化 | 原版的 `Chinese\Normalization\`（来自 PaddleSpeech） | 121 句样本加 2998 句随机压力测试，规范化文本、分词（带词性）、音素全部一致 |
| 分词和词性 | jieba_fast（Python 部分加 C 部分） | 同上 |
| 词语读音切分 | pypinyin 的 `mmseg` | 同上 |
| 变调、儿化 | 原版的 `tone_sandhi.py`、`chinese.py` | 同上 |

移植时为了逐音素一致，照搬了原版的几处错误和细节，不要「顺手修掉」：

- jieba_fast 不带词性的分词走 C 实现，带词性的走 Python 实现，两者在概率相同时取的候选不一样；C 实现每个位置最多记 12 个候选；Viterbi 的分数有下限钳位。`zh.js` 里分别是 `calcRouteFast`、`calcRoute`、`finalViterbi`，注释里写了区别。
- 时间范围里判断「半」用的是前一个时刻的分钟数；温度单位总是读「度」；「mm」会先被「m」替换成「米米」。这些是原版的行为。
- 原版遇到读不出来的拼音会直接报错中断；`zh.js` 改成跳过那个音节。

日文的分词、读音、重音在 iPad 上由 [jpreprocess](https://github.com/jpreprocess/jpreprocess) 0.15.0 提供（OpenJTalk 的 Rust 重写，内置 naist-jdic 词典）。40 句样本上与电脑上的 pyopenjtalk 完全一致。**电脑上的原版还加载了一个 16MB 的自定义用户词典**（`D:\k\models\g2p\ja\userdict.csv`），iPad 上没有，所以生僻词和专名的读音可能不同，这一点没有系统测过。

### 5.4 测试

```powershell
node D:\ios\gsv-ipad\frontend\test\test_ja.js
```

```powershell
node D:\ios\gsv-ipad\frontend\test\test_zh.js
```

```powershell
node D:\ios\gsv-ipad\frontend\test\test_prepare.js
```

重新生成对照样本（改了句子清单之后）。两种语言**必须分两次运行**，pyopenjtalk 和 jieba_fast 在同一个进程里先后初始化会崩溃：

```powershell
& "D:\ios\gsv-ipad\tools\.venv\Scripts\python.exe" "D:\ios\gsv-ipad\tools\make_golden.py" ja
```

```powershell
& "D:\ios\gsv-ipad\tools\.venv\Scripts\python.exe" "D:\ios\gsv-ipad\tools\make_golden.py" zh
```

`make_golden.py zh --extra 文本文件` 可以对任意每行一句的文本生成对照样本，再用 `node test_zh.js 样本.json` 比对，压力测试就是这么做的。

## 6. Rust 库（`native\`）

只做一件事：日文文本 → 全上下文标签。导出两个 C 函数：

```c
char *gsv_ja_labels(const char *text);  // 每行一个标签；失败返回 NULL
void gsv_free(char *ptr);
```

- 依赖只有 `jpreprocess = "=0.15.0"`，开 `naist-jdic` 特性把词典编进库里。编译时要联网取词典。
- 编出三份：`aarch64-apple-ios`（真机）、`aarch64-apple-ios-sim`（模拟器）、本机的命令行工具 `gsv-labels`（对照测试用）。
- 没有提交 `Cargo.lock`（本机没有 cargo 生成不了），每次编译会解析到依赖的最新兼容版本。想要可复现，在有 Rust 的机器上生成后提交。
- 音素和韵律符号是 `frontend.js` 从标签里取的，不在 Rust 里。

## 7. App 代码

| 文件 | 职责 |
| --- | --- |
| `GSVPadApp.swift` | 入口 |
| `RootView.swift` | 界面（SwiftUI）。模型或角色包不全时显示拷贝说明 |
| `Theme.swift` | 颜色、字体、按钮样式 |
| `AppModel.swift` | 界面状态和合成流程。扫描文档目录找模型和角色包，预热，逐句合成，写日志 |
| `TextFrontend.swift` | 建 JavaScriptCore 环境，注入 `__jaLabels`、`__loadText`，加载脚本，调用 `GSV.prepareJSON` |
| `SynthEngine.swift` | ONNX Runtime 推理 |
| `TensorPack.swift` | 读 `.gsvpack` |
| `AudioOutput.swift` | `AudioPost`（裁剪、淡入淡出，与原版 `_trim_audio`、`_fade` 相同）、`StreamPlayer`（AVAudioEngine 排队播放）、`WavWriter` |

实现要点：

- **线程**：合成在一条串行后台队列上跑，界面状态只在主线程上改。`AppModel` 没有标 `@MainActor`，靠手工 `DispatchQueue.main`。`loadIfNeeded` 从后台队列 `main.sync` 取缓存，主线程不能反过来等后台队列。
- **内存**：解码循环每一步的输出里有 48 个随长度增长的缓存张量，循环体必须包在 `autoreleasepool` 里，否则内存一路涨上去。
- **预热**：启动后在后台载入模型和脚本；有中文角色时顺带处理一句中文，把词典载入做掉。
- **重试**：`SynthEngine` 里语义数量少于音素数量的 0.8 倍就重来，最多 3 次。`AppModel` 里一句抛错重试 1 次，还不行就跳过这一句、继续后面的，结束时报告哪几句没合成出来。
- **屏幕常亮**：合成期间设 `isIdleTimerDisabled`。锁屏后系统会挂起 App。没有开后台音频模式。
- **文件**：模型和角色包从 App 的文档目录里找，放在哪一层子文件夹都行（7 个模型文件要在同一层）。合成结果存 `outputs\时间.wav`，每次合成的经过追加到 `outputs\log.txt`。
- **线程数**：写死 2，对应 A16 的 2 个性能核。没有试过别的值。
- **目标文本的 BERT 特征是全零**，见第 10.2 节。

依赖：ONNX Runtime 的 Swift 包 1.24.2（模块名 `OnnxRuntimeBindings`）；JavaScriptCore、AVFoundation（系统自带）；Rust 静态库。

安装包 41.7MB。解开后主程序 115MB，大头是静态链接的 ONNX Runtime 和编进 Rust 库的日文词典。

## 8. 电脑端脚本

Python 环境在 `tools\.venv`，由 `D:\k\env`（Python 3.11、PyTorch 2.10）创建并共用它的包，另外只装了 genie-tts（`--no-deps`）、onnx、onnxruntime 1.22.1。

| 脚本 | 作用 |
| --- | --- |
| `convert_models.py` | 本机底模 → `work\onnx\` |
| `export_pack.py bundle` | 按 `work\presets.json` 导出角色包，并把模型集中到 `work\ipad\` |
| `export_pack.py verify` | 只用 `work\ipad\` 里的文件、按 iPad 的做法（目标文本 BERT 全零）给每个角色合成一句，存到 `work\out\verify\`。**这就是 iPad 上应该出来的声音** |
| `make_golden.py ja`、`zh` | 符号表和文本处理的对照样本 |
| `make_zh_data.py` | 导出中文词典数据到 `App\Frontend\` |
| `ref_pipeline.py` | 电脑上的参考流程。`Frontend`（调用原版文本处理）、`build_voice`、`OnnxSynth` |
| `compare.py` | 原版（显卡）与 ONNX（CPU）各合成一遍，用语音识别和声纹相似度打分 |
| `asr_check.py 文件夹` | 用 `D:\k\models\qwen3_asr` 转写一个文件夹里的 wav |
| `probe_first_sentence.py` | 检查漏读第一句 |
| `onnx_pack.py`、`gsvpack.py` | 写 ONNX 权重和角色包的公共代码，只依赖 numpy 和 onnx |
| `ci_fixtures.py` | 生成模拟器测试用的随机权重模型和假角色包 |

导入 `pyopenjtalk` 之前必须先设好 `OPEN_JTALK_DICT_DIR`，否则它会自己联网下载词典、卡很久。`make_golden.py` 里已经处理。

## 9. 编译、发布、安装

### 9.1 自动编译

推送到 `main` 且改动了 `App\`、`Config\`、`Tests\`、`native\`、`frontend\`、`project.yml`、工作流文件或三个公共 Python 模块时触发。只改文档不触发。全程约 6 分钟（有缓存时）。

步骤：编译 Rust 库（真机、模拟器、本机工具）→ Node 对照测试（只出报告）→ XcodeGen 生成工程 → 编译未签名的真机版并打成 IPA，**检查 Info.plist 里有文件共享的两项，缺了算失败** → 生成随机权重的测试模型 → 在 iPad 模拟器上跑 `SmokeTests`（只出报告）→ 把产物传到名为 `ci` 的 Release。

本机没有 `gh`，结果都从 Release 取，仓库公开所以不用登录：

| 文件 | 内容 |
| --- | --- |
| `build-info.txt` | 对应的提交和状态。**先看它，确认是不是你刚推的那个提交** |
| `errors.txt` | Rust 和 Xcode 的报错摘要 |
| `parity.txt` | Node 对照测试结果 |
| `sim-test.txt` | 模拟器测试结果，以 `GSVTEST` 开头的行是耗时 |
| `GSVPad-unsigned.ipa` | 安装包。编译失败时会被撤掉 |
| `build.log`、`rust-build.log`、`sim-test.log` | 完整日志 |

查编译状态（公开接口，每小时限 60 次）：

```
https://api.github.com/repos/lunar0cean/gsv-ipad/actions/runs?per_page=3
```

下载安装包（走加速线路）：

```
https://gh-proxy.com/https://github.com/lunar0cean/gsv-ipad/releases/download/ci/GSVPad-unsigned.ipa
```

校验码在 `https://api.github.com/repos/lunar0cean/gsv-ipad/releases/tags/ci` 的 `assets[].digest` 里。

### 9.2 模拟器测试测了什么

真模型不在仓库里，所以 `tools\ci_fixtures.py` 用随机数填出结构完全相同的模型。

| 测试 | 内容 |
| --- | --- |
| `testFrontendMatchesNode` | 样本里的每段中日文，JavaScriptCore 的结果要与 Node 算出的逐音素相同。同时验证了 iOS 版 Rust 库与本机版给出的标签相同 |
| `testSynthesisRunsEndToEnd` | 文字 → 音素 → 编码 → 解码循环 → 声码器 → WAV 文件，再测一次取消 |
| `testAudioPostHandlesEdgeCases` | 空输入、极短输入、含非有限数值的输入 |

它验证的是流程，不是音质，耗时也不代表真机（随机权重、共享虚拟机）。界面和播放没有测试覆盖。

### 9.3 安装和拷文件

1. Sideloadly 签名安装 IPA，覆盖安装不会丢文档目录里的文件。
2. 在 iPad 上打开一次 App。
3. 电脑上打开「Apple 设备」→ 左边「文件」→ App 列表里选「GPT Sovits」→ 把 `work\ipad\models` 里的 7 个文件和 `work\ipad\voices` 里的角色包加进去。按苹果的说明有两种加法：点「添加文件」按钮，或者把文件拖到 App 列表里的名字上。**往右边的文档列表里拖是拖不进去的**，用户在这里卡过。用户最后传成功了，但用的是哪一种我没有确认。文件不用放在文件夹里。
4. 回到 App 点「重新检查」。

装好的安装包和给用户看的说明在 `D:\ios\builds\GPTSovits-v*\`。

## 10. 真机上的结果和已知问题

用户在 2026-10-07 装了 0.1.1，确认能合成中文和日文。合成速度的数字用户没有报，**A16 上的实时率仍然未知**。电脑上的参考：i7-14650HX，4 线程实时率约 0.75，2 线程约 1.0。

### 10.1 日文长文本中途停下（已找到原因并修复，修复后未在真机复测）

现象：十几行日文对话，每行用「」括着。每次生成到不固定的某一行就停下，状态栏显示「没有生成任何语义，无法合成」，偶尔也能全部生成。

原因：行尾是「」」时，切句逻辑认为这行没有以标点收尾，补了一个句号，切出一个只含 `」.` 的片段。它转出来的音素只有句号，模型对它生成不出语义（电脑上实测 12 次里 5 次是 0 个），而旧代码只要有一句抛错就中断整段。

修复（提交 `02b5f04`、`0d250a4`，版本 0.1.2）：切句前去掉引号；丢掉读不出声音的片段；一句失败只跳过这一句；合成期间不让屏幕锁定；每次合成写日志。`frontend\test\test_prepare.js` 覆盖了这个情况。

如果真机上还有中途停下的情况，先看 App 文档目录里的 `outputs\log.txt`，里面有每一句的音素数、语义数、重试次数、耗时和报错。

### 10.2 中文效果不理想（原因未确定）

用户的评价是「有些不尽人意」，没有说具体是哪方面。我能确认的只有内容正确：电脑上按 iPad 的做法合成的中文，语音识别结果与原文一字不差。音质和语气我没法听，没有验证。

可能的原因，按我估计的可能性排序，都没有验证：

1. **缺中文语调模型。** 原版对中文目标文本用 `chinese-roberta-wwm-ext-large` 提取特征，iPad 上喂的是全零。参考文本那一半（`ref_bert`）是真特征，目标文本是零，这种组合模型在训练时见过（中文参考加别的语种目标），但对中文目标来说少了语调信息。
2. **只有一个中文角色。** 另外三个是日文角色，用它们读中文是跨语种合成，口音和原版一样会有，但用户可能主要在这上面试。
3. **采样参数可能和原版不同。** 模板里写死的 top-k、温度、噪声比例没有核对过。
4. **切句偏碎。** 最小长度 10 是原版默认值，逗号处也会切，每段单独合成，段与段之间的语气不连贯。

建议先做一个听感对比再决定投入：同一个中文角色、同一句话，在电脑上合成三份，分别是「目标文本用真 BERT」「目标文本全零（即现在的 iPad）」「参考和目标都全零」。第二份 `export_pack.py verify` 已经能出；另外两份要在 `ref_pipeline.py` 的基础上写几行。如果第一份明显更好，就值得把 BERT 搬上 iPad，做法见第 12 节。**这个对比需要加载 1.3GB 的模型，跑之前确认电脑可用内存在 5GB 以上。**

### 10.3 其他已知情况

| 问题 | 说明 |
| --- | --- |
| 很短的第一句偶尔被漏读 | GPT-SoVITS 的通病。每个片段前加句号后明显改善但没有根除：电脑上「你好。我今天有点累…」加句号后 4 次里仍有 1 次只读了「你好」就停了 |
| 英文单词 | 中文模式下被直接丢弃（原版在语种锁定时也是这样）；日文模式下 OpenJTalk 按字母读。没有移植英文处理 |
| 日文里的括号等符号 | 会变成 `UNK` 音素送进模型，这是原版的行为。引号已经在切句前去掉 |
| 自定义用户词典 | 电脑上有，iPad 上没有，见第 5.3 节 |
| 免费签名 | 7 天要重签一次 |
| 没有自动判断语种 | 原版的启发式（按假名、汉字逐字判断）没有移植，靠用户手动选 |

## 11. 并入其他 iOS 程序

### 11.1 要带走的东西

| 东西 | 说明 |
| --- | --- |
| `App\SynthEngine.swift`、`TensorPack.swift`、`TextFrontend.swift`、`AudioOutput.swift` | 核心，不依赖界面。`AppModel.swift` 里的合成流程可以照着改写 |
| `App\Frontend\` 下的 7 个文件 | 作为资源打进包里 |
| `App\Native\gsv_native.h` 和 `native\` | 桥接头文件和 Rust 库 |
| ONNX Runtime 的 Swift 包 | `https://github.com/microsoft/onnxruntime-swift-package-manager`，1.24.2，产品名 `onnxruntime` |
| 模型和角色包 | 不在仓库里，运行时从文档目录读，或在本机打进安装包（见 11.3） |

最小调用顺序：

```swift
let frontend = try TextFrontend()
let engine = try SynthEngine(modelDirectory: modelsURL, threads: 2)
let voice = try TensorPack(url: voicePackURL)
for segment in try frontend.prepare(text, language: "ja") {
    guard let result = try engine.synthesize(voice: voice, phonemes: segment.ids,
                                             isCancelled: { false }) else { break }
    let samples = AudioPost.trimAndFade(result.samples, sampleRate: SynthEngine.sampleRate)
    // 播放或保存 samples，然后静音 segment.pause 秒
}
```

### 11.2 要注意的地方

- **工程配置**：`SWIFT_OBJC_BRIDGING_HEADER` 指向 `gsv_native.h`；`LIBRARY_SEARCH_PATHS` 按 SDK 分别指向真机和模拟器的 Rust 库目录；`OTHER_LDFLAGS` 加 `-lgsv_native -liconv -lresolv`。照 `project.yml` 抄。目标程序如果已经有桥接头文件，把那两行声明并进去。
- **资源重名**：7 个资源文件现在平铺在包的根目录，名字很普通（`data.js`、`frontend.js`、`zh.js`）。并入别的程序前建议放进子目录，并相应改 `TextFrontend.swift` 里两处 `Bundle.main.url(forResource:)`。
- **`.js` 文件的构建阶段**：Xcode 默认可能把 `.js` 当源码处理。`project.yml` 里用 `buildPhase: resources` 明确指定了。
- **文件共享**：要让用户从电脑拷模型，Info.plist 里必须有 `UIFileSharingEnabled` 和 `LSSupportsOpeningDocumentsInPlace`。**`INFOPLIST_KEY_UIFileSharingEnabled` 这种写法 Xcode 不认**，要写在真实的 Info.plist 文件里（0.1 版就栽在这里）。
- **只有 arm64**：Rust 库只编了 arm64 的真机和模拟器，没有 x86_64 模拟器。
- **内存**：模型全精度约 570MB，加上运行时和中文词典，估计常驻 1GB 上下，没有在真机上量过。目标程序如果本身占内存多，要留意。
- **耗时操作都别放主线程**：建 `SynthEngine` 在模拟器上要 7–9 秒，真机没量过；第一次处理中文要载入词典。
- **`SynthEngine` 不是线程安全的**，同一时间只在一个线程上用。
- **`D:\ios\reader`（「页间」阅读器）** 是用户另一个 iPad 程序，同样用 XcodeGen 和 GitHub Actions，最可能是并入的目标。它的仓库和这个仓库是分开的。

### 11.3 模型怎么带上

现在的做法是用户用「Apple 设备」把文件加进 App 的文档目录，这一步用户操作时卡过两次。有一个没做的替代方案：安装包是 zip，可以在本机用脚本把 `work\ipad\` 里的文件加进 `Payload\*.app\`，再交给 Sideloadly 签名。这样模型不经过 GitHub，用户只装一个文件。代价是安装包变成约 600MB，每周重签要重传一遍，并且 App 要改成也从包内找模型。包内文件名建议用纯英文，避免签名工具处理中文文件名出问题。

## 12. 后续可以做的事

| 事项 | 做法和估计 |
| --- | --- |
| 在真机上量速度 | 让用户报「完成」那一行的两个数字，或取 `outputs\log.txt`。这是最该先拿到的数据 |
| 把中文 BERT 搬上 iPad | 先做第 10.2 节的听感对比。要做的话：Genie 在 Hugging Face（`High-Logic/Genie` 的 `GenieData(Optional)/RoBERTa`）有现成的 `RoBERTa.onnx`（599MB），输入 `input_ids`、`attention_mask`、`repeats`，直接输出按音素展开的特征；也可以从 `D:\k\models\chinese-roberta-wwm-ext-large` 自己导出，原版取的是 `hidden_states[-3]`、去掉首尾特殊符号、按每个字的音素数重复（见 `D:\k\gsv_tts\GPT_SoVITS\Featurizer\cnroberta.py`）。规范化后的中文只有汉字和 6 种标点，分词器可以简化成按字查词表。`zh.js` 的 `g2p` 现在不返回每个字的音素数（`word2ph`），要加上。内存会再多 0.6–1.2GB |
| 提速 | 没试过的方向：线程数 3 或 4；声码器换 CoreML 或 XNNPACK 执行器；权重半精度或量化。先有真机数据再说 |
| 英文 | 原版的英文处理在 `D:\k\gsv_tts\GPT_SoVITS\G2P\English\`，词典在 `D:\k\models\g2p\en\`，没有移植 |
| 日文用户词典 | jpreprocess 支持用户词典，格式和 OpenJTalk 的不同，要转换 |
| 后台合成 | 加 `UIBackgroundModes` 的 `audio`，锁屏后也能继续。没做 |
| 提交 `Cargo.lock` | 见第 6 节 |

## 13. 踩过的坑

| 坑 | 结论 |
| --- | --- |
| 电脑内存被撑爆 | 症状是 Python 报「not enough memory」或访问冲突（退出码 3221225477），桌面应用也退出。不是代码问题，等内存空出来再跑 |
| pyopenjtalk 自己联网下载词典 | 导入前设 `OPEN_JTALK_DICT_DIR` |
| pyopenjtalk 和 jieba_fast 同进程崩溃 | 两种语言分开进程跑 |
| `import genie_tts` 有副作用 | 只取它的包目录 |
| ORT 的 Objective-C 接口没有 bool | 模型输出里的 bool 要转成整数 |
| `INFOPLIST_KEY_UIFileSharingEnabled` 无效 | 写进真实的 Info.plist，并在编译里检查 |
| 「Apple 设备」拖不进文件 | 不能往文档列表里拖。用「添加文件」按钮，或拖到 App 的名字上 |
| PowerShell 5.1 弄坏提交信息里的引号 | `git commit -F 文件` |
| 提交失败后 `git push` 仍会把旧提交推上去 | 推送前看一眼 `git log` |
| 只改文档的提交不触发编译 | Release 里的安装包对应的是上一个触发了编译的提交，以 `build-info.txt` 为准 |
| 行尾的引号被切成空片段 | 见第 10.1 节 |

## 14. 相关文档

| 文档 | 内容 |
| --- | --- |
| `README.md` | 项目说明，面向仓库的读者 |
| `NOTICE.md` | 第三方代码和数据的来源、许可证 |
| `D:\ios\builds\GPTSovits-v*\安装说明.md` | 给用户看的安装和使用步骤 |
| `D:\ios\AI_HANDOFF.md` | `D:\ios` 下另一个项目（三维场景）的交接说明，第 0 节的用户规则通用 |
| `D:\ios\reader\README.md`、`ipad-reader\docs\INSTALL.zh-CN.md` | 「页间」阅读器及其安装流程 |
