# 第三方代码和数据

这个仓库里有一部分内容来自其他开源项目，许可证以各自仓库为准。

| 位置 | 来源 | 说明 |
| --- | --- | --- |
| `App/Frontend/zh_dict.txt`、`zh_hmm.json` | [jieba](https://github.com/fxsjy/jieba)（MIT），经由 jieba_fast | 分词词典和两套 HMM 参数，原样转换格式 |
| `App/Frontend/zh_pinyin.json` | [pypinyin](https://github.com/mozillazg/python-pinyin)（MIT） | 单字和词语读音，按声母、带调韵母的形式导出 |
| `App/Frontend/zh.js` 里的文本规范化 | [PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech) 的 zh_normalization（Apache-2.0） | 从 Python 移植到 JavaScript |
| `App/Frontend/zh.js`、`frontend.js` 的其余部分 | [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)（MIT）及其衍生的 GSV-TTS-Lite | 中文、日文的文字转音素流程，从 Python 移植 |
| `App/Frontend/zh.js` 里的分词和拼音切分 | jieba、jieba_fast、pypinyin | 算法从 Python 和 C 移植 |
| `App/Frontend/zh_misc.json` 里的拼音对照表 | GPT-SoVITS 的 `opencpop-strict.txt` | 原样转换格式 |
| `App/Frontend/zh_bert_vocab.json` | [chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)（Apache-2.0）的分词器词表 | 只取单个汉字和标点的编号。模型权重不在仓库里 |
| `native/`（编译时取得） | [jpreprocess](https://github.com/jpreprocess/jpreprocess)（BSD-3-Clause）和它内置的 naist-jdic 词典 | 日文分词、读音和重音 |
| App（编译时取得） | [ONNX Runtime](https://github.com/microsoft/onnxruntime)（MIT） | 模型推理 |
| `tools/convert_models.py` 用到的计算图模板（不在仓库里） | [Genie-TTS](https://github.com/High-Logic/Genie-TTS) | 从本机安装的 genie-tts 包里读取 |
| `tools/templates/hubert.onnx` | [chinese-hubert-base](https://github.com/TencentGameMate/chinese_speech_pretrain)（经由 GPT-SoVITS） | 只有计算图，权重已全部去掉 |
| `tools/templates/sv.onnx` | [3D-Speaker](https://github.com/modelscope/3D-Speaker) 的 ERes2NetV2（经由 GPT-SoVITS） | 只有计算图，权重已全部去掉 |
| `App/Fbank.swift`、`tools/fbank_ref.py` | [torchaudio](https://github.com/pytorch/audio)（BSD-2-Clause）的 `compliance.kaldi.fbank` | 算法按源码重写 |
| `App/Enhancer.swift`、`tools/enhance_ref.py` | [pedalboard](https://github.com/spotify/pedalboard)（GPL-3.0）底层的 [JUCE](https://github.com/juce-framework/JUCE) 算法，[pyloudnorm](https://github.com/csteinmetz1/pyloudnorm)（MIT） | 按公式重写，没有复制代码 |

模型权重和角色音频不在这个仓库里。
