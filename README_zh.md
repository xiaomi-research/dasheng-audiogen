# Dasheng-AudioGen

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.XXXXX)
[![Hugging Face Model](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/mispeech/Dasheng-AudioGen)
[![Hugging Face Demo](https://img.shields.io/badge/HuggingFace-Demo-orange?logo=huggingface)](https://huggingface.co/spaces/mispeech/Dasheng-AudioGen)
[![Web Demo](https://img.shields.io/badge/Website-Demo-181717?logo=google-chrome)](https://nieeim.github.io/Dasheng-AudioGen-Web/)


[**English**](./README.md) | [**中文**](./README_zh.md)

**Dasheng-AudioGen** 是一个统一的音频生成模型，能够根据文本描述同时合成**语音、音乐、音效和环境声**。

<p align="center">
  <video
    src="https://github.com/user-attachments/assets/497f5688-8731-4830-8ee7-b9cf4234d900"
    controls
    autoplay
    muted
    loop
    playsinline
    width="85%">
  </video>
</p>

## 模型

| 模型 | HuggingFace  | 语言支持 |
|------|-------------|-----------|:--------:|
| Dasheng-AudioGen | [mispeech/Dasheng-AudioGen](https://huggingface.co/mispeech/Dasheng-AudioGen)  | 英语 |
| Dasheng-AudioGen-Multilingual | [mispeech/Dasheng-AudioGen-Multilingual](https://huggingface.co/mispeech/Dasheng-AudioGen-Multilingual) | 多语言 |

> **注意：** 当前多语言模型在所有非英语语言上的合成错误率都明显偏高。如果仅需英语生成，建议使用基础模型 (`mispeech/Dasheng-AudioGen`)。

## 安装

```bash
pip install torch torchaudio "transformers<5" einops
```

> 已在 Python 3.10、torch 2.8.0+cu128、transformers 4.57 上测试通过。已知不兼容 transformers 5.x。

## 快速开始

### 基本用法

```python
import torchaudio
from transformers import AutoModel

model = AutoModel.from_pretrained("mispeech/Dasheng-AudioGen", trust_remote_code=True).cuda()

# 或者加载多语言模型
# model = AutoModel.from_pretrained("mispeech/Dasheng-AudioGen-Multilingual", trust_remote_code=True).cuda()

audio = model.generate("A dog barking in a park")
torchaudio.save("output.wav", audio.cpu(), 16000)
```

### 分项 Prompt

使用 `compose_prompt` 分别描述不同的音频维度：

```python
prompt = model.compose_prompt(
    caption="A gritty detective narrating over the sound of heavy rain and a melancholic solo jazz saxophone.",
    speech="gritty deep male voice",
    music="melancholic solo saxophone",
    env="distant urban ambience",
    sfx="heavy rain hitting pavement",
    asr="The city never sleeps, but it sure knows how to cry.",
)
audio = model.generate(prompt)
torchaudio.save("output.wav", audio.cpu(), 16000)
```

也可以直接传入包含标签的完整字符串：

```python
audio = model.generate(
    "<|caption|> A helicopter passing overhead. <|sfx|> Rhythmic helicopter blade sounds. <|env|> Open sky ambience."
)
```

### 批量推理

```python
prompts = [
    model.compose_prompt(caption="A cat meowing softly.", sfx="Soft cat meow."),
    model.compose_prompt(caption="Thunder rolling in the distance.", env="Stormy night ambience."),
    model.compose_prompt(caption="A piano playing a gentle melody.", music="Soft piano ballad."),
]
audios = model.generate(prompts)

for i, audio in enumerate(audios):
    torchaudio.save(f"output_{i}.wav", audio.unsqueeze(0).cpu(), 16000)
```

### 生成参数

```python
audio = model.generate(
    prompts="A dog barking in a park",
    num_steps=25,              # 去噪步数（默认：25）
    guidance_scale=5.0,        # 无分类器引导强度（默认：5.0）
    sway_sampling_coef=-1.0,   # sway 采样系数（默认：-1.0，设为 0 使用线性调度）
)
```

## Prompt 格式

Dasheng-AudioGen 使用结构化标签来描述不同的音频维度：

| 标签 | 描述 |
|------|------|
| `<\|caption\|>` | 整体音频场景描述 |
| `<\|speech\|>` | 说话人身份和说话风格 |
| `<\|asr\|>` | 语音转写内容 / 对话文本 |
| `<\|sfx\|>` | 音效 |
| `<\|music\|>` | 背景音乐 |
| `<\|env\|>` | 环境音 |

你可以传入包含标签的完整 `content` 字符串，也可以通过 `compose_prompt` 分别提供各维度字段（`caption`、`speech`、`asr`、`sfx`、`music`、`env`），系统会自动拼接。

> **多语言 prompt 规范：** 使用多语言模型时，所有描述性标签（`caption`、`speech`、`sfx`、`music`、`env`）应使用**英文**填写，仅 `<|asr|>` 字段（实际要合成的语音内容）使用目标语言。

## 致谢

Dasheng-AudioGen 由**小米 LLM PLUS** 和 **上海交通大学 X-LANCE** 联合开发。

## 引用

```bibtex
@article{dasheng-audiogen,
  title={Dasheng-AudioGen},
  author={},
  journal={arXiv preprint arXiv:2505.XXXXX},
  year={2025}
}
```

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 发布。
