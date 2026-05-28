# Dasheng-AudioGen

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv)](https://arxiv.org/abs/2605.27838)
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
|------|-------------|:--------:|
| Dasheng-AudioGen | [mispeech/Dasheng-AudioGen](https://huggingface.co/mispeech/Dasheng-AudioGen)  | 英语 |
| Dasheng-AudioGen-Multilingual | [mispeech/Dasheng-AudioGen-Multilingual](https://huggingface.co/mispeech/Dasheng-AudioGen-Multilingual) | 多语言 |

> **注意：** 当前多语言模型在所有非英语语言上的合成错误率都明显偏高。如果仅需英语生成，建议使用基础模型 (`mispeech/Dasheng-AudioGen`)。

## 安装

```bash
pip install torch torchaudio "transformers<5" einops
```

> 已在 Python 3.10、torch 2.8.0+cu128、transformers 4.57 上测试通过。已知不兼容 transformers 5.x。

## Prompt 格式

Dasheng-AudioGen 使用结构化标签来描述不同的音频维度。有效的 prompt **必须以 `<|caption|>` 标签开头**，用于描述整体音频场景。其他标签为可选项，按需使用。

| 标签 | 描述 | 是否必需 |
|------|------|:--------:|
| `<\|caption\|>` | 整体音频场景描述 | 是 |
| `<\|speech\|>` | 说话人身份和说话风格 | 否 |
| `<\|asr\|>` | 语音转写内容 / 对话文本 | 否 |
| `<\|sfx\|>` | 音效 | 否 |
| `<\|music\|>` | 背景音乐 | 否 |
| `<\|env\|>` | 环境音 | 否 |

**规则：**
- Prompt 必须以 `<|caption|>` 开头，否则会报错。
- 仅包含有实际内容的标签；没有对应内容的标签请省略（例如没有音乐则不传 `<|music|>`）。

> **多语言 prompt 规范：** 使用多语言模型时，所有描述性标签（`caption`、`speech`、`sfx`、`music`、`env`）应使用**英文**填写，仅 `<|asr|>` 字段（实际要合成的语音内容）使用目标语言。

## 快速开始

### 用法一：分维度组装

通过命名参数分别传入各个维度的描述。`caption` 为必填项，其他字段可选。

```python
import torchaudio
from transformers import AutoModel

model = AutoModel.from_pretrained("mispeech/Dasheng-AudioGen", trust_remote_code=True).cuda()

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

### 用法二：传入完整 Prompt 字符串

通过 `prompt` 参数传入预格式化的标签字符串，该字符串必须以 `<|caption|>` 开头。

```python
import torchaudio
from transformers import AutoModel

model = AutoModel.from_pretrained("mispeech/Dasheng-AudioGen", trust_remote_code=True).cuda()

prompt = model.compose_prompt(
    prompt="<|caption|> A gritty detective narrating over the sound of heavy rain and a melancholic solo jazz saxophone. <|speech|> gritty deep male voice <|asr|> The city never sleeps, but it sure knows how to cry. <|sfx|> heavy rain hitting pavement <|music|> melancholic solo saxophone <|env|> distant urban ambience"
)
audio = model.generate(prompt)
torchaudio.save("output.wav", audio.cpu(), 16000)
```

### 批量推理

```python
import torchaudio
from transformers import AutoModel

model = AutoModel.from_pretrained("mispeech/Dasheng-AudioGen", trust_remote_code=True).cuda()

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
import torchaudio
from transformers import AutoModel

model = AutoModel.from_pretrained("mispeech/Dasheng-AudioGen", trust_remote_code=True).cuda()

prompt = model.compose_prompt(caption="A dog barking in a park")
audio = model.generate(
    prompts=prompt,
    num_steps=25,              # 去噪步数（默认：25）
    guidance_scale=5.0,        # 无分类器引导强度（默认：5.0）
    sway_sampling_coef=-1.0,   # sway 采样系数（默认：-1.0，设为 0 使用线性调度）
)
torchaudio.save("output.wav", audio.cpu(), 16000)
```

## 评估

Dasheng-AudioGen 使用混合音频数据集 [MECAT-Caption](https://huggingface.co/datasets/mispeech/MECAT-Caption) 测试集中的英文子集进行评估，约占原始测试集的 60%。

为了保证论文结果可复现，我们在 [`evaluation/`](./evaluation) 文件夹中提供了该英文测试子集对应的 audio ID 和 caption。

## 致谢

Dasheng-AudioGen 由**小米 LLM PLUS** 和 **上海交通大学 X-LANCE** 联合开发。

## 引用
```bibtex id="l3t4d8"
@article{mei2026dashengaudiogen,
  title   = {Dasheng AudioGen: A Unified Model for Generating Coherent Audio Scenes from Text},
  author  = {Jiahao Mei and Heinrich Dinkel and Yadong Niu and Xingwei Sun and Gang Li and Yifan Liao and Jiahao Zhou and Junbo Zhang and Jian Luan and Mengyue Wu},
  journal = {arXiv preprint arXiv:2605.27838},
  year    = {2026}
}
```

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 发布。
