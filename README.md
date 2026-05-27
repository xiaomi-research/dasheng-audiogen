# Dasheng-AudioGen

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.XXXXX)
[![Hugging Face Model](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/mispeech/Dasheng-AudioGen)
[![Hugging Face Demo](https://img.shields.io/badge/HuggingFace-Demo-orange?logo=huggingface)](https://huggingface.co/spaces/mispeech/Dasheng-AudioGen)
[![Web Demo](https://img.shields.io/badge/Website-Demo-181717?logo=google-chrome)](https://nieeim.github.io/Dasheng-AudioGen-Web/)


[**English**](./README.md) | [**中文**](./README_zh.md)

**Dasheng-AudioGen** is a unified audio generation model that can jointly synthesize **intelligible speech, music, sound effects, and environmental acoustics** from text descriptions.

<div align="center">
  <video src="./examples/dasheng-audiogen-demo-video.mp4" width="70%" controls>
    Your browser does not support the video tag.
  </video>
</div>

## Models

| Model | HuggingFace | Text Encoder | Language |
|-------|-------------|-------------|:--------:|
| Dasheng-AudioGen | [mispeech/Dasheng-AudioGen](https://huggingface.co/mispeech/Dasheng-AudioGen) | `google/flan-t5-large` | English |
| Dasheng-AudioGen-Multilingual | [mispeech/Dasheng-AudioGen-Multilingual](https://huggingface.co/mispeech/Dasheng-AudioGen-Multilingual) | `google/mt5-large` | Multilingual |

> **Note:** The current multilingual model has notably higher synthesis error rates for all non-English languages. Languages outside the table above are even less reliable. For English-only use cases, the base model (`mispeech/Dasheng-AudioGen`) is recommended.

## Installation

```bash
pip install torch torchaudio "transformers<5" einops
```

> Tested with Python 3.10, torch 2.8.0+cu128, transformers 4.57. Not compatible with transformers 5.x.

## Quick Start

### Basic Usage

```python
import torchaudio
from transformers import AutoModel

model = AutoModel.from_pretrained("mispeech/Dasheng-AudioGen", trust_remote_code=True).cuda()

# Or load the multilingual model
# model = AutoModel.from_pretrained("mispeech/Dasheng-AudioGen-Multilingual", trust_remote_code=True).cuda()

audio = model.generate("A dog barking in a park")
torchaudio.save("output.wav", audio.cpu(), 16000)
```

### Aspect-wise Prompt

Use `compose_prompt` to describe different audio aspects separately:

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

You can also pass a pre-formatted string with tags directly:

```python
audio = model.generate(
    "<|caption|> A helicopter passing overhead. <|sfx|> Rhythmic helicopter blade sounds. <|env|> Open sky ambience."
)
```

### Batch Inference

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

### Generation Parameters

```python
audio = model.generate(
    prompts="A dog barking in a park",
    num_steps=25,              # number of denoising steps (default: 25)
    guidance_scale=5.0,        # classifier-free guidance scale (default: 5.0)
    sway_sampling_coef=-1.0,   # sway sampling coefficient (default: -1.0, 0 for linear)
)
```

## Prompt Format

Dasheng-AudioGen uses structured tags to describe different audio aspects:

| Tag | Description |
|-----|-------------|
| `<\|caption\|>` | Overall audio scene description |
| `<\|speech\|>` | Speaker identity and speaking style |
| `<\|asr\|>` | Spoken transcript / dialogue |
| `<\|sfx\|>` | Sound effects |
| `<\|music\|>` | Background music |
| `<\|env\|>` | Environmental ambience |

You can either pass a pre-formatted `content` string with tags, or provide individual aspect fields (`caption`, `speech`, `asr`, `sfx`, `music`, `env`) via `compose_prompt` which will be automatically composed.

> **Multilingual prompt convention:** When using the multilingual model, all tags (`caption`, `speech`, `sfx`, `music`, `env`) should be written in **English**. Only the `<|asr|>` field (the actual spoken content to be synthesized) should use the target language.

## Acknowledgments

Dasheng-AudioGen was developed with contributions from **XIAOMI LLM PLUS** and **SJTU X-LANCE**.

## Citation

```bibtex
@article{dasheng-audiogen,
  title={Dasheng-AudioGen},
  author={},
  journal={arXiv preprint arXiv:2505.XXXXX},
  year={2025}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE).
