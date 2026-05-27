
# Dasheng-AudioGen

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.XXXXX)
[![Hugging Face Model](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/mispeech/Dasheng-AudioGen)
[![Hugging Face Demo](https://img.shields.io/badge/HuggingFace-Demo-orange?logo=huggingface)](https://huggingface.co/spaces/mispeech/Dasheng-AudioGen)
[![Web Demo](https://img.shields.io/badge/Website-Demo-181717?logo=google-chrome)](https://nieeim.github.io/Dasheng-AudioGen-Web/)


[**English**](./README.md) | [**中文**](./README_zh.md)

**Dasheng-AudioGen** is a unified audio generation model that can jointly synthesize **intelligible speech, music, sound effects, and environmental acoustics** from text descriptions.

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

## Models

| Model | HuggingFace  | Language |
|-------|-------------|:--------:|
| Dasheng-AudioGen | [mispeech/Dasheng-AudioGen](https://huggingface.co/mispeech/Dasheng-AudioGen)  | English |
| Dasheng-AudioGen-Multilingual | [mispeech/Dasheng-AudioGen-Multilingual](https://huggingface.co/mispeech/Dasheng-AudioGen-Multilingual)  | Multilingual |

> **Note:** The current multilingual model has notably higher synthesis error rates for all non-English languages. For English-only use cases, the base model (`mispeech/Dasheng-AudioGen`) is recommended.

## Installation

```bash
pip install torch torchaudio "transformers<5" einops
```

> Tested with Python 3.10, torch 2.8.0+cu128, transformers 4.57. Not compatible with transformers 5.x.

## Prompt Format

Dasheng-AudioGen uses structured tags to describe different audio aspects. A valid prompt **must start with the `<|caption|>` tag**, which provides the overall scene description. Other tags are optional and can be included as needed.

| Tag | Description | Required |
|-----|-------------|:--------:|
| `<\|caption\|>` | Overall audio scene description | Yes |
| `<\|speech\|>` | Speaker identity and speaking style | No |
| `<\|asr\|>` | Spoken transcript / dialogue | No |
| `<\|sfx\|>` | Sound effects | No |
| `<\|music\|>` | Background music | No |
| `<\|env\|>` | Environmental ambience | No |

**Rules:**
- The prompt must begin with `<|caption|>` — prompts without it will be rejected.
- Only include tags that are relevant; omit tags with no content (e.g., skip `<|music|>` if there is no music).

> **Multilingual note:** When using the multilingual model, all descriptive tags (`caption`, `speech`, `sfx`, `music`, `env`) should be in **English**. Only the `<|asr|>` field (the actual speech content to synthesize) uses the target language.

## Quick Start

### Usage 1: Aspect-wise Composition

Pass each aspect as a named argument. The `caption` field is required; all other fields are optional.

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

### Usage 2: Pre-formatted Prompt String

Pass a complete tagged string via the `prompt` parameter. The string must start with `<|caption|>`.

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

### Batch Inference

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

### Generation Parameters

```python
import torchaudio
from transformers import AutoModel

model = AutoModel.from_pretrained("mispeech/Dasheng-AudioGen", trust_remote_code=True).cuda()

prompt = model.compose_prompt(caption="A dog barking in a park")
audio = model.generate(
    prompts=prompt,
    num_steps=25,              # number of denoising steps (default: 25)
    guidance_scale=5.0,        # classifier-free guidance scale (default: 5.0)
    sway_sampling_coef=-1.0,   # sway sampling coefficient (default: -1.0, 0 for linear)
)
torchaudio.save("output.wav", audio.cpu(), 16000)
```

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
