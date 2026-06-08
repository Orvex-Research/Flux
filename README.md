# Flux

<p align="center">
  <img src="https://img.shields.io/badge/Version-5.10.0-blue.svg?style=for-the-badge" alt="Version">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Maintained%20By-Orvex%20Research-purple.svg?style=for-the-badge" alt="Orvex Research">
</p>

---

**Flux** acts as the core model-definition framework for state-of-the-art machine learning with text, computer vision, audio, video, and multimodal models, for both inference and training. 

It is developed and maintained by **Orvex Research**, providing a centralized, eco-system-wide framework. `flux` is the pivot across frameworks: if a model definition is supported, it is fully compatible with the majority of training frameworks (Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, ...), inference engines (vLLM, SGLang, TGI, ...), and adjacent modeling libraries (llama.cpp, mlx, ...).

We pledge to help support new state-of-the-art models and democratize their usage by having their model definition be simple, customizable, and efficient.

---

## 🚀 Why Use Flux?

1. **Easy-to-use State-of-the-Art Models:**
   - High performance on natural language understanding & generation, computer vision, audio, video, and multimodal tasks.
   - Low barrier to entry for researchers, engineers, and developers.
   - Few user-facing abstractions with just three classes to learn.
   - A unified API for using all our pretrained models.

2. **Lower Compute Costs, Smaller Carbon Footprint:**
   - Share trained models instead of training from scratch.
   - Reduce compute time and production costs.
   - Hundreds of model architectures with 1M+ pretrained checkpoints across all modalities.

3. **Choose the Right Framework for Every Phase:**
   - Train state-of-the-art models in 3 lines of code.
   - Move a single model between PyTorch/JAX/TF2.0 frameworks at will.
   - Pick the right framework for training, evaluation, and production.

4. **Easily Customize to Your Needs:**
   - Model internals are exposed as consistently as possible.
   - Model files can be used independently of the library for quick experiments.

---

## 📦 Installation

Flux requires Python 3.10+ and PyTorch 2.4+.

### 1. From PyPI
```bash
# Using pip
pip install "flux[torch]"

# Using uv
uv pip install "flux[torch]"
```

### 2. From Source
```bash
git clone https://github.com/Orvex-Research/Flux.git
cd Flux

# Using pip
pip install -e '.[torch]'

# Using uv
uv pip install -e '.[torch]'
```

---

## 📖 Quickstart Guide

Get started with Flux right away using the **Pipeline** API. The `Pipeline` is a high-level inference class that handles preprocessing, model forwarding, and returning clean predictions.

### 1. Text Generation
```python
from flux import pipeline

# Instantiate a pipeline and specify a model
generator = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
response = generator("the secret to baking a really good cake is ")
print(response)
```

### 2. Multi-turn Conversational Chat
You can chat with models programmatically or directly from your terminal:
```bash
flux chat Qwen/Qwen2.5-0.5B-Instruct
```

Or via the Python API:
```python
import torch
from flux import pipeline

chat = [
    {"role": "system", "content": "You are a helpful assistant developed by Orvex Research."},
    {"role": "user", "content": "Tell me three fun things to do in New York."}
]

generator = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", dtype=torch.bfloat16, device_map="auto")
response = generator(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

### 3. Speech Recognition
```python
from flux import pipeline

asr = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
result = asr("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
print(result)
# {'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

### 4. Image Classification
```python
from flux import pipeline

classifier = pipeline(task="image-classification", model="facebook/dinov2-small-imagenet1k-1-layer")
result = classifier("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
print(result)
```

---

## 🎨 Supported Modalities & Models

Flux supports a wide range of architectures. Here are some featured models for various use cases:

### 🎙️ Audio
- **Audio Classification:** CLAP
- **Automatic Speech Recognition:** Parakeet, Whisper, GLM-ASR, Moonshine-Streaming
- **Keyword Spotting:** Wav2Vec2
- **Speech to Speech Generation:** Moshi
- **Text to Audio/Speech:** MusicGen, CSM

### 👁️ Computer Vision
- **Automatic Mask Generation:** SAM
- **Depth Estimation:** DepthPro
- **Image Classification:** DINO v2
- **Keypoint Detection & Matching:** SuperPoint, SuperGlue
- **Object Detection:** RT-DETRv2
- **Pose Estimation:** VitPose
- **Universal Segmentation:** OneFormer
- **Video Classification:** VideoMAE

### 🤝 Multimodal
- **Audio/Text to Text:** Voxtral, Audio Flamingo
- **Document Question Answering:** LayoutLMv3
- **Image/Text to Text:** Qwen-VL, Llava-OneVision
- **Image Captioning & OCR:** BLIP-2, GOT-OCR2
- **Table Question & Answering:** TAPAS
- **Visual Question Answering:** Llava, Kosmos-2

### 📝 NLP (Natural Language Processing)
- **Masked Word Completion:** ModernBERT
- **Named Entity Recognition:** Gemma
- **Question Answering & Summarization:** Mixtral, BART
- **Translation:** T5
- **Text Generation:** Llama, Qwen


