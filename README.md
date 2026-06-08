# Flux

<p align="center">
  <img src="https://img.shields.io/badge/Version-5.10.0-blue.svg?style=for-the-badge" alt="Version">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Maintained%20By-Orvex%20Research-purple.svg?style=for-the-badge" alt="Orvex Research">
</p>

---

**Flux** is a state-of-the-art model-definition and machine learning framework developed and maintained by **Orvex Research**. It provides unified, simple, and highly efficient abstractions to download, train, and run cutting-edge models across various modalities, including:

- 📝 **Natural Language Processing** (Text Generation, Classification, Translation)
- 👁️ **Computer Vision** (Image Classification, Depth Estimation, Object Detection)
- 🎙️ **Audio Processing** (Speech Recognition, Audio Classification)
- 🖼️ **Multimodal Systems** (Visual Question Answering, Document Parsing)

---

## 🚀 Key Features

* **Unified API:** Access 100+ state-of-the-art architectures using identical interfaces.
* **Multi-Framework Flexibility:** Seamlessly load and move models across PyTorch, JAX, and TensorFlow.
* **Low-Compute Optimization:** Integrated with state-of-the-art quantization techniques and execution backends (Eager, Compiled, Kernelized).
* **Direct Hub Integration:** Access over 1M+ open-weight checkpoints on the model registry.

---

## 📦 Installation

Flux requires Python 3.10+ and PyTorch 2.4+.

### 1. From PyPI (Recommended)
Install the stable version of `flux` along with its PyTorch dependencies:
```bash
pip install flux[torch]
```

### 2. From Source
If you are developing or running custom forks:
```bash
git clone https://github.com/Orvex-Research/Flux.git
cd Flux
pip install -e ".[torch]"
```

---

## 📖 Quickstart Guide

Flux allows you to load and run models in just a few lines of code.

### Text Generation Pipeline
```python
from flux import pipeline

# Instantiate the text generation pipeline
generator = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")

# Generate text
output = generator("The future of artificial intelligence is ")
print(output)
```

### Multi-turn Conversation (Chat)
```python
import torch
from flux import pipeline

# Set up the chat history
chat = [
    {"role": "system", "content": "You are a helpful assistant from Orvex Research."},
    {"role": "user", "content": "Explain what Flux is in one sentence."}
]

# Load pipeline with BF16 precision
generator = pipeline(
    task="text-generation", 
    model="meta-llama/Meta-Llama-3-8B-Instruct", 
    dtype=torch.bfloat16, 
    device_map="auto"
)

response = generator(chat, max_new_tokens=100)
print(response[0]["generated_text"][-1]["content"])
```

---

## 🛠️ How to Publish/Upload to PyPI

To publish your custom version of the `flux` package to **PyPI**, follow these steps:

### Step 1: Install Packaging Tools
Make sure you have `build` and `twine` installed in your environment:
```bash
pip install --upgrade build twine
```

### Step 2: Build the Package
Generate the source distribution and built wheels:
```bash
python -m build
```
This will create a `dist/` directory containing the `.tar.gz` and `.whl` files.

### Step 3: Upload to PyPI
Use `twine` to securely upload your package to PyPI:
```bash
python -m twine upload dist/*
```
> [!TIP]
> You will be prompted to enter your PyPI token. For security, use your PyPI API token (`pypi-` prefix) as the password and `__token__` as the username.

---

## ⚖️ License & Attribution

This project is licensed under the **Apache License 2.0**. 

```text
Copyright 2026 Orvex Research. All rights reserved.
Portions Copyright 2018- The Hugging Face team. All rights reserved.
```
For more information, please read the [LICENSE](./LICENSE) file.
