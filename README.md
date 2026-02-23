# DeepSeek-OCR-2 — Setup with UV on Ubuntu 24.10
### (Nvidia Driver 580 / CUDA 13.0 · Python 3.12 · uv)

> **This is a fork of [deepseek-ai/DeepSeek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2)**
> adapted for Ubuntu 24.10 using `uv` instead of conda.
> No code changes — only the environment setup differs.

---

## Contents

- [Requirements](#requirements)
- [1. Clone](#1-clone)
- [2. Create the virtual environment](#2-create-the-virtual-environment)
- [3. Install PyTorch](#3-install-pytorch)
- [4. Install project dependencies](#4-install-project-dependencies)
- [5. Install flash-attn](#5-install-flash-attn)
- [6. Transformers Inference](#6-transformers-inference)
- [7. vLLM Inference (optional)](#7-vllm-inference-optional)
- [Prompts](#prompts)
- [Troubleshooting](#troubleshooting)

---

## Requirements

Before starting, check your environment:

```bash
nvidia-smi          # look for "CUDA Version" in the top-right corner
python3 --version   # 3.11 or 3.12 recommended
uv --version        # 0.4+ required
```

Choose your CUDA index based on your driver:
| Driver CUDA version | PyTorch index to use |
|---|---|
| 11.x | `cu118` |
| 12.1 / 12.2 | `cu121` |
| 12.4+ / 13.x | **`cu124`** ← use this for driver 580 |

---

## 1. Clone

```bash
git clone https://github.com/IcelandicCenterArtificialIntelligence/DeepSeek-OCR-2.git
cd DeepSeek-OCR-2
```

---

## 2. Create the virtual environment

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
```

> If Python 3.12 is not available on your system, uv can download it automatically:
> `uv venv .venv --python 3.12 --managed-python`

---

## 3. Install PyTorch

Pick the block that matches your CUDA version:

### CUDA 12.4 / 13.x (driver 550+)
```bash
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
```

### CUDA 12.1
```bash
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

### CUDA 11.8
```bash
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118
```

Verify CUDA is available:
```bash
python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
# Expected: 2.6.0+cu124 | CUDA: True | GPU: NVIDIA GeForce RTX 4060 Ti
```

---

## 4. Install project dependencies

```bash
uv pip install -r requirements.txt
```

---

## 5. Install flash-attn

flash-attn requires a prebuilt wheel matching your exact CUDA + torch + Python combination.
Since `nvcc` is typically not available on Ubuntu 24.10 without the CUDA toolkit installed,
**do not use `--no-build-isolation`** — download the prebuilt wheel instead.

```bash
# Download the wheel for: cu124 + torch2.6.0 + Python 3.12 + Linux x86_64
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu124torch2.6.0cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

uv pip install flash_attn-2.7.3+cu124torch2.6.0cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

> Browse all available wheels at:
> https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.3

### Skip flash-attn entirely

If the prebuilt wheel is not available for your combination, just use `eager` attention
in the inference script (see next section). There is no functional difference, only a
minor speed impact on large images.

---

## 6. Transformers Inference

```python
from transformers import AutoModel, AutoTokenizer
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name = "deepseek-ai/DeepSeek-OCR-2"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation="flash_attention_2",  # use "eager" if flash-attn is not installed
    trust_remote_code=True,
    use_safetensors=True,
)
model = model.eval().cuda().to(torch.bfloat16)

prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = "your_image.jpg"
output_path = "./output"

os.makedirs(output_path, exist_ok=True)

res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    base_size=1024,
    image_size=768,
    crop_mode=True,
    save_results=True,
)
print(res)
```

Or use the provided script:
```bash
cd DeepSeek-OCR2-master/DeepSeek-OCR2-hf
python run_dpsk_ocr2.py
```

---

## 7. vLLM Inference (optional)

Edit `DeepSeek-OCR2-master/DeepSeek-OCR2-vllm/config.py` with your input/output paths first.

Install vLLM using the prebuilt wheel (ABI3 — compatible with Python 3.12):
```bash
# The cu118 wheel works fine with driver 580 (CUDA is backward compatible)
wget https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
uv pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
```

```bash
cd DeepSeek-OCR2-master/DeepSeek-OCR2-vllm

python run_dpsk_ocr2_image.py       # single image, streaming output
python run_dpsk_ocr2_pdf.py         # PDF, concurrent processing
python run_dpsk_ocr2_eval_batch.py  # batch eval (e.g. OmniDocBench v1.5)
```

> **Note:** you may see a warning about `vllm 0.8.5 requires transformers>=4.51.1`.
> This is non-blocking — ignore it if the script runs correctly.

---

## Quick install (copy-paste)

```bash
git clone https://github.com/IcelandicCenterArtificialIntelligence/DeepSeek-OCR-2.git
cd DeepSeek-OCR-2
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu124torch2.6.0cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
uv pip install flash_attn-2.7.3+cu124torch2.6.0cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Prompts

```
# Document to markdown:   <image>\n<|grounding|>Convert the document to markdown.
# Generic OCR:            <image>\n<|grounding|>OCR this image.
# No layout preservation: <image>\nFree OCR.
# Figures / charts:       <image>\nParse the figure.
# General description:    <image>\nDescribe this image in detail.
# Locate region:          <image>\nLocate <|ref|>text to find<|/ref|> in the image.
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` returns `False` | Wrong index used | Reinstall torch with the correct `--index-url` |
| flash-attn wheel not found | Wrong filename | Check the full release asset list at the GitHub link above |
| OOM on large images | Not enough VRAM | Reduce `image_size=512` or `base_size=768` in `model.infer()` |
| `ModuleNotFoundError` when loading model | `trust_remote_code` needs local files | Make sure you cloned the full repo |
| `libcuda.so` error with vLLM | Library path not set | `export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH` |
| Process already using GPU VRAM | Another Python process running | Check with `nvidia-smi` and kill it before loading the model |

---

## Acknowledgements

[DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR/), [Vary](https://github.com/Ucas-HaoranWei/Vary/), [GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/), [MinerU](https://github.com/opendatalab/MinerU), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [OmniDocBench](https://github.com/opendatalab/OmniDocBench).

## Citation

```bibtex
@article{wei2026deepseek,
  title={DeepSeek-OCR 2: Visual Causal Flow},
  author={Wei, Haoran and Sun, Yaofeng and Li, Yukun},
  journal={arXiv preprint arXiv:2601.20552},
  year={2026}
}
```
