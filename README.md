# AnimeFaceGen

A high-resolution anime face dataset generator powered by **StyleGAN3** and **Real-ESRGAN**. This tool synthesizes diverse 512×512 anime face images and packages them into efficient [WebDataset](https://github.com/webdataset/webdataset) shards, ready for large-scale deep learning workflows.

---

## ✨ Features

- **StyleGAN3 Generation** — Leverages a [pre-trained StyleGAN3 anime face model](https://huggingface.co/akiyamasho/stylegan3-anime-faces-generator) (auto-downloaded from Hugging Face Hub) to produce high-quality, diverse anime faces.
- **4× Super-Resolution** — Upscales native 128×128 outputs to **512×512** using the anime-optimised [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) model (`RealESRGAN_x4plus_anime_6B`).
- **WebDataset Output** — Saves images as `.tar` shards (default 1 000 images/shard) with paired PNG and JSON metadata, enabling fast sequential I/O for training pipelines.
- **Reproducible Seeds** — Each image is tied to a deterministic seed recorded in the metadata JSON, making any subset of the dataset fully reproducible.
- **Scalable** — Designed to generate **200 000+** images in a single run with configurable batch sizes and GPU/CPU fallback.

---

## 📂 Project Structure

```
AnimeFace/
├── generate_dataset.py      # Main generation script
├── requirements.txt         # Python dependencies
├── checkpoints/             # Pre-trained upscaler weights
│   └── RealESRGAN_x4plus_anime_6B.pth
└── stylegan3/               # (auto-cloned) NVIDIA StyleGAN3 repo
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU recommended (CPU fallback supported)

### Installation

```bash
# Clone the repository
git clone https://github.com/Zouu-X/AnimeFaceGen.git
cd AnimeFaceGen

# Install dependencies
pip install -r requirements.txt
```

> **Note:** The StyleGAN3 repository will be cloned automatically on first run. The pre-trained generator is downloaded from Hugging Face Hub automatically.

### Download Upscaler Weights

Place the Real-ESRGAN anime model in the `checkpoints/` directory:

```bash
mkdir -p checkpoints
# Download RealESRGAN_x4plus_anime_6B.pth into checkpoints/
```

---

## 🎨 Usage

```bash
python generate_dataset.py [OPTIONS]
```

### Options

| Flag               | Default     | Description                         |
| ------------------ | ----------- | ----------------------------------- |
| `--num_images`     | `200000`    | Total number of images to generate  |
| `--output_dir`     | `./dataset` | Output directory for `.tar` shards  |
| `--batch_size`     | `16`        | Images generated per batch          |
| `--truncation_psi` | `0.7`       | Truncation ψ (lower = less diverse) |
| `--seed_start`     | `0`         | Starting seed value                 |
| `--shard_size`     | `1000`      | Number of images per `.tar` shard   |
| `--device`         | `cuda`      | Device (`cuda` or `cpu`)            |

### Example

```bash
# Generate 10,000 images with higher diversity
python generate_dataset.py --num_images 10000 --truncation_psi 0.9

# Generate on CPU
python generate_dataset.py --num_images 100 --device cpu --batch_size 4
```

---

## 📦 Output Format

The dataset is written as WebDataset `.tar` shards:

```
dataset/
├── images-000000.tar
├── images-000001.tar
└── ...
```

Each shard contains paired files per sample:

| File          | Content                                                |
| ------------- | ------------------------------------------------------ |
| `000000.png`  | 512×512 RGB anime face image                           |
| `000000.json` | `{"seed": 0, "truncation_psi": 0.7, "upscaled": true}` |

### Loading with WebDataset

```python
import webdataset as wds

dataset = wds.WebDataset("dataset/images-{000000..000199}.tar")
    .decode("pil")
    .to_tuple("png", "json")

for image, metadata in dataset:
    # image: PIL Image (512×512)
    # metadata: dict with seed, truncation_psi, upscaled
    pass
```

---

## 🔧 Pipeline Overview

```
Seed → StyleGAN3 (128×128) → Real-ESRGAN 4× (512×512) → WebDataset .tar
```

1. **Seed-to-latent** — Each integer seed generates a deterministic latent vector `z`.
2. **StyleGAN3 inference** — The latent is passed through the pre-trained generator to produce a 128×128 anime face.
3. **Real-ESRGAN upscaling** — The image is upscaled 4× to 512×512 using the anime-optimised super-resolution model.
4. **WebDataset packaging** — The upscaled PNG and its metadata JSON are written to a `.tar` shard.

---

## 📄 License

This project uses third-party models and code:

- [StyleGAN3](https://github.com/NVlabs/stylegan3) — NVIDIA Source Code License
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — BSD 3-Clause License
- Pre-trained anime face model by [akiyamasho](https://huggingface.co/akiyamasho/stylegan3-anime-faces-generator)
