#!/usr/bin/env python3
"""Generate anime face images using a pre-trained StyleGAN3 model and save them in WebDataset format."""

import argparse
import io
import json
import os
import pickle
import subprocess
import sys

import numpy as np
import torch
import webdataset as wds
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm


def setup_stylegan3():
    """Clone the official StyleGAN3 repo if not present and add to sys.path."""
    repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stylegan3")
    if not os.path.isdir(repo_dir):
        subprocess.check_call(
            ["git", "clone", "https://github.com/NVlabs/stylegan3.git", repo_dir]
        )
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)


def download_model():
    """Download the pre-trained model from Hugging Face Hub."""
    return hf_hub_download(
        repo_id="akiyamasho/stylegan3-anime-faces-generator",
        filename="stylegan3-anime-face-gen.pkl",
    )


def load_generator(model_path, device):
    """Load the StyleGAN3 generator network from a pickle file."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    G = data["G_ema"].to(device)
    G.eval()
    return G


def generate_batch(G, seeds, truncation_psi, device):
    """Generate a batch of images from the given seeds.

    Returns a list of uint8 numpy arrays (H, W, 3).
    """
    z_list = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        z_list.append(rng.randn(1, G.z_dim))
    z = np.concatenate(z_list, axis=0)
    z = torch.from_numpy(z).float().to(device)

    label = torch.zeros([len(seeds), 0], device=device)

    with torch.no_grad():
        imgs = G(z, label, truncation_psi=truncation_psi, noise_mode="const")

    # Convert from [-1, 1] float to [0, 255] uint8
    imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return imgs.cpu().numpy()


def image_to_png_bytes(img_array):
    """Convert a uint8 numpy array to PNG bytes."""
    buf = io.BytesIO()
    Image.fromarray(img_array).save(buf, format="PNG")
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(
        description="Generate anime faces with StyleGAN3 and save as WebDataset."
    )
    parser.add_argument(
        "--num_images", type=int, default=200000, help="Number of images to generate"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./dataset", help="Output directory"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for generation"
    )
    parser.add_argument(
        "--truncation_psi",
        type=float,
        default=0.7,
        help="Truncation psi for generation",
    )
    parser.add_argument(
        "--seed_start", type=int, default=0, help="Starting seed value"
    )
    parser.add_argument(
        "--shard_size", type=int, default=1000, help="Number of images per shard"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    # Setup StyleGAN3 repo
    setup_stylegan3()

    # Download and load model
    print("Downloading model...")
    model_path = download_model()
    print("Loading generator...")
    G = load_generator(model_path, device)

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    pattern = os.path.join(args.output_dir, "images-%06d.tar")

    # Generate images and write to WebDataset shards
    seeds = list(range(args.seed_start, args.seed_start + args.num_images))
    idx = 0

    with wds.ShardWriter(pattern, maxcount=args.shard_size) as sink:
        for batch_start in tqdm(
            range(0, len(seeds), args.batch_size),
            desc="Generating images",
            total=(len(seeds) + args.batch_size - 1) // args.batch_size,
        ):
            batch_seeds = seeds[batch_start : batch_start + args.batch_size]
            images = generate_batch(G, batch_seeds, args.truncation_psi, device)

            for i, img in enumerate(images):
                sample = {
                    "__key__": f"{idx:06d}",
                    "png": image_to_png_bytes(img),
                    "json": json.dumps(
                        {
                            "seed": batch_seeds[i],
                            "truncation_psi": args.truncation_psi,
                        }
                    ).encode("utf-8"),
                }
                sink.write(sample)
                idx += 1

    print(f"Done. Generated {idx} images in {args.output_dir}")


if __name__ == "__main__":
    main()
