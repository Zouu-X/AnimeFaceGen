# Design: Build Pipeline

## Context
We need to generate a large dataset of synthetic anime faces (200,000 images) using a high-quality pre-trained model. We chose StyleGAN3 for its superior image quality and `webdataset` format for efficient storage and training pipeline integration.

## Goals / Non-Goals
**Goals:**
-   Generate 200,000 high-resolution (likely 512x512 depending on model) anime face images.
-   Use the pre-trained model `akiyamasho/stylegan3-anime-faces-generator` from Hugging Face.
-   Save outputs directly to WebDataset format (tar shards).
-   Ensure reproducibility via fixed random seeds.

**Non-Goals:**
-   Training or fine-tuning the model.
-   Building a user interface for generation.

## Decisions
-   **Execution Model**: A standalone Python script `generate_dataset.py`.
-   **Model Loading**: Use `huggingface_hub` to download the specific model pickle file.
-   **Generator**: Instantiate `stylegan3` generator from the pickle. We will need to ensure `stylegan3` source code is available (likely via cloning the official repo or `torch_utils` from it, as it relies on custom ops).
-   **Data Format**:
    -   Images stored as `.png` inside tar archives.
    -   Shards created every 1000 images (resulting in 200 shards).
    -   Metadata (seed, truncation psi) stored in accompanying `.json` files within the tar.
-   **Reproducibility**: Use seeds `0` to `199999`.

## Risks / Trade-offs
-   **Dependency Complexity**: StyleGAN3 requires custom CUDA kernels which can be finicky to compile.
    -   *Mitigation*: Use a standard environment with `ninja` installed. If compilation fails, fall back to slower pure PyTorch implementation if available (StyleGAN3 specific ops might require compilation).
-   **Disk Usage**: 200,000 images at 512x512 PNG ~100GB.
    -   *Mitigation*: Ensure sufficient disk space. WebDataset format avoids inode exhaustion.
