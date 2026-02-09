## Why

StyleGAN3 generates 128×128 anime face images. For downstream tasks and higher visual quality, we need 512×512 output. Real-ESRGAN with the anime-optimized model (`RealESRGAN_x4plus_anime_6B.pth`) can upscale 4× while preserving anime art style details.

## What Changes

- **Add Real-ESRGAN upscaling step** after StyleGAN3 generation, before saving to WebDataset
- Integrate the pre-trained model from `checkpoints/RealESRGAN_x4plus_anime_6B.pth`
- Final output will be 512×512 PNG images in WebDataset format
- Update metadata to include upscaler info

## Capabilities

### New Capabilities
- `upscaling`: Real-ESRGAN 4× upscaling pipeline for anime images (128→512)

### Modified Capabilities
- `generation`: Update to integrate upscaling step before WebDataset save

## Impact

- **Modified file:** `generate_dataset.py` — add upscaler initialization and batch processing
- **New dependency:** `realesrgan` package in `requirements.txt`
- **Model file:** Uses existing `checkpoints/RealESRGAN_x4plus_anime_6B.pth`
- **Output:** Images will be 512×512 instead of 128×128 (increased storage ~16×)
