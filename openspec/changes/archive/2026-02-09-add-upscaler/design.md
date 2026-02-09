## Context

The current `generate_dataset.py` generates 128×128 anime face images using StyleGAN3 and saves them in WebDataset format. For downstream tasks (e.g., MAE pre-training), higher resolution images (512×512) are required. A pre-trained Real-ESRGAN model (`RealESRGAN_x4plus_anime_6B.pth`) is already available in `checkpoints/`.

## Goals / Non-Goals

**Goals:**
- Integrate Real-ESRGAN 4× upscaling into the generation pipeline
- Upscale images from 128×128 to 512×512 before saving to WebDataset
- Maintain batch processing efficiency

**Non-Goals:**
- Training or fine-tuning the upscaler model
- Supporting variable upscale factors
- Upscaling existing datasets (only new generation)

## Decisions

### 1. Use `RealESRGANer` from the `realesrgan` package
**Rationale:** Official API with GPU support, tiling for memory efficiency, and FP16 option.
**Alternative considered:** Direct `basicsr` model loading — requires more boilerplate and manual preprocessing.

### 2. Process in batches using numpy array input
**Rationale:** `RealESRGANer.enhance()` accepts BGR numpy arrays, matching our existing `generate_batch()` output format (just need RGB→BGR conversion).
**Alternative considered:** Save to disk and re-load — adds I/O overhead.

### 3. Initialize upscaler once, reuse across batches
**Rationale:** Model loading is expensive; reusing avoids repeated initialization.

### 4. Use tiling with `tile=128` for memory safety
**Rationale:** 128×128 input is small, but tiling ensures we don't OOM on edge cases. Can be disabled if performance is a concern.

## Risks / Trade-offs

| Risk                                | Mitigation                                      |
| ----------------------------------- | ----------------------------------------------- |
| Upscaling adds ~0.5-1s per image    | Accept as necessary trade-off; use GPU and FP16 |
| Increased storage (16× more pixels) | Expected; document in proposal impact           |
| Dependency on `basicsr` (indirect)  | Required by `realesrgan`; add to requirements   |

## Implementation Approach

1. Add `realesrgan` and `basicsr` to `requirements.txt`
2. Create `load_upscaler()` function to initialize `RealESRGANer`
3. Modify `generate_batch()` or add `upscale_batch()` to process images
4. Insert upscaling step between generation and WebDataset write
5. Update metadata JSON to include `upscaled: true`
