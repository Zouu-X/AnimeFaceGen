## 1. Dependencies

- [x] 1.1 Add `realesrgan` and `basicsr` to `requirements.txt`

## 2. Upscaler Module

- [x] 2.1 Create `load_upscaler()` function to initialize `RealESRGANer` with `checkpoints/RealESRGAN_x4plus_anime_6B.pth`
- [x] 2.2 Create `upscale_batch()` function accepting numpy arrays (H, W, 3) RGB uint8, returning upscaled arrays

## 3. Pipeline Integration

- [x] 3.1 Initialize upscaler in `main()` after loading StyleGAN3 generator
- [x] 3.2 Add upscaling step after `generate_batch()`, before WebDataset write
- [x] 3.3 Update metadata JSON to include `upscaled: true` flag

## 4. Verification

- [x] 4.1 Run generation on CPU with small batch (`--num_images 4`) and verify 512×512 output
