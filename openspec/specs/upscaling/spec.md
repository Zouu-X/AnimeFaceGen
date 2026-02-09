# Specs: Upscaling

## Requirements

### Requirement: Upscale Images 4x
The system SHALL upscale 128×128 images to 512×512 using Real-ESRGAN with the anime-optimized model.
**Reason:** Higher resolution images are required for downstream tasks and improved visual quality.

#### Scenario: Successful upscaling
- **WHEN** a 128×128 image is passed to the upscaler
- **THEN** system returns a 512×512 image
- **AND** anime art style details are preserved

### Requirement: Load Pre-trained Model
The system SHALL load the upscaler model from `checkpoints/RealESRGAN_x4plus_anime_6B.pth`.
**Reason:** Use the anime-optimized model for best quality on anime faces.

#### Scenario: Model loading
- **WHEN** the upscaler is initialized
- **THEN** the model is loaded from the checkpoints directory
- **AND** the model is ready for inference on the specified device (GPU/CPU)

### Requirement: Batch Processing Compatibility
The system SHALL accept numpy arrays (H, W, 3) in RGB uint8 format for upscaling.
**Reason:** To integrate seamlessly with the existing StyleGAN3 generation output format.

#### Scenario: Numpy array input
- **WHEN** a uint8 numpy array (128, 128, 3) is provided
- **THEN** system converts to BGR, upscales, and returns uint8 numpy array (512, 512, 3) in RGB
