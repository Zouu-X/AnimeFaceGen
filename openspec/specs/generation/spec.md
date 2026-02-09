# Specs: Generation

## Requirements

### Requirement: Generate 200k Images
The system SHALL generate widely diverse anime face images using the pre-trained StyleGAN3 model.
**Reason:** To build a large dataset for downstream tasks.

#### Scenario: Successful batch generation
- **WHEN** user runs the generation script with `--num_images 200000`
- **THEN** system generates images and saves them into WebDataset shards
- **AND** the processes completes without error

### Requirement: WebDataset Output
The system SHALL save generated images in WebDataset format (tar files).
**Reason:** To improve I/O performance and manage large numbers of files efficiency.
**Details:**
- Each shard should contain 1000 images.
- Images should be PNG format.
- Metadata (seeds) should be JSON.

#### Scenario: Shard creation
- **WHEN** generation crosses a 1000-image boundary
- **THEN** a new tar file is created (e.g. `images-000000.tar`, `images-000001.tar`)
- **AND** each tar contains paired `.png` and `.json` files
