# Tasks: Build Pipeline

## 1. Environment Setup

- [x] 1.1 Create `requirements.txt` with `torch`, `torchvision`, `webdataset`, `huggingface_hub`, `ninja`


## 2. Implementation

- [x] 2.1 Implement `generate_dataset.py` with StyleGAN3 loading
- [x] 2.2 Implement batched generation loop
- [x] 2.3 Implement WebDataset writing logic (sharded tar files)
- [x] 2.4 Add command-line arguments for flexibility

## 3. Pre-flight Checks (Local)

- [x] 3.1 Verify script syntax (`python -m py_compile generate_dataset.py`)
- [x] 3.2 Check `requirements.txt` completeness
