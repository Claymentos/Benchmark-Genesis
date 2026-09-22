# Manual Installation

If some packages fail to install using the standard methods due to network issues or other reasons, this guide provides detailed instructions for manual installation.

## Setting Up Conda Environment and CUDA 12.1

```bash
# Create a new conda environment
conda create -n fastsam3d python=3.11

# Activate the environment
conda activate fastsam3d

# Install dependencies from YAML file
conda env update -f environments/default.yml
```

```bash
# Configure PyPI indexes for NVIDIA and PyTorch CUDA packages
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
```

## Installing requirements.txt and dev.txt

```bash
# Install the package in development mode with dev dependencies
pip install -e '.[dev]'
```

## Installing PyTorch for Python 3.11 and CUDA 12.1

```bash
# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Manually Installing utils3d

Download utils3d from GitHub to your local machine:

```bash
# Clone the utils3d repository
git clone https://github.com/EasternJournalist/utils3d.git

# Navigate to the repository
cd utils3d

# Checkout the specific commit
git checkout 3913c65d81e05e47b9f367250cf8c0f7462a0900
```

Transfer the utils3d directory to the server at `MoGe/utils3d` and install manually:

```bash
# Navigate to utils3d directory
cd utils3d

# Install the package (NOT in editable mode)
pip install .
```

> **Important**: Do NOT use `pip install . -e` (editable mode)

## Manually Installing MoGe

MoGe depends on utils3d and is not available on PyPI, so it must be downloaded from Git.

```bash
# 1. Comment out the utils3d Git dependency in MoGe/requirements.txt:
# Original line: git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900
# Change to: # git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900
```

```bash
# 2. In pyproject.toml, change:
# "EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900"
# To:
# "utils3d"
# (since we've already installed utils3d)
```

```bash
# Navigate to MoGe directory and install
cd MoGe
pip install .
```

## Manually Installing gsplat

First, install the required build tools to avoid `no-build-isolation` errors:

```bash
pip install hatchling hatch-requirements-txt
```

```bash
# Navigate to gsplat directory and compile
cd gsplat
MAX_JOBS=4 pip install . --no-build-isolation -v
```

> **Successful compilation output**:
> Successfully installed gsplat-1.5.3 jaxtyping-0.3.3 wadler-lindig-0.1.7

## Manually Installing pytorch3d

First, check if these dependencies are already installed:

```bash
pip show fvcore
pip show iopath
```

```bash
# Navigate to pytorch3d directory and compile
cd pytorch3d
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.0"
MAX_JOBS=4 pip install . --no-build-isolation -v
```

> **Successfully installed**: pytorch3d-0.7.8

## Installing inference.txt Dependencies

Install these manually (avoid `pip install -e '.[inference]'` as it reinstalls requirements.txt):

```bash
# Configure Kaolin download link
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
```

```bash
# Install inference dependencies
pip install kaolin==0.17.0
conda install -c conda-forge "numpy>=2,<2.3.0"
pip install seaborn==0.13.2
pip install gradio==5.49.0
```

## Compiling flash-attention

Installing `[p3d]` would reinstall requirements.txt, so we install flash-attention manually.

> **Note**: This compilation is time-consuming (approximately 1 hour).

Default installation method:
```bash
pip install -e '.[p3d]'
```

Alternative (recommended):
```bash
MAX_JOBS=4 pip install flash_attn==2.8.3 --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Reinstalling requirement.txt (if some packages failed)

If some packages from requirements.txt were not installed successfully:

```bash
pip install -r requirement.txt
```
