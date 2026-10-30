# AIGC-5500 Midterm Project

| | |
| --- | --- |
| **Group** | MVPs |
| **Members** | Daniel Alejandro Castillo Martin (N10019284)<br>Eric Ly<br>Bryan Jacome<br>Nouroz Ali<br>Abhi Patel |
| **Course** | Advanced Deep Learning (AIGC-5500-0NA) |
| **Professor** | Hossein Pourmodheji |

## Project overview

Text classification project using the Enron spam dataset. The current implementation includes the initial LSTM notebook, environment configuration, reproducibility utilities, and dataset loading.

## Dataset

The project uses
[SetFit/enron_spam](https://huggingface.co/datasets/SetFit/enron_spam)
from Hugging Face.

| Source partition | Emails |
| --- | ---: |
| Train | 31,716 |
| Test | 2,000 |

The dataset includes email text and spam/non-spam labels. It downloads automatically into the ignored `data/hf_cache/` folder.

The original partitions are currently preserved. No preprocessing, validation splitting, or model training has been performed.

## Repository structure

```text
docs/                         Project instructions
notebooks/
    lstm_experiment.ipynb     Initial LSTM notebook
src/
    __init__.py               Python package initialization
    data.py                   Dataset loading
    reproducibility.py        Random seed utility
.gitignore                    Git exclusion rules
README.md                     Project documentation
requirements.txt              Frozen development environment
```

## Environment setup

Use Python 3.12. Run commands from the repository root.

### 1. Create and activate the environment

**Windows — PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS — Bash/Zsh**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install PyTorch

Choose one option for your computer.

**Windows or Linux — NVIDIA GPU**

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

**Windows or Linux — CPU only**

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
```

**macOS**

```bash
python -m pip install torch==2.9.1
```

### 3. Install supporting packages

```bash
python -m pip install numpy pandas scikit-learn matplotlib datasets ipykernel
```

`requirements.txt` records the original Windows/CUDA environment using `pip freeze`. Its cross-platform compatibility has not yet been verified. The commands above install the required packages but do not pin all supporting package versions.

## Run the notebook

Open `notebooks/lstm_experiment.ipynb` in VS Code or Cursor, select the `.venv` kernel, and run all cells from the top.

The notebook automatically selects CUDA when available and otherwise uses CPU. The current implementation does not select Apple's MPS backend.

The setup has been verified on Windows with Python 3.12.10, PyTorch 2.9.1+cu128, and an NVIDIA GeForce RTX 3060 Laptop GPU.

## Reproducibility configuration

The notebook currently defines:

| Configuration | Value |
| --- | --- |
| Split seed | 42 |
| Training seeds | 42, 123, 2026 |

The `set_seed()` function is defined in `src/reproducibility.py`. It controls Python, NumPy, and PyTorch random generators and configures
cuDNN for improved repeatability.

These seeds are configured; data splitting and training runs have not yet been implemented. Identical results across different hardware and software versions are not guaranteed.