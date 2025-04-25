# Rezbin AI Model v2.0

## Description

This repository will serve as the development platform of a multiclassification AI model to detect and classify multiple types of trash using the [TrashNet](https://www.kaggle.com/datasets/feyzazkefe/trashnet) dataset.

It references the [WasteNet](https://arxiv.org/pdf/2006.05873) paper as a guide to ensure cutting-edge performance and adhere with the best practices.

## Setup
Follow the steps below to properly set up our development environment.

```bash
# 1. Clone the repository
git clone https://github.com/Rezbin/ai-model-v2.0.git
cd ai-model-v2.0

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment (PowerShell)
.venv\Scripts\Activate.ps1

# 4. Install required packages
pip install -r requirements.txt

# 5. Register a new Jupyter kernel for this environment
python -m ipykernel install --user --name=.venv --display-name "Rezbin AI Model (v2.0)"
```

## Developers
Rezbin AI Model version 2.0 is developed by the AI interns team of Rezbin (April - June 2025).