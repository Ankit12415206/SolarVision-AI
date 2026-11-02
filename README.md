# SolarVision AI 🔆

**Automated Solar Panel Dust and Micro-Crack Defect Inspection from Drone Images**

A deep learning computer vision system for detecting and classifying defects in solar panels using drone-captured imagery.

## Overview

SolarVision AI applies state-of-the-art computer vision techniques — including image classification, object detection, and semantic segmentation — to automatically identify:

- **Dust accumulation** on solar panel surfaces
- **Micro-crack defects** in solar panel cells
- **Clean/healthy panels** as baseline reference

## Features (Planned)

- 🔍 Multi-task defect detection (classification + detection + segmentation)
- 🚁 Drone image processing pipeline
- 📊 Interactive Streamlit dashboard
- 📈 Real-time inspection reports (PDF export)
- 🗄️ SQLite-backed inspection history
- 🧪 Comprehensive test suite

## Project Structure

```
SolarVision-AI/
├── data/               # Datasets (not committed to git)
├── notebooks/          # Jupyter exploration & experiments
├── src/                # Core source code
│   ├── data/           # Dataset loading & preprocessing
│   ├── models/         # Classification, detection, segmentation
│   ├── training/       # Training loops & schedulers
│   ├── evaluation/     # Metrics & evaluation scripts
│   ├── inference/      # Inference pipelines
│   ├── utils/          # Shared utilities
│   └── database/       # SQLite inspection logging
├── app/                # Streamlit dashboard
├── reports/            # Generated figures & PDF reports
├── tests/              # Unit, integration tests
├── docs/               # Documentation
├── configs/            # YAML configuration files
├── scripts/            # Utility scripts (data prep, export)
├── checkpoints/        # Saved model weights (not committed)
└── logs/               # Training & application logs
```

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows

# 2. Install base dependencies (without PyTorch)
pip install -r requirements.txt

# 3. Install PyTorch (CPU-only — recommended for this machine)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Run tests
pytest

# 5. Launch dashboard (once built)
streamlit run app/main.py
```

## Documentation

- [Architecture](docs/architecture.md)
- [Dataset Research](docs/dataset_research.md)
- [Development Plan](docs/development_plan.md)
- [Technical Risks](docs/technical_risks.md)
- [Agent Guidelines](AGENTS.md)

## License

MIT License — see LICENSE file for details.
