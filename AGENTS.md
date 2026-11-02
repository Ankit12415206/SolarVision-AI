# AGENTS.md — SolarVision AI Development Guidelines

> This file defines the rules, conventions, and development guidelines for all
> human developers and AI agents working on **SolarVision AI**.
> Read this file before making any changes to the codebase.

---

## 1. Project Identity

| Field        | Value                                                           |
|--------------|-----------------------------------------------------------------|
| **Name**     | SolarVision AI                                                  |
| **Purpose**  | Automated defect inspection of solar panels from drone imagery  |
| **Stack**    | Python 3.11+, PyTorch, OpenCV, Streamlit, SQLite, Plotly        |
| **Stage**    | Step 8 — Unified Inference Integration Complete                  |

---

## 2. Step-Gate Workflow

The project is divided into numbered steps. **No step may begin without explicit
user approval of the previous step's completion report.**

| Step | Title                         | Status          |
|------|-------------------------------|-----------------|
| 1    | Project Setup                 | ✅ Complete      |
| 2    | Dataset Research & Acquisition| ✅ Complete      |
| 3    | Data Preprocessing Pipeline   | ✅ Complete      |
| 4    | Model Architecture Design     | ✅ Complete      |
| 5    | Classification Module         | ✅ Complete      |
| 6    | Object Detection Module       | ✅ Complete      |
| 7    | Segmentation Module           | ✅ Complete      |
| 8    | Unified Inference Integration | ⏳ Awaiting approval |
| 9    | Streamlit Dashboard           | 🔒 Locked        |
| 10   | Reports, Tests, Deployment    | 🔒 Locked        |

---

## 3. Coding Conventions

### 3.1 Style
- **Formatter:** `black` (line length = 88)
- **Import sorter:** `isort` with `black` profile
- **Linter:** `flake8`
- **Type checker:** `mypy` with `ignore_missing_imports = true`
- All public functions **must** have type hints and docstrings (Google style).

### 3.2 Naming
| Entity          | Convention          | Example                     |
|-----------------|---------------------|-----------------------------|
| Modules/files   | `snake_case`        | `data_loader.py`            |
| Classes         | `PascalCase`        | `SolarDataset`              |
| Functions       | `snake_case`        | `load_image()`              |
| Constants       | `UPPER_SNAKE_CASE`  | `IMAGE_SIZE = 224`          |
| Config keys     | `snake_case`        | `batch_size: 32`            |

### 3.3 File Headers
Every Python source file must start with:
```python
"""
Module: <module_name>
Description: <brief description>
Author: SolarVision AI Team
Created: <date>
"""
```

---

## 4. Project Structure Rules

- `src/` — All core library code. No scripts, no notebooks here.
- `notebooks/` — Exploration only. No production logic.
- `scripts/` — One-off CLI scripts. Must use `click` for argument parsing.
- `app/` — Streamlit UI only. Must import from `src/`, never implement logic inline.
- `tests/` — All test files. Mirror `src/` structure (`tests/unit/`, `tests/integration/`).
- `configs/` — YAML files only. No hardcoded hyperparameters in source code.
- `data/` — **Never commit raw or processed data to git.**
- `checkpoints/` — **Never commit model weights to git.**
- `logs/` — **Never commit logs to git.**

---

## 5. Configuration Management

- All hyperparameters go into YAML files under `configs/`.
- Load configs with `pyyaml`. Never hardcode values in training scripts.
- Sensitive values (API keys, paths) go in `.env` (never committed).
- Use `python-dotenv` to load `.env` at runtime.

---

## 6. Model Development Rules

- Each task (classification, detection, segmentation) lives in its own subdirectory under `src/models/`.
- Every model class must inherit from `torch.nn.Module`.
- Models must be saveable/loadable via `torch.save` / `torch.load`.
- No CUDA-specific code without a CPU fallback:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```
- Batch size, learning rate, and epochs are always config-driven.

---

## 7. Dataset Rules

- Raw data is **read-only**. Never modify files in `data/raw/`.
- All preprocessing outputs go to `data/processed/`.
- Augmented data goes to `data/augmented/`.
- Dataset splits (train/val/test) are defined in config YAML, not hardcoded.
- Reproducibility: always set and log a random seed.

---

## 8. Testing Requirements

- Minimum **80% test coverage** for `src/`.
- Every new module must have a corresponding unit test file.
- Tests must pass before any pull request / step approval.
- Run tests: `pytest` (configured in `pyproject.toml`).
- Use `pytest-mock` for mocking external dependencies (filesystem, models).

---

## 9. Git Workflow

- **Main branch:** `main` — production-ready, step-approved code only.
- **Feature branches:** `feature/<step>-<short-description>`.
- **Commit format:** `[StepN] <type>: <short description>` (e.g., `[Step1] feat: add project structure`).
- Commit types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`.
- Never force-push to `main`.
- Never commit: `.venv/`, `data/`, `checkpoints/`, `logs/`, `.env`.

---

## 10. AI Agent Rules

When an AI agent (e.g., Antigravity / Gemini) is working on this project:

1. **Read this file first** before any code action.
2. **Do not skip steps** — respect the step-gate table above.
3. **Do not install heavy packages** (torch, torchvision) without hardware inspection.
4. **Do not overwrite existing files** without reading them first.
5. **Do not delete files or directories** without explicit user approval.
6. **Do not hardcode paths** — use `pathlib.Path` and config files.
7. **Always report** hardware specs, created files, and errors at step completion.
8. **Prefer CPU-safe code** — never assume CUDA availability.
9. **Ask before proceeding** to the next step.
10. **Document decisions** — update `docs/` when architecture choices are made.

---

## 11. Dependency Policy

- Pin all dependencies in `requirements.txt` with `>=` lower bounds.
- Separate concerns: core deps vs dev deps (see `pyproject.toml`).
- PyTorch must be installed with the correct wheel for the target hardware:
  - **CPU-only:** `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
  - **CUDA 12.1:** `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
- Never `pip install torch` without specifying an index URL.

---

## 12. Contact / Escalation

If uncertain about a design decision, a data source, or a hardware constraint:
1. Document the uncertainty in `docs/technical_risks.md`.
2. Ask the user before proceeding.
3. Default to the simpler, CPU-compatible, beginner-friendly approach.
