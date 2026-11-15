# Contributing to SiliconPilot AI

Thank you for your interest in contributing to **SiliconPilot AI**! We welcome contributions from Analog IC designers, EDA software engineers, and machine learning researchers.

---

## 1. Development Principles & Code of Ethics

* **Technical Transparency:** Never claim an unverified ML estimate is an "electrical signoff" or that an optical screenshot check constitutes "DRC/LVS verification."
* **Fail-Closed Security:** All new model integrations, external APIs, and file parsers must fail closed with actionable error messages.
* **Zero Secret Leaks:** Never commit API keys, passwords, proprietary foundry PDK files, or local path configurations.
* **Process Isolation:** Long-running or native C-API tasks (like SPICE runs) must remain isolated from the primary UI thread.

---

## 2. Setting Up Your Development Environment

1. Fork and clone the repository:
   ```powershell
   git clone https://github.com/aakashlamba2021-create/SiliconPilot-AI.git
   cd SiliconPilot-AI
   ```
2. Create and activate a Python 3.11 virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Configure your `NGSPICE_DLL` environment variable as described in [Installation Guide](docs/INSTALLATION_GUIDE.md).

---

## 3. Testing Requirements

Before submitting a Pull Request, run the complete automated test suite:
```powershell
pytest
```
* **Acceptance Criteria:** All unit tests must pass cleanly. Any new feature must be accompanied by comprehensive tests in `tests/`.

---

## 4. Code Standards & Style

* Follow **PEP 8** conventions with clean formatting.
* Use Python 3.10+ type annotations (`from __future__ import annotations`).
* Provide docstrings for all classes, methods, and public functions.
* Keep Streamlit UI logic decoupled from core calculation engines in `src/`.

---

## 5. Submitting Pull Requests

1. Create a feature branch:
   ```powershell
   git checkout -b feature/your-feature-name
   ```
2. Commit your changes with descriptive messages:
   ```powershell
   git commit -m "feat(simulation): add operating point saturation check"
   ```
3. Push to your fork and submit a Pull Request targeting the `main` branch.
4. Include a detailed description of changes, motivation, and verification steps in your PR description.
