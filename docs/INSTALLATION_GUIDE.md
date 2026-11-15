# SiliconPilot AI — Windows Installation & Setup Guide

This guide walks you through setting up **SiliconPilot AI** on Windows 10 or 11 with full SPICE simulation capabilities and optional multimodal vision support.

---

## 1. System Prerequisites

* **Operating System:** Windows 10 / 11 (64-bit).
* **Python Runtime:** Python 3.11.x (tested on 3.11.9).
* **SPICE Engine:** KiCad 8.0, 9.0, or 10.0 (provides `ngspice.dll`), or standalone ngspice-shared.
* **Hardware:** 8 GB RAM minimum; 16 GB recommended.

---

## 2. Step-by-Step Installation

### Step 2.1: Clone the Repository
Open PowerShell and clone the repository:
```powershell
git clone https://github.com/aakashlamba2021-create/SiliconPilot-AI.git
cd SiliconPilot-AI
```

### Step 2.2: Create and Activate Virtual Environment
Create a dedicated Python 3.11 virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell blocks script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### Step 2.3: Upgrade pip and Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Configuring the ngspice Backend

SiliconPilot AI uses the official `ngspice.dll` shared library via `ctypes` for high-speed simulation.

### Option A: Via KiCad (Recommended)
If you have KiCad installed, the DLL is already on your system at one of these locations:
* KiCad 10: `C:\Users\<YourUsername>\AppData\Local\Programs\KiCad\10.0\bin\ngspice.dll`
* KiCad 9: `C:\Program Files\KiCad\9.0\bin\ngspice.dll`
* KiCad 8: `C:\Program Files\KiCad\8.0\bin\ngspice.dll`

Set the environment variable in your terminal session or `.env` file:
```powershell
$env:NGSPICE_DLL = "C:\Program Files\KiCad\9.0\bin\ngspice.dll"
$env:PATH = "C:\Program Files\KiCad\9.0\bin;" + $env:PATH
```

### Option B: Standalone ngspice
Download the official 64-bit Windows release from [ngspice SourceForge](https://sourceforge.net/projects/ngspice/files/), extract it, and point `NGSPICE_DLL` to the extracted `bin/ngspice.dll`.

---

## 4. Configuring Multimodal AI Providers (Optional)

The core application, SPICE simulation studio, and text assistant work **100% locally and offline without external API keys**.

To enable image analysis of analog IC layout screenshots:

1. Copy the secrets template:
   ```powershell
   Copy-Item .streamlit/secrets.example.toml .streamlit/secrets.toml
   ```
2. Open `.streamlit/secrets.toml` in your text editor and provide your keys:
   ```toml
   GEMINI_API_KEY = "AIzaSy..."
   OPENAI_API_KEY = "sk-..."
   ```

*Note: `.streamlit/secrets.toml` is permanently ignored by `.gitignore` and will never be committed.*

---

## 5. Verifying the Installation (Running Tests)

Run the automated unit and integration tests to verify everything is functioning correctly:
```powershell
pytest
```
Expected output: **105 passed, 2 skipped, 0 failed**.

---

## 6. Launching the Application

### Method 1: Using the Automated Windows Launcher (Recommended)
```powershell
./scripts/run_local.ps1
```
The launcher sets up short scratch paths, checks port availability, and launches the browser at `http://127.0.0.1:8502`.

### Method 2: Standard Streamlit Command
```powershell
streamlit run app.py
```
Open your web browser and navigate to `http://localhost:8501`.

---

## 7. Troubleshooting & Common Pitfalls

| Issue | Cause | Resolution |
| :--- | :--- | :--- |
| `ngspice.dll not found` | Environment variable not pointing to valid DLL. | Set `$env:NGSPICE_DLL` to the absolute path of `ngspice.dll` and ensure its directory is on `$env:PATH`. |
| `Spaces in path ngspice error` | ngspice C-API has issues with paths containing spaces (e.g. OneDrive). | Run via `./scripts/run_local.ps1` which automatically converts temporary working paths to Windows 8.3 short format. |
| `Script execution disabled` | Windows PowerShell execution policy. | Run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `ModuleNotFoundError: No module named 'src'` | Running scripts outside the project root directory. | Always run commands from the project root (`SiliconPilot-AI`). |
