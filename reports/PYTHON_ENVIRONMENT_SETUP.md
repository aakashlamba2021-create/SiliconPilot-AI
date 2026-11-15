# SiliconPilot AI — Python Environment Setup Plan

**Prepared:** 2026-11-12  
**Scope:** Read-only environment/dependency/test-safety review and recommended manual setup commands. No software was installed, no settings or project files other than this report were changed, and no tests were run.

## Summary

Python 3.11 is a suitable target for this project, but the existing user Python directories previously found on PATH do not contain a `python.exe`. The KiCad-bundled Python 3.11.5 runs, but it lacks most project dependencies. KiCad 9 does provide a loadable ngspice DLL at `C:\Program Files\KiCad\9.0\bin\ngspice.dll`; the project runner already searches that exact path. Use a dedicated `.venv` in the project after installing a complete Python 3.11 distribution. The test suite requires `pytest`, which is absent from `requirements.txt`.

## Declared dependencies and Python 3.11 fit

`requirements.txt` declares minimum versions, not exact pins:

| Dependency | Declared constraint | Role |
|---|---:|---|
| numpy | `>=2.0.0` | numerical computing |
| pandas | `>=2.2.0` | tabular data |
| scipy | `>=1.14.0` | scientific computing |
| scikit-learn | `>=1.5.0` | ML models |
| joblib | `>=1.4.0` | saved model loading |
| pyarrow | `>=17.0.0` | Parquet data |
| streamlit | `>=1.35.0` | dashboard |
| plotly | `>=5.22.0` | visualizations |
| matplotlib | `>=3.9.0` | plotting |
| seaborn | `>=0.13.0` | plotting |
| requests | `>=2.32.0` | HTTP utilities |
| click | `>=8.1.0` | CLI utilities |

The file states Python `3.10+` and says it was tested on `3.15.2`. The minimum versions listed are broadly compatible with Python 3.11, and a current pip resolver should select releases whose `Requires-Python` metadata permits 3.11. There are no explicit contradictory version constraints. **However, the open-ended constraints do not guarantee a reproducible or mutually tested set.** As of this report date, newest releases can raise their minimum Python version; for example, PyPI lists NumPy 2.5.4 as requiring Python >=3.12, while NumPy 2.3.5 requires >=3.11. On Python 3.11 pip should choose a compatible release, but that means it may not install the latest release. Current PyPI metadata lists Python 3.11 Windows wheels for pandas and scikit-learn. [NumPy release metadata](https://pypi.org/project/numpy/), [NumPy 2.3.5 metadata](https://pypi.org/project/numpy/2.3.5/), [pandas metadata](https://pypi.org/project/pandas/), [scikit-learn metadata](https://pypi.org/project/scikit-learn/), [Streamlit metadata](https://pypi.org/project/streamlit/)

### Missing or notable dependencies

- `pytest` is **not declared**, but `tests/test_pipeline.py` and `tests/test_phase3_optimizer.py` expose plain `test_*` functions. `unittest discover` will not collect those functions; pytest is the straightforward way to run all 23 test cases together. Add pytest to the dedicated environment only (command below); do not edit requirements for this setup plan.
- `colorlog` is an optional import with a standard logging fallback, so it is not required.
- scikit-learn depends on packages such as `threadpoolctl`; pip should install transitive dependencies automatically.
- Saved `.joblib` models may be sensitive to scikit-learn and NumPy versions used when they were serialized. `requirements.txt` has no upper bounds or lockfile, and the model manifest does not specify training runtime versions. Loading the committed model artifacts is therefore a compatibility point to verify; it is separate from whether Python 3.11 can install the declared packages.
- The phase test suite also assumes a functioning ngspice DLL, so the Phase 4 integration cases are not purely unit tests.

## Existing machine paths checked

- The earlier PATH contains `C:\Users\SHREYA\AppData\Local\Programs\Python\Python313`, but that directory had no interpreter executable. Python311 similarly had Scripts/pip launchers but no `python.exe` in the inspected installation directory.
- KiCad Python runs from `C:\Program Files\KiCad\9.0\bin\python.exe` (Python 3.11.5), but only NumPy among the checked project packages was importable; pandas, Streamlit, scikit-learn, joblib and pyarrow were not.
- `C:\Program Files\KiCad\9.0\bin\ngspice.dll` exists and was loadable with `ngSpice_Init` and `ngSpice_Command` exports.
- No `.venv` exists at the project root at review time.

## Test side effects and data safety

The suite has 23 cases across 7 pipeline functions, 6 Phase 3 functions, and 10 Phase 4 unittest methods.

- `tests/test_pipeline.py` imports `config.settings`; that module calls `mkdir(..., exist_ok=True)` for data/model/report/log directories. If those directories are missing, it creates them. Imported project loggers create/write `logs/siliconpilot.log`; its rotating handler can rotate existing logs when size limits are reached. The pipeline tests otherwise mostly use in-memory objects and read saved model artifacts.
- `tests/test_phase3_optimizer.py` primarily uses in-memory data and reads committed model files. No direct project-data deletion is present in its test bodies.
- `tests/test_phase4_simulation.py` creates a SQLite file at `tests/test_sim_results.db` during class setup and unconditionally removes that path in class teardown. **It would delete a pre-existing file of that exact name.** At inspection time that path did not exist. The test writes AC/transient output files into the OS temp directory; the runner removes its temporary `.cir` netlists but does not clean up the generated AC/transient files. The test's DB is separate from `data/simulation_results.db`.
- Running Python without `-B` can create `__pycache__` bytecode files in the project. Pytest normally creates `.pytest_cache` unless disabled.

For a cautious run, keep the test DB path absent, set `PYTHONDONTWRITEBYTECODE=1`, and disable pytest's cache provider. Expect project log writes/possible log rotation and ngspice scratch files in `%TEMP%`; those effects are not prevented by the proposed commands. The existing `data/simulation_results.db` was observed but the tests shown do not target it.

## ngspice DLL loading

`src/simulation/ngspice_runner.py` reads `NGSPICE_DLL` first, then tries a built-in Windows search list that includes the exact KiCad 9 path `C:\Program Files\KiCad\9.0\bin\ngspice.dll`. It loads the file with `ctypes.CDLL` and checks for `ngSpice_Init` and `ngSpice_Command`. Therefore the existing DLL path is compatible with the runner's discovery mechanism.

Setting `NGSPICE_DLL` explicitly makes discovery deterministic. On Windows, loading a DLL can also depend on adjacent native DLL dependencies. If the runner still reports it cannot load the DLL, use a **process-local** PATH addition for KiCad's `bin` directory in the test PowerShell session (command below); this does not edit system or user PATH. The runner does not explicitly call `os.add_dll_directory`.

## Recommended manual commands

These are recommendations only; they have not been executed. Run in PowerShell. The Python installation command uses WinGet and may require accepting its prompts. It does not require changing PATH manually. If WinGet is unavailable, install CPython 3.11 using the official Python for Windows installer and then use the interpreter path shown by that installer.

### 1. Install Python 3.11 if a complete interpreter is still absent

```powershell
winget install --id Python.Python.3.11 --exact --source winget
```

Open a new PowerShell session after installation, then verify the interpreter:

```powershell
$py311 = Join-Path $env:LocalAppData 'Programs\Python\Python311\python.exe'
& $py311 --version
```

If WinGet installed Python elsewhere, replace `$py311` with that installation's full `python.exe` path. Do not use the KiCad Python for the venv: it is application-bundled and lacks the project dependencies.

### 2. Create the dedicated project environment

```powershell
Set-Location 'C:\Users\SHREYA\OneDrive\Desktop\samsung project\SiliconPilot-AI'
& $py311 -m venv .venv
```

This writes the virtual environment under the project directory and does not activate or alter PATH globally.

### 3. Install project and test dependencies into that environment

```powershell
$venvPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt pytest
```

The extra `pytest` argument is for running the plain-function tests; it does not modify `requirements.txt`.

### 4. Configure ngspice for this PowerShell session and run all tests

```powershell
$env:NGSPICE_DLL = 'C:\Program Files\KiCad\9.0\bin\ngspice.dll'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PATH = 'C:\Program Files\KiCad\9.0\bin;' + $env:PATH
if (Test-Path -LiteralPath 'tests\test_sim_results.db') { throw 'Refusing to run: tests\test_sim_results.db already exists and Phase 4 teardown deletes it.' }
& $venvPython -m pytest -p no:cacheprovider tests -v
```

The PATH change above is limited to the current PowerShell process and helps Windows resolve any DLL dependencies beside ngspice. If direct DLL loading works without it, the assignment may be omitted. The guard protects against the Phase 4 teardown deleting an existing test DB. Pytest should collect the 7 + 6 function tests and 10 Phase 4 unittest methods. **No tests have been run as part of preparing this setup plan.**

### Optional installation validation before tests

```powershell
& $venvPython -m pip check
& $venvPython -c "import numpy, pandas, scipy, sklearn, joblib, pyarrow, streamlit, plotly, matplotlib, seaborn, requests, click; print('Imports OK')"
& $venvPython -c "from src.simulation.ngspice_runner import NGSpiceRunner; r=NGSpiceRunner(); print(r.get_backend_info()); assert r.is_available()"
```

Run the validation commands from the project root after setting `NGSPICE_DLL` as above. These commands are also recommendations, not checks performed for this report.
