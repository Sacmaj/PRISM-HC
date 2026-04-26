# PRISM-HC Local Setup

This repository contains both reference material and runnable project source. The `AI Papers/` directory holds the specification / research corpus (markdown, PDF, TeX, DOCX, MATLAB, and the original Gemini stubs) — treat it as build context. Project source lives at the repo root: `prism_hc/` (PRISM-HC-lite PyTorch prototype) and `rebus_synthesis/` (REBUS identification + supervisor-gain synthesis, numpy + optional cvxpy).

Canonical local path:

```powershell
C:\Users\jjor3\Dev\PRISM-HC
```

## Environment

Create and activate a local Python environment from the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Required runtime/tooling:

- Python 3.10+
- Git
- ripgrep (`rg`) for fast repository search
- Python packages in `requirements.txt`

MATLAB/Octave files are present, but MATLAB/Octave setup is not required for the current Codex-only workflow.

## Trust

Mark the moved repository as trusted for Git:

```powershell
git config --global --add safe.directory C:/Users/jjor3/Dev/PRISM-HC
```

Future Codex work should use this directory as the working root.

## Verify

Run the unit tests (single discover pass — `prism_hc/` and `rebus_synthesis/` are both importable packages at repo root):

```powershell
.\.venv\Scripts\python -m unittest discover -s . -p "test_*.py" -v
```

This picks up both the PRISM-HC-lite tests under `prism_hc\test_*.py` and the REBUS scaffold tests in `rebus_synthesis\test_identification.py`. Solver-dependent REBUS tests skip automatically when cvxpy is not installed.

Run the PRISM-HC-lite end-to-end demo (60 steps, lands one plasticity commit at t=30):

```powershell
.\.venv\Scripts\python prism_hc\demo.py
```

Run the REBUS lightweight smoke path:

```powershell
.\.venv\Scripts\python -m rebus_synthesis.demo --smoke-test
```

Run the full CVXPY-backed demo:

```powershell
.\.venv\Scripts\python -m rebus_synthesis.demo --T 40 --nx 2 --B 6 --block-len 8 --solver SCS
```

## Fast Navigation

Useful project-wide searches:

```powershell
rg -n "PRISM-HC|PGSTR|PGTR" "AI Papers"
rg -n "LATCH|safety-gated|anchor|safety spine|cross-modal" "AI Papers"
rg -n "reservoir|Count-Mixer|without gradients|routing|Hebbian|pathway" "AI Papers"
rg -n "CLF|CBF|Lyapunov|ISS|REBUS|topology|precision" "AI Papers"
rg -n -i "dmt|ketamine|cannabis|mdma|lsd|psilocybin|psychedelic|opioid|nicotine|methamphetamine|amphetamine|cocaine|alcohol|benzodiazepine|ecstasy|drug" "AI Papers"
rg -n -i "salience|precision|plasticity|entropy|attractor|gain|rebound|dissociation|reward|inhibitory|uncertainty|maladaptive" "AI Papers"
```
