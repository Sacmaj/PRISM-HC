# AGENTS.md

## Working Root

Use `C:\Users\jjor3\Dev\PRISM-HC` as the canonical repository root. The old OneDrive clone path should not be treated as the active workspace.

This repository contains both reference material and runnable project source.

- **`AI Papers/`** is the specification / research corpus: markdown, PDF, TeX, DOCX, MATLAB, the REBUS identification scaffold (`rebus_identification.py` and friends), and the original Gemini stubs. Treat it as build context, not as a Python package directory.
- **`prism_hc/`** at the repo root is the runnable PRISM-HC-lite PyTorch prototype (PrismHCLite + LATCH supervisor + REBUS scalar state + frozen anchor core + null-space adapter).

## Commands

Set up the Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run tests and demos:

```powershell
.\.venv\Scripts\python -m unittest discover -s . -p "test_*.py" -v
.\.venv\Scripts\python -m unittest discover -s "AI Papers" -p "test_*.py" -v
.\.venv\Scripts\python prism_hc\demo.py
.\.venv\Scripts\python "AI Papers\rebus_identification_demo.py" --smoke-test
.\.venv\Scripts\python "AI Papers\rebus_identification_demo.py" --T 40 --nx 2 --B 6 --block-len 8 --solver SCS
```

The two `unittest discover` invocations are separate because `AI Papers` has a space in its name and is not an importable Python package, so a root-level discover does not recurse into it. The first picks up `prism_hc\test_*.py` (20 PRISM-HC-lite tests); the second picks up `AI Papers\test_rebus_identification.py` (6 REBUS scaffold tests).

## Mental Map

### First implementation target: PRISM-HC

Start from `AI Papers\architectural_computational_analysis.md` and `AI Papers\Assessment of Helix-FEP, PRISM-HC, and PGTR-LM.md`.

Core interpretation: PRISM-HC is the shortest path from the corpus to executable primitives. Treat it as a primary inference substrate with controlled plasticity, safety anchoring, reset discipline, and bounded adaptation.

Implementation target order:

1. Typed control-state and telemetry primitives.
2. Scalar REBUS/CLF-CBF supervisor.
3. Seeded reservoir or low-rank/hash feature expansion.
4. Safety anchor observability and drift checks.
5. PRISM-HC-lite integration.
6. LATCH safety-plasticity gate around commits/adaptation.

### Safety-plasticity controller: LATCH

Primary files:

- `AI Papers\# LATCH A Lexicographically-Anchored Two-State Controlled Hierarchy for Safety-Gated Plasticity in Language Models.md`
- `AI Papers\architectural_computational_analysis.md`

Core interpretation: LATCH is not a full model architecture. It is the governor for plasticity. Keep entropic openness `E` separate from safety state `S`, and require safety establishment before plasticity or adapter commits increase.

### Research architecture: HELIX-FEP

Primary files:

- `AI Papers\helix_fep_synthesis.md`
- `AI Papers\Helix-FEP and the Missing Candy and Hippie Flipping Logic.md`
- `AI Papers\# Helix-FEP and the Missing Candy and Hippie Flipping Logic.md`

Core interpretation: HELIX-FEP is a later research target with bounded excursion inference, vector plasticity, topology-as-belief, and spectral safety-code ideas. Do not implement it before PRISM-HC-lite primitives are stable.

### REBUS / CLF-CBF math and runnable prototype

Primary files:

- `AI Papers\rebus_identification.py`
- `AI Papers\test_rebus_identification.py`
- `AI Papers\rebus_identification_demo.py`
- `AI Papers\rebus_control_framework.tex`

Core interpretation: this is the current runnable scaffold. `rebus_identification.py` estimates robust one-sided safety constants and synthesizes supervisor gains. CVXPY is required for the full solver path; smoke tests can skip it.

### Reservoir, routing, and gradient-free ideas

Primary files:

- `AI Papers\Seeded Multilinear Reservoir with Count-Mixer Readout.docx`
- `AI Papers\Without gradients.docx`
- `AI Papers\architectural_computational_analysis.md`
- `AI Papers\Practical LLM dev design.md`

Core interpretation: reservoir/routing concepts are useful as research primitives and possible PRISM-HC components, but they should start as small unit-tested modules rather than replacements for attention.

### Safety spine and cross-modal anchors

Primary files:

- `AI Papers\Invariant Cross-Modal Safety Spine.docx`
- `AI Papers\Cross-Modal Alignment Anchors.docx`
- `AI Papers\Cross modal anchor alignment.docx`
- `AI Papers\Practical LLM dev design.md`

Core interpretation: use these ideas as observability, regression gates, protected probes, and rollback triggers. Do not treat hidden anchor geometry as a hard semantic safety proof.

### Formula-bearing and control-heavy files

Prioritize Markdown and TeX files for searchable formulas and implementation translation:

- `AI Papers\architectural_computational_analysis.md`
- `AI Papers\corrected_codebase_architecture_analysis.md`
- `AI Papers\rebus_control_framework.tex`
- `AI Papers\psilocybin_rebus_report.md`
- `AI Papers\*_research.md`
- `AI Papers\*_analysis.md`
- `AI Papers\*_Framework.md`

### Drug / Neuromodulation Research Files

These files are first-class research sources. They are not indexed for pharmacology as the end goal; they are indexed for reusable computational motifs: gain modulation, precision weighting, entropy windows, bounded plasticity, salience/reward attractors, inhibitory control, reset/rebound dynamics, uncertainty, and maladaptive routing.

Primary groups:

- Psychedelic / REBUS: `AI Papers\psilocybin_rebus_report.md`, `AI Papers\dmt_brain_dynamics_report.md`, `AI Papers\dmt_ultra_rapid_psychedelic_state.md`, `AI Papers\lsd_brain_effects_research.md`, `AI Papers\Psychedelics, MDMA, and Neuroplasticity Synergy.docx`, `AI Papers\Psychedelics, MDMA, and Neuroplasticity Synergy.pdf`, `AI Papers\Psychedelics and MDMA-Rebus perspective.docx`, `AI Papers\Psychedelics, Neuroplasticity, and LLM Design.docx`.
- Dissociation / plasticity: `AI Papers\ketamine_dissociation_plasticity.md`, plus the matching `.tex` and `.pdf`.
- Stimulant / salience / precision: `AI Papers\amphetamine_stimulant_precision_weighting.md`, `AI Papers\methamphetamine_neurotoxicity_analysis.md`, `AI Papers\cocaine_maladaptive_plasticity.md`, `AI Papers\cocaine_maladaptive_salience_reward_attractor.md`.
- Modulators / inhibitors: `AI Papers\Cannabis_Chemotype_Framework.md`, `AI Papers\Cannabis_Neurobiology_and_Labels.md`, `AI Papers\nicotine_cholinergic_modulation.md`, `AI Papers\benzodiazepines_anxiolytic_inhibitory_gain_modulators.md`, `AI Papers\alcohol_cognitive_disruptor.md`, `AI Papers\Opioid_Addiction_Dynamics_Analysis.md`.
- MDMA / ecstasy and uncertainty: `AI Papers\mdma_brain_effects_research.md`, `AI Papers\ecstasy_adulteration_uncertainty.md`, `AI Papers\Mechanistic Critique of the MDMA-Psychedelic Synergy Hypothesis_ Formal, Pharmacological, and Experimental.pdf`.
- Meta-analysis / inclusion rationale: `AI Papers\DRUGS_Codebase_Architectural_Analysis.md`, `AI Papers\A paper on why files on drugs aren't actually about drugs and why they should be included.docx`, `AI Papers\psychedelic files actually contribute.docx`.

Use these files when designing or evaluating control frameworks, plasticity budgets, routing states, precision controllers, reservoir/attractor behavior, safety gates, and reset discipline. Treat `Practical LLM dev design.md` as the cautionary lens against overclaiming direct biological transfer.

## Search Patterns

Use `rg` before manual browsing:

```powershell
rg -n "PRISM-HC|PGSTR|PGTR" "AI Papers"
rg -n "LATCH|safety-gated|plasticity gate|anchor|safety spine|cross-modal" "AI Papers"
rg -n "reservoir|Count-Mixer|without gradients|routing|Hebbian|pathway" "AI Papers"
rg -n "CLF|CBF|Lyapunov|ISS|REBUS|control|topology|precision|formula|equation" "AI Papers"
rg -n -i "dmt|ketamine|cannabis|mdma|lsd|psilocybin|psychedelic|opioid|nicotine|methamphetamine|amphetamine|cocaine|alcohol|benzodiazepine|ecstasy|drug" "AI Papers"
rg -n -i "salience|precision|plasticity|entropy|attractor|gain|rebound|dissociation|reward|inhibitory|uncertainty|maladaptive" "AI Papers"
```

## Development Guidance

- Preserve the distinction between specification documents and executable code.
- Prefer building small tested primitives before broad architecture integration.
- Keep generated caches and solver artifacts out of git.
- Use `rebus_identification.py` as the first executable reference for control/safety work.
- Treat `Practical LLM dev design.md` as a corrective engineering lens when architecture documents overclaim safety, invariance, or production readiness.
