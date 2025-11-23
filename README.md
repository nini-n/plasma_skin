# Plasma Skin – Adaptive Control Benchmark

**Version 1 – Simulation Only**

This repository provides a virtual benchmark for adaptive plasma-skin control on an aircraft-like panel using DBD actuators. The environment exposes three continuous control inputs (voltage, frequency, duty cycle) and returns a surrogate EM metric, power consumption, and board temperature under explicit safety constraints.

The benchmark is used to compare
- a hand-tuned **Baseline** controller,
- a gradient-free **Extremum-Seeking (ES)** controller,
- and a simple policy-gradient **Reinforcement Learning (RL)** agent,

using KPIs such as EM visibility reduction (dB), EM-per-power efficiency, and safe-operation ratio.

The full project report is included as:

- `report_of_plasma_skin.pdf`  
- `report_of_plasma_skin.docx`

---

## Repository Structure

```text
plasma_skin/
  controllers/
    __init__.py
    baseline_controller.py      # Baseline controller
    extremum_seeking.py         # Extremum-seeking controller
    rl_policy_gradient.py       # Policy-gradient RL agent

  env/
    __init__.py
    plasma_env.py               # PlasmaEnv class and scenario modes (A/B/C)

  experiments/
    __init__.py
    eval_rl_policy.py               # Train and evaluate RL agent
    plot_rl_training_curves.py      # RL training curves
    plot_timeseries_baseline_es.py  # Time series for Baseline vs ES
    run_many_episodes_es.py         # Batch evaluation for Baseline + ES
    run_rl_episodes.py              # Multiple-episode runner for RL
    run_single_episode.py           # Single-episode runner (Baseline / RL)
    run_single_episode_es.py        # Single-episode runner for ES

  metrics/
    __init__.py
    kpi.py                      # KPI computation and summarization

  delta_em_bar.png
  eta_em_bar.png
  Figure_1.png
  Figure_1(1).png
  Figure_3.png
  Figure_4.png
  rl_training_curves_mode_A.png
  time_series_baseline_vs_es_mode_A.png
```

---

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

An example `requirements.txt`:

```text
numpy>=1.26
scipy>=1.11
pandas>=2.1
matplotlib>=3.8
gymnasium>=0.29
tqdm>=4.66
```

If you use classic `gym` instead of `gymnasium`, replace the `gymnasium` line with:

```text
gym>=0.26
```

---

## Usage

### Baseline and ES controllers

Run batch evaluations for the Baseline and ES controllers (Scenario A):

```bash
python -m experiments.run_many_episodes_es --mode A --num_episodes 20
```

This produces KPI summaries, including EM performance, power cost, EM-per-power efficiency, and safe-operation ratio.

### RL agent

Train and evaluate the policy-gradient RL agent:

```bash
python -m experiments.eval_rl_policy --mode A --num_episodes 100
```

This script trains the agent on the selected scenario and evaluates the learned policy on multiple episodes.

---

## Reproducing Figures

```bash
# Time series: Baseline vs ES (Scenario A)
python -m experiments.plot_timeseries_baseline_es

# RL training curves
python -m experiments.plot_rl_training_curves
```

The generated figures correspond to the PNG files in the root directory.

---

## Scenarios

`PlasmaEnv` provides three predefined scenarios:

- **A** – nominal conditions, low noise, slow EM drift  
- **B** – high measurement noise  
- **C** – faster EM drift  

Select the scenario via the `--mode` argument, e.g.

```bash
python -m experiments.run_many_episodes_es --mode B
python -m experiments.eval_rl_policy --mode C
```

---

## Versioning and Planned V2

This repository corresponds to **Version 1 (simulation only)**. The report outlines a planned **Version 2 (V2)** including:

- calibrated EM models against experimental or published DBD/RCS data,
- joint EM–aerodynamic optimization,
- advanced and safe RL algorithms (PPO, SAC, constrained RL, shields),
- digital-twin and MPC / model-based controllers,
- a small hardware DBD panel demonstrator.

---

## Citation

If you use this benchmark, please cite the project report, for example:

```bibtex
@misc{ozoglu2025plasmaskin,
  author       = {Nihan Ozoglu},
  title        = {Adaptive Plasma Skin Control in a Virtual Benchmark Environment: Extremum-Seeking vs. Reinforcement Learning},
  year         = {2025},
  note         = {Project Report -- Version 1 (Simulation Only)},
  howpublished = {GitHub repository},
}
```

---

## License

Add an open-source license of your choice (e.g., MIT / BSD-3-Clause / Apache-2.0) as `LICENSE` in the repository root.
