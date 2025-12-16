# Bioreactor scaling up
Mechanistically Constrained Multi-Fidelity Learning for Reliable Extrapolation in Bioprocess Scale-Up
A Python project for bioreactor scale-up prediction using multi-fidelity learning: combining low-fidelity scaling laws / mechanistic proxies with higher-fidelity data (lab/pilot measurements) to improve accuracy across volumes and operating regimes.

## Intellectual Property / Registration

The work described in this manuscript is covered by a patent application currently under
review by the Intellectual Property Agency of the Republic of Uzbekistan (Ref: DT 202509951).

The workflow is organized as modular components for:

scaling feature/basis generation,

baseline ML models,

a hybrid multi-fidelity model,

visualization utilities,

a single runnable entry script.

Project Structure

bioreactor_core.py
Core utilities and shared logic (data handling, feature preparation, common helpers).

scaling_bases.py
Implements scaling “basis” functions / low-fidelity proxies (dimensionless groups, power laws, log terms, CFD-inspired surrogates, etc.) used as physics-informed features.

ml_baselines.py
Baseline machine learning models (e.g., linear/ridge/lasso, random forest, XGBoost, etc.) for comparison.

hybrid_model.py
The multi-fidelity hybrid model that combines low-fidelity basis predictions/features with data-driven learning (e.g., learned weighting, stacking, residual learning, or mixture model).

plotting.py
Plotting + evaluation visualization (parity plots, error vs scale, learning curves, residual diagnostics).

run_bioreactor_scaling.py
Main entry point to run training/evaluation end-to-end.

Installation
Option A — pip/venv (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

Option B — minimal requirements

If you don’t have requirements.txt yet, typical deps are:

numpy, pandas, scikit-learn

matplotlib

(optional) xgboost / lightgbm

Quick Start
1) Prepare data

Place your dataset (CSV recommended) in the project folder, e.g.:

bioreactor_scales.csv

Typical columns (example):

scale (or volume V)

time t

response C (or any target: titer, biomass, productivity, etc.)

optional operating variables (rpm, aeration, kLa, power input, geometry descriptors…)

If your current format differs, adjust the loader in bioreactor_core.py or run_bioreactor_scaling.py.

2) Run the pipeline
python run_bioreactor_scaling.py


If your script supports arguments, you can document them here, e.g.:

python run_bioreactor_scaling.py --data bioreactor_scales.csv --target C --model hybrid

What the Hybrid Multi-Fidelity Model Does (Concept)

The project uses multi-fidelity learning to improve scale-up prediction by leveraging:

Low-fidelity scaling bases (from scaling_bases.py) to capture known trends (e.g., power-law, log-law, dimensionless correlations).

Data-driven ML (from ml_baselines.py) to learn patterns and nonlinearities not captured by simple scaling rules.

A hybrid model (in hybrid_model.py) that merges them (common patterns include):

learning a weighted mixture of basis functions,

predicting a residual correction on top of low-fidelity estimates,

stacking/ensemble approaches.

This is helpful when you have:

many cheap/approximate “physics” features,

but limited expensive high-quality data across scales.

Outputs

Depending on your implementation, run_bioreactor_scaling.py typically produces:

metrics (RMSE/MAE/R²) overall and per scale,

saved plots (parity/residuals/error vs volume),

optional saved model artifacts.

If you already save to a folder, document it here, e.g.:

outputs/figures/

outputs/models/

outputs/metrics.json

## Authors

 - Babaa Moulay Rachid (correspodning author; For questions/collaboration mail to: [`m.babaa@newuu.uz `](m.babaa@newuu.uz ))

 - Maintained by: Atabaev Otabek
