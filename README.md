# Multi-algorithm SCAPS-1D optimization of FTO/WO3/MASnIBr2/CBTS/Au solar cells

Code, data and trained models for the article

> K. I. Atol, I. Mamoon, F. H. Al-Atawi, A. Irfan, M. F. Rahman, M. D. Haque,
> *Multi Algorithm Bayesian Optimization and Machine Learning Surrogate Design of Lead-Free
> FTO/WO3/MASnIBr2/CBTS/Au, Tin-perovskite Solar Cells*,
> Sustainable Energy Technologies and Assessments (under review).

The notebook reproduces every table and figure of the article from the saved results, step by step.

## Contents

```
notebook/MASnIBr2_SCAPS_optimization.ipynb   full workflow, Parts 0-9, with outputs
src/scaps_tools.py                           helper code (SCAPS-1D interface, physics precheck, PPSR, TPE, GA, Monte Carlo)
scaps_files/                                 SCAPS-1D device files
    template_working.scaps                   initial device (Table 1, PCE 17.80%)
    baseline_best_seed789.scaps              final device (TPE, seed 789, PCE 25.27%)
    best_device_seed*.scaps                  best PPSR device of each seed
    baseline_best_seed*.scaps                best TPE device of each seed
    ga_best_seed*.scaps                      best GA device of each seed
results/                                     saved results read by the notebook
    trials.db, baseline_trials.db, ga_trials.db   all trials of PPSR, TPE and GA (SQLite)
    ml_surrogate/ml_dataset.csv              machine-learning dataset (see below)
    mc_uncertainty/                          Monte Carlo run around the final device
    headline_device/, loss_decomposition/, phase6_thickness_sensitivity/,
    fabrication_validity_3D/, R3_Q6_material_sensitivity/, R2_Q4_WO3_bandgap/, ...
    figures/                                 figures written by the notebook (PNG, PDF; panels in figures/panels)
LICENSE, CITATION.cff, requirements.txt
```

## Quick check of the final device

Open `scaps_files/baseline_best_seed789.scaps` in SCAPS-1D 3.3.11 and run the J-V calculation under AM1.5G at 300 K.
It gives PCE = 25.27%, Voc = 1.416 V, Jsc = 19.99 mA cm-2 and FF = 89.30%.

## Search ranges and random seeds

Ten variables were optimized (cell 0.4). Densities were sampled on a log scale.

| Variable | Range |
|---|---|
| WO3 (ETL) thickness | 20-100 nm |
| MASnIBr2 (absorber) thickness | 300-1400 nm |
| CBTS (HTL) thickness | 30-300 nm |
| WO3 donor density | 1e16-1e19 cm-3 |
| CBTS acceptor density | 1e17-1e20 cm-3 (the physics precheck rejects values above Nv = 1.8e19 cm-3) |
| Defect density of WO3, MASnIBr2 and CBTS | 1e13-1e16 cm-3 each |
| Interface defect density, CBTS/MASnIBr2 and MASnIBr2/WO3 | 1e9-1e12 cm-2 each |

The back-contact work function is fixed at 5.10 eV (Au). Seeds: 42, 123, 456, 789, 1000, 2024.
TPE and GA run 1000 trials per seed; PPSR stops when its rounds converge (600 trials per seed).

## Running the notebook

1. Install Python 3.10 or newer and the packages: `pip install -r requirements.txt`
2. Open `notebook/MASnIBr2_SCAPS_optimization.ipynb` in Jupyter.
3. Run all cells (Kernel -> Restart & Run All).

`RUN_SIMULATIONS = False` (default, cell 0.4): the notebook reads the saved results in `results/` and rebuilds all
tables and figures in about one hour. SCAPS-1D is not needed.

Cell 8.2 (hybrid stacking) keeps four 500-tree Extra Trees models in memory and needs about 8 GB of free RAM.

`RUN_SIMULATIONS = True`: every SCAPS-1D simulation is run again (about 3-4 days). This needs Windows and
SCAPS-1D 3.3.11; set `SCAPS_DIR` in cell 0.4 to the SCAPS-1D folder. Cells that call SCAPS-1D are marked [SCAPS].

SCAPS-1D is developed by M. Burgelman and colleagues at the University of Gent and is available free of charge on
request from the authors (https://scaps.elis.ugent.be/). It is not included in this repository.

## Machine-learning dataset

`results/ml_surrogate/ml_dataset.csv` is written by cell 8.1. It holds the 13,961 valid devices of all
optimization runs (PPSR 3,338, TPE 4,884, GA 5,739) that were used to train and test the surrogate models.

| Column | Meaning | Unit |
|---|---|---|
| algorithm | PPSR, TPE or GA | - |
| seed, trial_number | random seed and trial of the run | - |
| WO3_t, MASnIBr2_t, CBTS_t | layer thicknesses | um |
| WO3_ND | WO3 donor density | cm-3 |
| CBTS_NA | CBTS acceptor density | cm-3 |
| MASnIBr2_defect, WO3_defect, CBTS_defect | bulk defect densities | cm-3 |
| iface_CBTS_MASnIBr2_Nt, iface_MASnIBr2_WO3_Nt | interface defect densities | cm-2 |
| Voc, Jsc, FF, PCE | SCAPS-1D results | V, mA cm-2, %, % |
| split | `train` or `test` (random 80/20 split, seed 42) | - |

The inputs are the values given to SCAPS-1D; the models use the density columns on a log10 scale.
The 336 independent devices of the final Monte Carlo run are in `results/mc_uncertainty/mc_samples.csv`.

## Trained models

The trained models are attached to the GitHub Release v1.0 of this repository (too large for the repository itself):

| File | Content |
|---|---|
| `best_model_Extra_Trees.zip` | Extra Trees model of Table 7 (`best_model_Extra_Trees.pkl`) |
| `hybrid_stacking_models.zip` | direct, physics-reconstruction and Ridge stacking models of Table 8 (`hybrid_stacking_models.pkl`) |

Unzip them into `results/ml_surrogate/` and load them with `joblib.load(...)`. They were saved with scikit-learn 1.8.0
and need that version. The same models are rebuilt by cells 8.1 and 8.2 (fixed random seed 42).

## Where each result comes from

| Article | Notebook cell |
|---|---|
| Section 2.2 search ranges, Table 1 | 0.4, 1.2 |
| Physics precheck (C1-C4) | 2.1, `src/scaps_tools.py` |
| PPSR, TPE, GA runs (six seeds) | 3.1-3.3 |
| Table 5, Section 3.1 | 4.2, 4.7 |
| Table 6 | 4.4 |
| Fig. 3 | 4.5 |
| Fig. 4 | 4.6 |
| Tables 3-4 (final device) | 5.1 |
| Fig. 5 (J-V, QE; data also as CSV) | 5.2 |
| Energy band diagram (data as CSV) | 5.3 |
| Section 3.5 recombination | 5.4 |
| 1D thickness sweeps (supporting data) | 5.5 |
| Fig. 9 | 5.6, 5.7 |
| Fig. 10, SQ limit | 5.8-5.10 |
| Fig. 11 | 5.11, 5.12 |
| Fig. 6 | 6.1 |
| Fig. 7 | 6.2 |
| Section 3.3 Sobol indices | 6.3 |
| Section 3.4 Monte Carlo, Fig. 8 | 7.1, 7.2 |
| Section 3.6 material-parameter sensitivity | 7.3, 7.4 |
| Table 7, Fig. 12, ML dataset | 8.1, 8.4 |
| Table 8 | 8.2 |
| Table 9, Fig. 13 | 8.3, 8.5 |
| Section 3.7 validation (5-fold CV, leave-one-seed-out, independent devices, stacking test, Mondrian intervals) | 8.6-8.11 |

## Citation

If you use this code or data, please cite the article above (see also `CITATION.cff`).

## Licence

MIT, see `LICENSE`.
