# Leakage-aware evaluation of student dropout prediction on OULAD

Code and result files for the study *"How Much of Dropout Prediction on OULAD Is Temporal Leakage? A Leakage-Aware Evaluation of Unsupervised LSTM Representations"* (manuscript in preparation).

**Author:** Ouafae El Aissaoui, Sidi Mohamed Ben Abdellah University, Fez, Morocco.

The study evaluates the same models under two designs on the Open University Learning Analytics Dataset (OULAD):

- **Design A** — all students are kept, including those who have already withdrawn during the observation window;
- **Design B** (leakage-aware) — only students still registered at the end of the window are kept, so that only future withdrawals are predicted.

It then compares an LSTM autoencoder embedding with simple controls (raw sequence, weekly totals, activity-type totals, PCA, untrained encoder) under Design B.

## What is in this repository

| Path | Content |
|---|---|
| `src/` | Initial end-of-course pipeline: data loading, cohort definition, weekly sequences, aggregated descriptors, autoencoder, baselines |
| `experiments/early_windows/` | All window experiments (Designs A and B), controls, seeds, bootstrap, sensitivity analyses. `run_all.sh` runs the chain; every step is skipped if its output already exists |
| `experiments/diagnostics/` | Cohort statistics and diagnostics of the end-of-course run |
| `results/` | Result files (JSON/CSV) from which every number of the article is generated; `results/early_windows/SUMMARY.md` is a generated overview (in French) |
| `docs/PROVENANCE.md` | For each result file: producing script, environment, inputs, fold partition (in French) |
| `tools/` | `summarize_results.py` regenerates `SUMMARY.md`; `check_ablation_folds.py` checks the end-of-course result files |
| `article/v13/` | `res.py` and `figs.py`: load the result files and draw the figures of the article |
| `logs/env_freeze.txt` | Exact package versions used |

Code comments and some documentation files are in French.

## What is not in this repository

- **The OULAD data.** It is public and must be downloaded from its official source (see below).
- **- **Fold assignments, out-of-fold predictions and autoencoder embeddings.** They are deposited in a separate archive on Zenodo: <https://doi.org/10.5281/zenodo.23314322>.

## Data

OULAD is distributed by the Open University under a CC-BY 4.0 licence: <https://analyse.kmi.open.ac.uk/open-dataset>. Please cite:

> Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data*, 4, 170171. https://doi.org/10.1038/sdata.2017.171

Create a `data/` folder at the root of the repository and place the seven CSV files in it: `studentInfo.csv`, `studentRegistration.csv`, `studentVle.csv`, `vle.csv`, `assessments.csv`, `studentAssessment.csv`, `courses.csv`.

## Environment

The results were produced with Python 3.11.9, TensorFlow 2.16.2 (Keras 3.15.1), scikit-learn 1.9.1, XGBoost 3.2.0, NumPy 1.26.4 and pandas 3.0.6, on macOS (Apple silicon, Metal GPU backend).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt            # or requirements-macos-arm64.txt on Apple silicon
```

`logs/env_freeze.txt` lists the exact versions of all installed packages.

## Running the experiments

```bash
# 1. Initial pipeline: builds the cohort, the weekly sequences and the descriptors in interim/
cd src && python run_pipeline.py && cd ..

# 2. Window experiments (idempotent: existing outputs are skipped)
bash experiments/early_windows/run_all.sh

# 3. Overview of the results
python tools/summarize_results.py
```

Each step is skipped when its output file exists; `FORCE=1` forces a recomputation.

## Notes on reproducibility

- **Fold partition.** All models use the same 5-fold partition grouped by student (`GroupKFold`). The partition produced by `GroupKFold` can differ between scikit-learn versions and platforms; the assignment used in the article is therefore stored with the out-of-fold predictions in the archive.
- **Autoencoders.** For each window, the main autoencoder is the one trained with seed 42 by `train_ae_seeds.py` (`interim/early_windows/seeds/ae_w<w>_s42.npz`, copied to `ae_w<w>.npz`); seeds 43 to 46 are used for the seed-variance analyses. `homogenize_ae.sh` and `recompute_r17.sh` are one-off scripts kept for traceability: they document, respectively, the switch to these autoencoders and an earlier recomputation on a single fold partition.
- **Neural networks.** Training on a GPU is not bit-reproducible. The archived embeddings and predictions make it possible to re-run every evaluation and every bootstrap without retraining.
- **End-of-course embedding.** `interim/embeddings.npy` (39-week autoencoder) comes from the initial run of the pipeline, for which package versions were not recorded. The files in `results/v11_end_of_course/` also come from that initial run, on a different fold partition; the article uses `results/early_windows/results_endcourse_w39.json` instead.
- **No hyperparameter tuning** was performed, with one exception reported in the article (two regularization strengths for one control under label scarcity).

## Licence

The code is released under the MIT licence (see `LICENSE`). OULAD itself is distributed by the Open University under CC-BY 4.0.

## How to cite

See `CITATION.cff`. A reference to the article will be added once it is published.
