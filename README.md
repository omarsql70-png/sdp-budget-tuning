# sdp-budget-tuning

Code, raw results and analysis scripts for the paper

> M. Y. Al-Tarawneh and O. I. Al-Mrayat, "Do Metaheuristics Really Improve Software Defect Prediction? A Budget-Fair Empirical Study," 2026 (under review).

The study compares seven classifiers (with and without SMOTE) and five ways of tuning an RBF-SVM — default settings, grid search, random search, Differential Evolution (DE) and Harris Hawks Optimization (HHO) — on eight PROMISE releases. Every tuner receives the same number of fitness evaluations inside a nested, repeated stratified cross-validation, and the Matthews correlation coefficient (MCC) is used as both fitness and primary metric.

## Repository layout

```
src/
  common.py        data loading, SVM pipeline (z-score -> SMOTE -> SVC), inner-CV fitness, metrics
  fetch_data.py    downloads the 8 releases from pinned public mirrors and verifies SHA-256 checksums
  classifiers.py   Part A: 7 classifiers + majority baseline, with/without SMOTE
  tune.py          Part B: Default / Grid / Random / DE / HHO tuning of the SVM (budget via $BUDGET)
  analysis.py      recomputes every table and statistic in the paper -> results/summary.json
  figures.py       Fig. 1 (protocol) and Fig. 2 (inner-CV vs held-out MCC)
results/           per-fold raw results used in the paper (CSV) and summary.json
figures/           generated figures
run_all.sh         reproduces everything end to end
```

## Reproducing the results

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh            # about 2.5 hours on one CPU core
```

To check the paper's numbers without rerunning the experiments, use the CSV files already in `results/`:

```bash
python src/fetch_data.py   # needed only for Table I (dataset sizes)
python src/analysis.py
```

## Experimental protocol

| Item | Setting |
|---|---|
| Data | ant-1.7, camel-1.6, ivy-2.0, prop-6, redaktor, xalan-2.4, xerces-1.2, xerces-1.3 (20 CK/OO metrics; defective = bug count > 0) |
| Outer validation | stratified 5-fold CV × 2 repeats (seed 42) → 10 held-out folds per release |
| Preprocessing | z-score, then SMOTE (k = 3), fitted on training folds only |
| Search space | log2 C ∈ [−5, 10], log2 γ ∈ [−10, 3] |
| Fitness | mean MCC over an inner stratified 3-fold CV of the training fold |
| Budget | 100 evaluations per tuner (sensitivity runs: 50 and 200 for Random, DE, HHO) |
| DE | SciPy `differential_evolution`, DE/rand/1/bin, population 10, F ∈ [0.5, 1.0], CR = 0.7, no polishing |
| HHO | own implementation of Heidari et al. (2019), 10 hawks, Lévy β = 1.5, greedy acceptance, stops at the budget |
| Statistics | Friedman on per-release means, Wilcoxon signed-rank on paired folds, Cliff's δ |

## Data sources

The data are not redistributed here. `src/fetch_data.py` downloads them from:

* [feiwww/PROMISE-backup](https://github.com/feiwww/PROMISE-backup) at commit `1e8fd48` — ant-1.7, camel-1.6, xalan-2.4, xerces-1.2, xerces-1.3
* [klainfo/DefectData](https://github.com/klainfo/DefectData) (MIT) at commit `e65993d` — ivy-2.0, prop-6, redaktor

Original collection: M. Jureczko and L. Madeyski, PROMISE 2010.

## Environment

Python 3.12, numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, imbalanced-learn 0.14.2, SciPy 1.17.1, matplotlib 3.10.8. Run times in the paper were measured on a single CPU core (`OMP_NUM_THREADS=1`); absolute times will differ on other machines, but the ratios between tuners should not.

## License

Code: MIT (see `LICENSE`). Data: see the licenses of the source repositories above.
