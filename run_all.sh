#!/usr/bin/env bash
# Reproduces every result in the paper. Single-core runtime: about 2.5 hours in total.
set -euo pipefail
export OMP_NUM_THREADS=1
RELEASES="redaktor ivy-2.0 xerces-1.2 xerces-1.3 prop-6 xalan-2.4 ant-1.7 camel-1.6"
python src/fetch_data.py                                   # download + verify checksums
python src/classifiers.py                                  # Part A  -> results/res_clf.csv        (~5 min)
python src/tune.py $RELEASES                               # Part B  -> results/res_tune_<r>.csv     (~30 min)
for B in 50 200; do                                        # budget sensitivity (~75 min)
  BUDGET=$B ONLY=Random,DE,HHO SKIPDEF=1 python src/tune.py $RELEASES
done
python src/analysis.py                                     # all tables and statistics
python src/figures.py                                      # Fig. 1 and Fig. 2
