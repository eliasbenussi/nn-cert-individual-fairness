#!/bin/bash

d=$(date "+%F %T")

python time_cutoff_experiments.py

export FAIR_SOLVER="CBC"

echo -e "\nSolver used in verification: ${FAIR_SOLVER}\n"

python time_cutoff_experiments.py --verify --experiment-name "pipeline - ${d}" --from-date "$d"

rm /tmp/.solver-log-*
