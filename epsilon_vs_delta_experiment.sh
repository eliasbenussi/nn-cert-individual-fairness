#!/bin/bash

d=$(date "+%F %T")

python epsilon_vs_delta_experiment.py

export FAIR_SOLVER="CBC"

echo -e "\nSolver used in verification: ${FAIR_SOLVER}\n"

python epsilon_vs_delta_experiment.py --verify --experiment-name "pipeline - ${d}" --from-date "$d"

rm /tmp/.solver-log-*
