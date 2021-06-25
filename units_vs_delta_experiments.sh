#!/bin/bash

d=$(date "+%F %T")

export FAIR_SOLVER="GUROBI"

echo -e "\nSolver used in training: ${FAIR_SOLVER}\n"

python units_vs_delta_experiments.py

export FAIR_SOLVER="CBC"

echo -e "\nSolver used in verification: ${FAIR_SOLVER}\n"

python units_vs_delta_experiments.py --verify --experiment-name "pipeline - ${d}" --from-date "$d"

rm /tmp/.solver-log-*
