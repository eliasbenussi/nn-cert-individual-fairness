#!/bin/bash

d=$(date "+%F %T")

python embedding_certification_experiments.py

export FAIR_SOLVER="CBC"

echo -e "\nSolver used in verification: ${FAIR_SOLVER}\n"

python embedding_certification_experiments.py --verify --experiment-name "pipeline - ${d}" --from-date "$d"

rm /tmp/.solver-log-*
