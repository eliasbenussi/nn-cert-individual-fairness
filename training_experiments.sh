#!/bin/bash


export FAIR_SOLVER="GUROBI"

echo -e "\nSolver used in training: ${FAIR_SOLVER}\n"

rm -rf tmp_saved_models
mkdir -p tmp_saved_models

# Models for Adult dataset
python training_experiments.py --save-dir tmp_saved_models --dataset adult_vanilla --layers 1 --units 8
python training_experiments.py --save-dir tmp_saved_models --dataset adult_vanilla --layers 1 --units 16
python training_experiments.py --save-dir tmp_saved_models --dataset adult_vanilla --layers 1 --units 24
python training_experiments.py --save-dir tmp_saved_models --dataset adult_vanilla --layers 1 --units 32
python training_experiments.py --save-dir tmp_saved_models --dataset adult_vanilla --layers 1 --units 64
python training_experiments.py --save-dir tmp_saved_models --dataset adult_vanilla --layers 2 --units 8
python training_experiments.py --save-dir tmp_saved_models --dataset adult_vanilla --layers 2 --units 16

python training_experiments.py --save-dir tmp_saved_models --dataset adult_pgd --layers 1 --units 8
python training_experiments.py --save-dir tmp_saved_models --dataset adult_pgd --layers 1 --units 16
python training_experiments.py --save-dir tmp_saved_models --dataset adult_pgd --layers 1 --units 24
python training_experiments.py --save-dir tmp_saved_models --dataset adult_pgd --layers 1 --units 32
python training_experiments.py --save-dir tmp_saved_models --dataset adult_pgd --layers 1 --units 64
python training_experiments.py --save-dir tmp_saved_models --dataset adult_pgd --layers 2 --units 8
python training_experiments.py --save-dir tmp_saved_models --dataset adult_pgd --layers 2 --units 16

python training_experiments.py --save-dir tmp_saved_models --dataset adult_milp --layers 1 --units 8
python training_experiments.py --save-dir tmp_saved_models --dataset adult_milp --layers 1 --units 16
python training_experiments.py --save-dir tmp_saved_models --dataset adult_milp --layers 1 --units 24
python training_experiments.py --save-dir tmp_saved_models --dataset adult_milp --layers 1 --units 32
python training_experiments.py --save-dir tmp_saved_models --dataset adult_milp --layers 1 --units 64
python training_experiments.py --save-dir tmp_saved_models --dataset adult_milp --layers 2 --units 8
python training_experiments.py --save-dir tmp_saved_models --dataset adult_milp --layers 2 --units 16

# Models for Crime dataset
python training_experiments.py --save-dir tmp_saved_models --dataset crime_vanilla --layers 1 --units 8
python training_experiments.py --save-dir tmp_saved_models --dataset crime_vanilla --layers 1 --units 12
python training_experiments.py --save-dir tmp_saved_models --dataset crime_vanilla --layers 1 --units 16
python training_experiments.py --save-dir tmp_saved_models --dataset crime_vanilla --layers 1 --units 24
python training_experiments.py --save-dir tmp_saved_models --dataset crime_vanilla --layers 2 --units 8
python training_experiments.py --save-dir tmp_saved_models --dataset crime_vanilla --layers 2 --units 16

python training_experiments.py --save-dir tmp_saved_models --dataset crime_pgd --layers 1 --units 8
python training_experiments.py --save-dir tmp_saved_models --dataset crime_pgd --layers 1 --units 12
python training_experiments.py --save-dir tmp_saved_models --dataset crime_pgd --layers 1 --units 16
python training_experiments.py --save-dir tmp_saved_models --dataset crime_pgd --layers 1 --units 24
python training_experiments.py --save-dir tmp_saved_models --dataset crime_pgd --layers 2 --units 8
python training_experiments.py --save-dir tmp_saved_models --dataset crime_pgd --layers 2 --units 16

python training_experiments.py --save-dir tmp_saved_models --dataset crime_milp --layers 1 --units 8
python training_experiments.py --save-dir tmp_saved_models --dataset crime_milp --layers 1 --units 12
python training_experiments.py --save-dir tmp_saved_models --dataset crime_milp --layers 1 --units 16
python training_experiments.py --save-dir tmp_saved_models --dataset crime_milp --layers 1 --units 24
python training_experiments.py --save-dir tmp_saved_models --dataset crime_milp --layers 2 --units 8
python training_experiments.py --save-dir tmp_saved_models --dataset crime_milp --layers 2 --units 16

# Models for Credit dataset
python training_experiments.py --save-dir tmp_saved_models --dataset credit_vanilla --layers 1 --units 8
python training_experiments.py --save-dir tmp_saved_models --dataset credit_vanilla --layers 1 --units 16
python training_experiments.py --save-dir tmp_saved_models --dataset credit_vanilla --layers 1 --units 24
python training_experiments.py --save-dir tmp_saved_models --dataset credit_vanilla --layers 1 --units 32
python training_experiments.py --save-dir tmp_saved_models --dataset credit_vanilla --layers 1 --units 64
python training_experiments.py --save-dir tmp_saved_models --dataset credit_vanilla --layers 2 --units 8
python training_experiments.py --save-dir tmp_saved_models --dataset credit_vanilla --layers 2 --units 16

python training_experiments.py --save-dir tmp_saved_models --dataset credit_pgd --layers 1 --units 8
python training_experiments.py --save-dir tmp_saved_models --dataset credit_pgd --layers 1 --units 16
python training_experiments.py --save-dir tmp_saved_models --dataset credit_pgd --layers 1 --units 24
python training_experiments.py --save-dir tmp_saved_models --dataset credit_pgd --layers 1 --units 32
python training_experiments.py --save-dir tmp_saved_models --dataset credit_pgd --layers 1 --units 64
python training_experiments.py --save-dir tmp_saved_models --dataset credit_pgd --layers 2 --units 8
python training_experiments.py --save-dir tmp_saved_models --dataset credit_pgd --layers 2 --units 16

python training_experiments.py --save-dir tmp_saved_models --dataset credit_milp --layers 1 --units 8
python training_experiments.py --save-dir tmp_saved_models --dataset credit_milp --layers 1 --units 16
python training_experiments.py --save-dir tmp_saved_models --dataset credit_milp --layers 1 --units 24
python training_experiments.py --save-dir tmp_saved_models --dataset credit_milp --layers 1 --units 32
python training_experiments.py --save-dir tmp_saved_models --dataset credit_milp --layers 1 --units 64
python training_experiments.py --save-dir tmp_saved_models --dataset credit_milp --layers 2 --units 8
python training_experiments.py --save-dir tmp_saved_models --dataset credit_milp --layers 2 --units 16

# Models for German dataset
python training_experiments.py --save-dir tmp_saved_models --dataset german_vanilla --layers 1 --units 8
python training_experiments.py --save-dir tmp_saved_models --dataset german_vanilla --layers 1 --units 16
python training_experiments.py --save-dir tmp_saved_models --dataset german_vanilla --layers 1 --units 24
python training_experiments.py --save-dir tmp_saved_models --dataset german_vanilla --layers 1 --units 32
python training_experiments.py --save-dir tmp_saved_models --dataset german_vanilla --layers 1 --units 64
python training_experiments.py --save-dir tmp_saved_models --dataset german_vanilla --layers 2 --units 8
python training_experiments.py --save-dir tmp_saved_models --dataset german_vanilla --layers 2 --units 16

python training_experiments.py --save-dir tmp_saved_models --dataset german_pgd --layers 1 --units 8
python training_experiments.py --save-dir tmp_saved_models --dataset german_pgd --layers 1 --units 16
python training_experiments.py --save-dir tmp_saved_models --dataset german_pgd --layers 1 --units 24
python training_experiments.py --save-dir tmp_saved_models --dataset german_pgd --layers 1 --units 32
python training_experiments.py --save-dir tmp_saved_models --dataset german_pgd --layers 1 --units 64
python training_experiments.py --save-dir tmp_saved_models --dataset german_pgd --layers 2 --units 8
python training_experiments.py --save-dir tmp_saved_models --dataset german_pgd --layers 2 --units 16

python training_experiments.py --save-dir tmp_saved_models --dataset german_milp --layers 1 --units 8
python training_experiments.py --save-dir tmp_saved_models --dataset german_milp --layers 1 --units 16
python training_experiments.py --save-dir tmp_saved_models --dataset german_milp --layers 1 --units 24
python training_experiments.py --save-dir tmp_saved_models --dataset german_milp --layers 1 --units 32
python training_experiments.py --save-dir tmp_saved_models --dataset german_milp --layers 1 --units 64
python training_experiments.py --save-dir tmp_saved_models --dataset german_milp --layers 2 --units 8
python training_experiments.py --save-dir tmp_saved_models --dataset german_milp --layers 2 --units 16

# Training one large model for the adult dataset as a scalability proof of concept
python training_experiments.py --save-dir tmp_saved_models --dataset adult_milp --layers 1 --units 100

# Verify all models stored in tmp_saved_models
export FAIR_SOLVER="CBC"

echo -e "\nSolver used in verification: ${FAIR_SOLVER}\n"
python training_experiments.py --save-dir tmp_saved_models --verify

mv tmp_saved_models/* saved_models

rm /tmp/.solver-log-*
