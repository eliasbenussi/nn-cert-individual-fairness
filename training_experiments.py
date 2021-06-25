import os
from datetime import datetime
from time import time
import random
import argparse
from multiprocessing import Pool, cpu_count
from uuid import uuid4
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import mlflow
import tensorflow as tf
from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference

from dataset.utils import (
    get_adult_data,
    get_credit_data,
    get_crime_data,
    get_german_data,
    PROJ_ADULT_FILENAME,
    VANILLA_WEIGHTS_ADULT_FILENAME,
    SSIF_WEIGHTS_ADULT_FILENAME,
    PROJ_CREDIT_FILENAME,
    VANILLA_WEIGHTS_CREDIT_FILENAME,
    SSIF_WEIGHTS_CREDIT_FILENAME,
)
from training import generate_proj_for_distance, train_models_for_dataset
from verification import verify

RANDOM_SEED=42

# Try to arbitrarily not take all the cores
NPROC = 15 if cpu_count() > 10 else 4

CRIME_DROP_COLUMNS = [
    'HispPerCap', 'LandArea', 'LemasPctOfficDrugUn', 'MalePctNevMarr',
    'MedOwnCostPctInc', 'MedOwnCostPctIncNoMtg', 'MedRent',
    'MedYrHousBuilt', 'OwnOccHiQuart', 'OwnOccLowQuart',
    'OwnOccMedVal', 'PctBornSameState', 'PctEmplManu',
    'PctEmplProfServ', 'PctEmploy', 'PctForeignBorn', 'PctImmigRec5',
    'PctImmigRec8', 'PctImmigRecent', 'PctRecImmig10', 'PctRecImmig5',
    'PctRecImmig8', 'PctRecentImmig', 'PctSameCity85',
    'PctSameState85', 'PctSpeakEnglOnly', 'PctUsePubTrans',
    'PctVacMore6Mos', 'PctWorkMom', 'PctWorkMomYoungKids',
    'PersPerFam', 'PersPerOccupHous', 'PersPerOwnOccHous',
    'PersPerRentOccHous', 'RentHighQ', 'RentLowQ', 'Unnamed: 0',
    'agePct12t21', 'agePct65up', 'householdsize', 'indianPerCap',
    'pctUrban', 'pctWFarmSelf', 'pctWRetire', 'pctWSocSec', 'pctWWage',
    'whitePerCap'
]

ADULT_VANILLA_PARAMS = {
    'dataset_name': 'adult',
    'sensitive_features': ['sex', 'race'],
    'drop_columns': ['native-country', 'education'],
    'training_goal': 'classification',
    'vanilla_epochs': 35,
    'fair_hyperplane_epochs': 20,
    'sensitive_batch_size': 64,
    'sensitive_reg': 0.02,
    'fair_epochs': 400,
    'fair_batch_size': 5000,
    'reg': 0.0125,
    'n_units': [24],
    'lr': 0.025,
    'debiased_training': False,
    'training_MILP': False,
    'epsilon': 0.2,
    'delta': 0.1,
    'lambda': 0.5,
    'training_opt_mode': 'milp',
    'training_verif_time_limit': None,
}

ADULT_PGD_PARAMS = {
    'dataset_name': 'adult',
    'sensitive_features': ['sex', 'race'],
    'drop_columns': ['native-country', 'education'],
    'training_goal': 'classification',
    'vanilla_epochs': 55,
    'fair_hyperplane_epochs': 20,
    'sensitive_batch_size': 64,
    'sensitive_reg': 0.05,
    'fair_epochs': 400,
    'fair_batch_size': 5000,
    'reg': 0.05,
    'lr': 0.001,
    'debiased_training': True,
    'training_MILP': False,
    'epsilon': 0.2,
    'delta': 0.1,
    'lambda': 0.5,
    'training_opt_mode': 'milp',
    'training_verif_time_limit': None,
}

ADULT_MILP_PARAMS = {
    'dataset_name': 'adult',
    'sensitive_features': ['sex', 'race'],
    'drop_columns': ['native-country', 'education'],
    'training_goal': 'classification',
    'vanilla_epochs': 55,
    'fair_hyperplane_epochs': 20,
    'sensitive_batch_size': 64,
    'sensitive_reg': 0.05,
    'fair_epochs': 400,
    'fair_batch_size': 5000,
    'reg': 0.05,
    'lr': 0.001,
    'debiased_training': True,
    'training_MILP': True,
    'epsilon': 0.2,
    'delta': 0.1,
    'lambda': 0.5,
    'training_opt_mode': 'milp',
    'training_verif_time_limit': None,
}

CREDIT_VANILLA_PARAMS = {
    'dataset_name': 'credit',
    'sensitive_features': ['x2'],
    'drop_columns': [],
    'training_goal': 'classification',
    'vanilla_epochs': 50,
    'fair_hyperplane_epochs': 20,
    'sensitive_batch_size': 64,
    'sensitive_reg': 0.02,
    'fair_epochs': 100,
    'fair_batch_size': 5000,
    'reg': 0.02,
    'n_units': [8],
    'lr': 0.002,
    'debiased_training': False,
    'training_MILP': False,
    'epsilon': 0.2,
    'delta': 0.1,
    'lambda': 0.5,
    'training_opt_mode': 'milp',
    'training_verif_time_limit': None,
}

CREDIT_PGD_PARAMS = {
    'dataset_name': 'credit',
    'sensitive_features': ['x2'],
    'drop_columns': [],
    'training_goal': 'classification',
    'vanilla_epochs': 50,
    'fair_hyperplane_epochs': 20,
    'sensitive_batch_size': 64,
    'sensitive_reg': 0.04,
    'fair_epochs': 100,
    'fair_batch_size': 5000,
    'reg': 0.04,
    'n_units': [8],
    'lr': 0.0025,
    'debiased_training': True,
    'training_MILP': False,
    'epsilon': 0.2,
    'delta': 0.1,
    'lambda': 0.5,
    'training_opt_mode': 'milp',
    'training_verif_time_limit': None,
}

CREDIT_MILP_PARAMS = {
    'dataset_name': 'credit',
    'sensitive_features': ['x2'],
    'drop_columns': [],
    'training_goal': 'classification',
    'vanilla_epochs': 50,
    'fair_hyperplane_epochs': 20,
    'sensitive_batch_size': 64,
    'sensitive_reg': 0.04,
    'fair_epochs': 100,
    'fair_batch_size': 5000,
    'reg': 0.04,
    'lr': 0.0025,
    'debiased_training': True,
    'training_MILP': True,
    'epsilon': 0.2,
    'delta': 0.1,
    'lambda': 0.5,
    'training_opt_mode': 'milp',
    'training_verif_time_limit': None,
}

GERMAN_VANILLA_PARAMS = {
    'dataset_name': 'german',
    'sensitive_features': ['status_sex'],
    'drop_columns': [],
    'training_goal': 'classification',
    'vanilla_epochs': 35,
    'fair_hyperplane_epochs': 20,
    'sensitive_batch_size': 64,
    'sensitive_reg': 0.02,
    'fair_epochs': 250,
    'fair_batch_size': 300,
    'reg': 0.02,
    'n_units': [8],
    'lr': 0.001,
    'debiased_training': False,
    'training_MILP': False,
    'epsilon': 0.2,
    'delta': 0.1,
    'lambda': 0.5,
    'training_opt_mode': 'milp',
    'training_verif_time_limit': None,
}

GERMAN_PGD_PARAMS = {
    'dataset_name': 'german',
    'sensitive_features': ['status_sex'],
    'drop_columns': [],
    'training_goal': 'classification',
    'vanilla_epochs': 35,
    'fair_hyperplane_epochs': 20,
    'sensitive_batch_size': 64,
    'sensitive_reg': 0.04,
    'fair_epochs': 250,
    'fair_batch_size': 300,
    'reg': 0.04,
    'n_units': [8],
    'lr': 0.0025,
    'debiased_training': True,
    'training_MILP': False,
    'epsilon': 0.2,
    'delta': 0.1,
    'lambda': 0.5,
    'training_opt_mode': 'milp',
    'training_verif_time_limit': None,
}

GERMAN_MILP_PARAMS = {
    'dataset_name': 'german',
    'sensitive_features': ['status_sex'],
    'drop_columns': [],
    'training_goal': 'classification',
    'vanilla_epochs': 35,
    'fair_hyperplane_epochs': 20,
    'sensitive_batch_size': 64,
    'sensitive_reg': 0.04,
    'fair_epochs': 250,
    'fair_batch_size': 300,
    'reg': 0.04,
    'lr': 0.0025,
    'debiased_training': True,
    'training_MILP': True,
    'epsilon': 0.2,
    'delta': 0.1,
    'lambda': 0.5,
    'training_opt_mode': 'milp',
    'training_verif_time_limit': None,
}

CRIME_VANILLA_PARAMS = {
    'dataset_name': 'crime',
    'sensitive_features': ['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp'],
    'drop_columns': CRIME_DROP_COLUMNS,
    'training_goal': 'regression',
    'vanilla_epochs': 35,
    'fair_hyperplane_epochs': 20,
    'sensitive_batch_size': 64,
    'sensitive_reg': 0.02,
    'fair_epochs': 100,
    'fair_batch_size': 1500,
    'reg': 0.02,
    'n_units': [24],
    'lr': 0.001,
    'debiased_training': False,
    'training_MILP': False,
    'epsilon': 0.2,
    'delta': 0.1,
    'lambda': 0.5,
    'training_opt_mode': 'milp',
    'training_verif_time_limit': None,
}

CRIME_PGD_PARAMS = {
    'dataset_name': 'crime',
    'sensitive_features': ['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp'],
    'drop_columns': CRIME_DROP_COLUMNS,
    'training_goal': 'regression',
    'vanilla_epochs': 150,
    'fair_hyperplane_epochs': 20,
    'sensitive_batch_size': 64,
    'sensitive_reg': 0.025,
    'fair_epochs': 100,
    'fair_batch_size': 1500,
    'reg': 0.025,
    'n_units': [8,8],
    'lr': 0.025,
    'debiased_training': True,
    'training_MILP': False,
    'epsilon': 0.2,
    'delta': 0.1,
    'lambda': 0.15,
    'training_opt_mode': 'milp',
    'training_verif_time_limit': None,
}

CRIME_MILP_PARAMS = {
    'dataset_name': 'crime',
    'sensitive_features': ['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp'],
    'drop_columns': CRIME_DROP_COLUMNS,
    'training_goal': 'regression',
    'vanilla_epochs': 150,
    'fair_hyperplane_epochs': 20,
    'sensitive_batch_size': 64,
    'sensitive_reg': 0.025,
    'fair_epochs': 100,
    'fair_batch_size': 1500,
    'reg': 0.025,
    'lr': 0.025,
    'debiased_training': True,
    'training_MILP': True,
    'epsilon': 0.2,
    'delta': 0.1,
    'lambda': 0.5,
    'training_opt_mode': 'milp',
    'training_verif_time_limit': None,
}

EXPERIMENTS = {
    'adult_vanilla': ADULT_VANILLA_PARAMS,
    'adult_pgd': ADULT_PGD_PARAMS,
    'adult_milp': ADULT_MILP_PARAMS,

    'credit_vanilla': CREDIT_VANILLA_PARAMS,
    'credit_pgd': CREDIT_PGD_PARAMS,
    'credit_milp': CREDIT_MILP_PARAMS,

    'german_vanilla': GERMAN_VANILLA_PARAMS,
    'german_pgd': GERMAN_PGD_PARAMS,
    'german_milp': GERMAN_MILP_PARAMS,

    'crime_vanilla': CRIME_VANILLA_PARAMS,
    'crime_pgd': CRIME_PGD_PARAMS,
    'crime_milp': CRIME_MILP_PARAMS,
}

VERIFICATION_CONFIGS = [
   {
       'strategy': 'global',
       'opt_mode': 'milp',
       'M': 16,
       'mip': True,
       'dist_metric': 'mahalanobis',
       'time_limit': 180,
       'verification_epsilon': 0.2,
   }
]

def parallel_verify(args):
    (config, i, model_path, j, experiment_name) = args

    if experiment_name is not None:
        mlflow.set_experiment(experiment_name)

    with open(f'{model_path}/info.json', 'r') as f:
        import json
        train_config = json.load(f)

    dataset_name = train_config['dataset_name']
    sensitive_features = train_config['sensitive_features']
    drop_columns = train_config['drop_columns']

    if dataset_name == 'adult':
        train_ds, test_ds = get_adult_data(sensitive_features, drop_columns=drop_columns)
    elif dataset_name == 'credit':
        train_ds, test_ds = get_credit_data(sensitive_features, drop_columns=drop_columns)
    elif dataset_name == 'german':
        train_ds, test_ds = get_german_data(sensitive_features, drop_columns=drop_columns)
    elif dataset_name == 'crime':
        train_ds, test_ds = get_crime_data(sensitive_features, drop_columns=drop_columns)
    else:
        raise ValueError(f'"{dataset_name}" is not a valid dataset name')

    model = tf.keras.models.load_model(f'{model_path}/model')

    with mlflow.start_run() as run:
        mlflow.log_param('model_path', model_path)
        logging_train_config = train_config.copy()
        logging_train_config.pop('drop_columns')
        mlflow.log_params(logging_train_config)
        mlflow.log_params(config)

        model_name = 'vanilla'
        if train_config['debiased_training']:
            if train_config['training_MILP']:
                model_name = 'MILP'
            else:
                model_name = 'PGD'
        mlflow.log_param('model_name', model_name)

        metrics = model.evaluate(test_ds.X_df.values, test_ds.y_df.values)

        metrics_dict = {}
        for metric_name, metric in zip(model.metrics_names, metrics):
            metrics_dict[metric_name] = metric
        print(f'METRICS = {metrics_dict}')
        mlflow.log_metrics(metrics_dict)

        tp = metrics_dict.get('tp')
        if tp is not None:
            fn = metrics_dict['fn']
            tn = metrics_dict['tn']
            fp = metrics_dict['fp']

            balanced_accuracy = ((tp/(tp+fn)) + (tn/(tn+fp))) / 2
            mlflow.log_metric('balanced_accuracy', balanced_accuracy)

        proj, _ = generate_proj_for_distance(
            train_ds, config=train_config)

        verification_proc_id = uuid4()

        verification_results = verify(
            train_ds,
            test_ds,
            model,
            proj,
            config={**config, **train_config},
            proc_id=verification_proc_id,
        )
        mlflow.log_metrics(verification_results)
        print("HERE ARE THE VERIFICATION RESULTS: ", verification_results)

        training_goal = train_config.get('training_goal') or 'classification'
        if training_goal == 'classification':
            sensitive_features_names = train_config['sensitive_features']
            sensitive_features = test_ds.X_raw[sensitive_features_names]
            y_pred_prob = model.predict(test_ds.X_df.values)
            y_pred = (y_pred_prob > 0.5).astype(int)
            y_true = test_ds.y_df

            equalized_odds_diff = equalized_odds_difference(
                y_true, y_pred, sensitive_features=sensitive_features)

            demographic_parity_diff = demographic_parity_difference(
                y_true, y_pred, sensitive_features=sensitive_features)

            group_fairness_metrics = {
                'equalized_odds_diff': equalized_odds_diff,
                'demographic_parity_diff': demographic_parity_diff,
            }
            mlflow.log_metrics(group_fairness_metrics)


if __name__ == '__main__':

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    parser = argparse.ArgumentParser()

    parser.add_argument('--layers', type=int)
    parser.add_argument('--units', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--save-dir', type=str)
    parser.add_argument('--verify', action='store_true', default=False)
    args = parser.parse_args()
    experiment_name = 'Training Experiments'

    verify = args.verify
    save_directory = args.save_dir
    if not verify:
        n_units = [args.units] * args.layers
        print(f'N UNITS: {n_units}')
        train_config = EXPERIMENTS[args.dataset]
        train_config['n_units'] = n_units

    if verify:
        tf.random.set_seed(RANDOM_SEED)

        mlflow.set_experiment(experiment_name)

        args = []

        for i, config in enumerate(VERIFICATION_CONFIGS):

            models_dir = save_directory
            model_paths = os.listdir(models_dir)

            print(f'THESE ARE THE PATHS CONSIDERED: {model_paths}')

            for j, model_path in enumerate(model_paths):

                model_path = os.path.join(models_dir, model_path)
                arg = (config, i, model_path, j, experiment_name)
                args.append((arg))

        print(f'\n\VERIFICATION... VERIFIED 0/{len(args)}\n')

        with Pool(NPROC) as p:
            task_completed = 0
            for i, _ in enumerate(p.imap(parallel_verify, args)):
                task_completed += 1
                print(f'\nVERIFIED MODEL {i+1}. VERIFIED {task_completed}/{len(args)} @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
    else:
        tf.compat.v1.disable_v2_behavior()
        tf.compat.v1.random.set_random_seed(RANDOM_SEED)

        dataset_name = train_config['dataset_name']
        sensitive_features = train_config['sensitive_features']
        drop_columns = train_config['drop_columns']

        if dataset_name == 'adult':
            train_ds, test_ds = get_adult_data(sensitive_features, drop_columns=drop_columns)
        elif dataset_name == 'credit':
            train_ds, test_ds = get_credit_data(sensitive_features, drop_columns=drop_columns)
        elif dataset_name == 'german':
            train_ds, test_ds = get_german_data(sensitive_features, drop_columns=drop_columns)
        elif dataset_name == 'crime':
            train_ds, test_ds = get_crime_data(sensitive_features, drop_columns=drop_columns)
        else:
            raise ValueError(f'"{dataset_name}" is not a valid dataset name')

        train_models_for_dataset(
            train_ds,
            test_ds,
            config=train_config,
            save_directory=save_directory,
        )

