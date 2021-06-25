import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable

from keras_balanced_batch_generator import make_generator

from dataset.biased_dataset import BiasedDataset

import training.SenSR
import training.simpleMILP as MILPSR
from training.sensitive_subspace import (
    get_sensitive_subspace_and_complement_projector,
    learn_sensitive_hyperplane_cat,
    learn_sensitive_hyperplane_num,
)
from training.utils import save_model
from dataset.utils import (
    get_adult_data,
    get_credit_data,
    PROJ_ADULT_FILENAME,
    SSIF_WEIGHTS_ADULT_FILENAME,
    VANILLA_WEIGHTS_ADULT_FILENAME,
    VANILLA_WEIGHTS_CREDIT_FILENAME,
    SSIF_WEIGHTS_CREDIT_FILENAME,
    PROJ_CREDIT_FILENAME,
)


def train_vanilla_model(train_ds, test_ds, config, should_save_model=True, save_directory=None):
    epochs = config['vanilla_epochs']
    regularizer = config['reg']
    n_units = config['n_units']
    lr = config['lr']
    goal = config['training_goal']

    dataset_train = tf.data.Dataset.from_tensor_slices((train_ds.X_df.values, train_ds.y_df.values))
    dataset_test = tf.data.Dataset.from_tensor_slices((test_ds.X_df.values, test_ds.y_df.values))

    in_shape = train_ds.X_df.values.shape[1]
    out_shape = train_ds.y_df.values.shape[1]

    BATCH_SIZE = 256
    SHUFFLE_BUFFER_SIZE = 100

    dataset_train = dataset_train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    dataset_test = dataset_test.batch(BATCH_SIZE)
    if regularizer is not None:
        regularizer = tf.keras.regularizers.l2(l=regularizer)

    # Build model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(in_shape,)))
    if(config.get('embedding') is not None and config.get('embedding') != 0):
        model.add(tf.keras.layers.Dense(config['embedding'], activation=tf.nn.relu, kernel_regularizer=regularizer))
    for n in n_units:
        model.add(tf.keras.layers.Dense(n, activation=tf.nn.relu, kernel_regularizer=regularizer))
    model.add(tf.keras.layers.Dense(out_shape, activation = tf.nn.sigmoid))

    metrics = []
    if goal == 'classification':
        metrics = [
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'),
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
        ]
    else:
        metrics = [
            tf.keras.metrics.MeanSquaredError(name='MSE'),
            tf.keras.metrics.MeanAbsoluteError(name='MAE'),
            tf.keras.metrics.RootMeanSquaredError(name='RMSE'),
        ]
    loss = 'binary_crossentropy' if goal == 'classification' else 'mean_squared_error'

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        loss=loss,
        metrics=metrics
    )

    start_time = time.time()

    model.fit(dataset_train, epochs=epochs)

    print(f"test [{model.metrics_names}] = {model.evaluate(dataset_test)}")

    end_time = time.time()
    training_time = end_time - start_time
    config['training_time'] = training_time

    if should_save_model:
        save_model(model, config, save_directory=save_directory)

    return model


# TODO: THERE IS NO NEED HERE FOR TRAIN AND TEST, BECAUSE WE SHOULD USE THIS WITH TRAIN DURING
# TRAINING AND TEST DURING VERIFICATION
def generate_proj_for_distance(
    train_ds,
    test_ds=None,
    config=None,
):
    fair_hyperplane_epochs = config['fair_hyperplane_epochs']
    sens_batch_size = config['sensitive_batch_size']
    sens_reg = config['sensitive_reg']

    weights = []
    for i, sensitive_feature in enumerate(train_ds.sensitive_features):
        X_test = test_ds.X_df if test_ds is not None else None
        y_test = test_ds.sensitive_dfs[i] if test_ds is not None else None

        if sensitive_feature in train_ds.sens_cat_cols:
            w = learn_sensitive_hyperplane_cat(
                train_ds.X_df,
                train_ds.sensitive_dfs[i],
                epochs=fair_hyperplane_epochs,
                regularizer=sens_reg,
                batch_size=sens_batch_size,
                X_test=X_test,
                y_test=y_test,
            )
        else:
            w = learn_sensitive_hyperplane_num(
                train_ds.X_df,
                train_ds.sensitive_dfs[i],
                epochs=fair_hyperplane_epochs,
                regularizer=sens_reg,
                batch_size=sens_batch_size,
                X_test=X_test,
                y_test=y_test,
            )
        weights.append(w)

    sensitive_directions, proj = get_sensitive_subspace_and_complement_projector(weights)

    return proj, sensitive_directions


def train_individually_fair_subspace_robust_model(
    train_ds,
    test_ds,
    config=None,
    save_directory=None,
):
    proj, sensitive_directions = generate_proj_for_distance(
        train_ds, test_ds=test_ds, config=config)

    if(config['training_MILP'] == False):
        print('TRAINING YUROCHKIN FAIR MODEL...')
        weights, train_logits, test_logits, model = SenSR.train_fair_nn_binary(train_ds.X_df.values, train_ds.y_df.values, sensitive_directions, X_test=test_ds.X_df.values, y_test=test_ds.y_df.values, train_ds=train_ds, config=config, save_directory=save_directory)
    else:
        print('TRAINING MILP FAIR MODEL...')
        weights, train_logits, test_logits, model = MILPSR.train_fair_nn_binary(train_ds.X_df.values, train_ds.y_df.values, sensitive_directions, X_test=test_ds.X_df.values, y_test=test_ds.y_df.values, train_ds=train_ds, config=config, save_directory=save_directory)

    return weights, proj, model


def train_models_for_dataset(
    train_ds,
    test_ds,
    config,
    save_directory=None,
):
    save_directory
    if config['debiased_training']:
        _, _, model = train_individually_fair_subspace_robust_model(train_ds, test_ds, config, save_directory=save_directory)
    else:
        print('TRAINING VANILLA MODEL...')
        model = train_vanilla_model(train_ds, test_ds, config, save_directory=save_directory)

    print('...TRAINED.')
    return model


