import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD


def learn_sensitive_hyperplane_cat(
        X_train, y_train, epochs, regularizer, batch_size, X_test=None, y_test=None):
    """
    X_train: the data frame of train features
    y_train: the data frame of train sensitive feature values
    epochs: number of epochs
    regularizer: l2 regularizer
    batch_size: batch size for training
    X_test: the data frame of test features
    y_test: the data frame of test sensitive feature values

    return: the weights of the single softmax layer indicating the correlation of other features to the target sensitive feature
    """

    out_shape = y_train.shape[1]
    SHUFFLE_BUFFER_SIZE = 100

    dataset_train = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
    dataset_train = dataset_train.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)

    if X_test is not None and y_test is not None:
        dataset_test = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))
        dataset_test = dataset_test.batch(batch_size)

    if regularizer is not None:
        regularizer = tf.keras.regularizers.l2(l=regularizer)

    model = tf.keras.models.Sequential([tf.keras.layers.Dense(
        out_shape, activation = tf.nn.softmax, kernel_regularizer=regularizer)])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(dataset_train, epochs=10)

    if X_test is not None and y_test is not None:
        print(f"test [loss, accuracy] = {model.evaluate(dataset_test)}")

    return model.layers[0].get_weights()[0]


def learn_sensitive_hyperplane_num(
        X_train, y_train, epochs, regularizer, batch_size, X_test=None, y_test=None):
    """
    X_train: the data frame of train features
    y_train: the data frame of train sensitive feature values
    epochs: number of epochs
    regularizer: l2 regularizer
    batch_size: batch size for training
    X_test: the data frame of test features
    y_test: the data frame of test sensitive feature values

    return: the weights of the single softmax layer indicating the correlation of other features to the target sensitive feature
    """

    out_shape = y_train.shape[1]
    SHUFFLE_BUFFER_SIZE = 100

    dataset_train = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
    dataset_train = dataset_train.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)

    if X_test is not None and y_test is not None:
        dataset_test = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))
        dataset_test = dataset_test.batch(batch_size)

    if regularizer is not None:
        regularizer = tf.keras.regularizers.l2(l=regularizer)

    model = tf.keras.models.Sequential([tf.keras.layers.Dense(
        out_shape, kernel_regularizer=regularizer)])

    model.compile(optimizer='adam',
                  loss='mean_absolute_error',
                  metrics=['accuracy'])

    model.fit(dataset_train, epochs=10)

    if X_test is not None and y_test is not None:
        print(f"test [loss, accuracy] = {model.evaluate(dataset_test)}")

    return model.layers[0].get_weights()[0]


def stack_weights(weights):
    """
    weights: list of weights of the correlation of each valid features with
        each of the sensitive features

    return: the redundant sensitive hyperplane
    """

    sensitive_directions = []
    for w in weights:
        sensitive_directions.append(w.T)

    sensitive_directions = np.vstack(sensitive_directions)

    return sensitive_directions


def get_span_of_sensitive_subspace(sensitive_subspace):
    """
    sensitive_subspace: the redundant sensitive subspace

    return: the span of the sensitive subspace
    """
    tSVD = TruncatedSVD(n_components=sensitive_subspace.shape[0])
    tSVD.fit(sensitive_subspace)
    span = tSVD.components_
    return span


def complement_projector(span):
    """
    span: the span of the sensitive directions

    return: the orthogonal complement projector of the span
    """

    basis = span.T

    proj = np.linalg.inv(basis.T @ basis)
    proj = basis @ proj @ basis.T
    proj_compl = np.eye(proj.shape[0]) - proj
    return proj_compl


def get_sensitive_subspace_and_complement_projector(weights):
    """
    weights: list of weights of the correlation of each valid features with
        each of the sensitive features

    return: the span of the sensitive subspace, the projection matrix onto the orthogonal complement
    """
    sensitive_subspace = stack_weights(weights)
    span = get_span_of_sensitive_subspace(sensitive_subspace)
    proj = complement_projector(span)
    return span, proj

