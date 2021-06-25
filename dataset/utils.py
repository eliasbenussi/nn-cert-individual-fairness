import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from dataset.biased_dataset import BiasedDataset

DATA_ADULT_TRAIN = './data/raw/adult.data.csv'
DATA_ADULT_TEST = './data/raw/adult.test.csv'

DATA_CRIME_FILENAME = './data/raw/crime.csv'

DATA_GERMAN_FILENAME = './data/raw/german.csv'

# ADULT DATASET
# Listing of attributes:

# target: >50K, <=50K.

# age: continuous.
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov,
#     Without-pay, Never-worked.
# fnlwgt: continuous.
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th,
#     7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# education-num: continuous.
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
#     Married-spouse-absent, Married-AF-spouse.
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
#     Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving,
#     Priv-house-serv, Protective-serv, Armed-Forces.
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# sex: Female, Male.
# capital-gain: continuous.
# capital-loss: continuous.
# hours-per-week: continuous.
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany,
#     Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras,
#     Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France,
#     Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua,
#     Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
def get_adult_data(sensitive_features, drop_columns=[], test_size=0.2, random_state=42):
    """
    train_path: path to training data
    test_path: path to test data

    returns: tuple of training features, training labels, test features and test labels
    """
    train_df = pd.read_csv(DATA_ADULT_TRAIN, na_values='?').dropna()
    test_df = pd.read_csv(DATA_ADULT_TEST, na_values='?').dropna()
    target = 'target'

    X_train = train_df.drop(columns=[target])
    y_train = train_df[[target]]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[[target]]

    train_ds = BiasedDataset(
        X_train, y_train, sensitive_features=sensitive_features, drop_columns=drop_columns)
    test_ds = BiasedDataset(
        X_test, y_test, sensitive_features=sensitive_features, drop_columns=drop_columns)

    return train_ds, test_ds


# CREDIT DATASET:
# This research employed a binary variable, default payment (Yes = 1, No = 0), as the response
# variable. This study reviewed the literature and used the following 23 variables as explanatory
# variables:
#     x1: Amount of the given credit (NT dollar): it includes both the individual consumer
#         credit and his/her family (supplementary) credit.
#     x2: Gender (1 = male; 2 = female).
#     x3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
#     x4: Marital status (1 = married; 2 = single; 3 = others).
#     x5: Age (year).
#     x6 - x11: History of past payment. We tracked the past monthly payment records (from April to
#         September, 2005) as follows: x6 = the repayment status in September, 2005; x7 = the
#         repayment status in August, 2005; . . .;x11 = the repayment status in April, 2005. The
#         measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one
#         month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months;
#         9 = payment delay for nine months and above.
#     x12-x17: Amount of bill statement (NT dollar). x12 = amount of bill statement in September,
#         2005; x13 = amount of bill statement in August, 2005; . . .; x17 = amount of bill
#         statement in April, 2005.
#     x18-x23: Amount of previous payment (NT dollar). x18 = amount paid in September, 2005;
#         x19 = amount paid in August, 2005; . . .;x23 = amount paid in April, 2005.
def get_credit_data(sensitive_features, drop_columns=[], test_size=0.2, random_state=42):
    """
    sensitive_features: features that should be considered sensitive when building the
        BiasedDataset object
    drop_columns: columns we can ignore and drop
    random_state: to pass to train_test_split

    return: two BiasedDataset objects, for training and test data respectively
    """
    credit_data = fetch_openml(data_id=42477, as_frame=True, data_home='./data/raw')

    # Force categorical data do be dtype: category
    features = credit_data.data
    categorical_features = ['x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11']
    for cf in categorical_features:
        features[cf] = features[cf].astype(str).astype('category')

    # Encode output
    target = (credit_data.target == "1") * 1
    target = pd.DataFrame({'target': target})

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state)

    train_ds = BiasedDataset(
        X_train, y_train, sensitive_features=sensitive_features, drop_columns=drop_columns)
    test_ds = BiasedDataset(
        X_test, y_test, sensitive_features=sensitive_features, drop_columns=drop_columns)

    return train_ds, test_ds


def get_crime_data(sensitive_features, drop_columns=[], test_size=0.2, random_state=42):
    data_df = pd.read_csv(DATA_CRIME_FILENAME, na_values='?').dropna()

    train_df, test_df = train_test_split(data_df, test_size=test_size, random_state=random_state)
    target = 'ViolentCrimesPerPop'

    X_train = train_df.drop(columns=[target])
    y_train = train_df[[target]]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[[target]]

    train_ds = BiasedDataset(
        X_train, y_train, sensitive_features=sensitive_features, drop_columns=drop_columns)
    test_ds = BiasedDataset(
        X_test, y_test, sensitive_features=sensitive_features, drop_columns=drop_columns)

    return train_ds, test_ds


def get_german_data(sensitive_features, drop_columns=[], test_size=0.2, random_state=42):
    data_df = pd.read_csv(DATA_GERMAN_FILENAME, na_values='?').dropna()

    train_df, test_df = train_test_split(data_df, test_size=test_size, random_state=random_state)
    target = 'target'

    X_train = train_df.drop(columns=[target])
    y_train = train_df[[target]]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[[target]]

    train_ds = BiasedDataset(
        X_train, y_train, sensitive_features=sensitive_features, drop_columns=drop_columns)
    test_ds = BiasedDataset(
        X_test, y_test, sensitive_features=sensitive_features, drop_columns=drop_columns)

    return train_ds, test_ds



if __name__ == '__main__':

    sensitive_features = ['x2']
    train_ds, test_ds = get_credit_data(sensitive_features)

    print(f'FEATURES: {len(train_ds.X_df.columns)}')

    # sensitive_features = ['sex']
    # drop_columns = ['native-country', 'education']
    # train_ds, test_ds = get_adult_data(sensitive_features, drop_columns=drop_columns)
