import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class BiasedDataset:

    def __init__(self, X_raw, y_raw, sensitive_features=[], drop_columns=[], drop_first=False, drop_first_labels=True):
        """
        X_raw: features dataframe
        y_raw: labels dataframe or column label
        sensitive_features: the features considered sensitive
        drop_columns: the columns considered superfluous and to be deleted
        drop_first: whether to drop first when one-hot encoding features
        drop_first_labels: whether to drop first when one-hot encoding labels
        """

        self.sensitive_features = sensitive_features

        X_raw.drop(columns=drop_columns, inplace=True)
        self.X_raw = X_raw

        num_cols, cat_cols, sens_num_cols, sens_cat_cols = self.get_num_cat_columns_sorted(X_raw, sensitive_features)
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.sens_num_cols = sens_num_cols
        self.sens_cat_cols = sens_cat_cols
        self.original_columns = X_raw.columns.values.tolist()

        X_df_all, y_df = self.prepare_dataset(X_raw, y_raw, drop_first=drop_first, drop_first_labels=drop_first_labels, drop_columns=drop_columns)

        self.y_df = y_df

        # Map from original column names to all new encoded ones
        all_columns_map = {}
        encoded_columns = X_df_all.columns.values.tolist()
        for c in self.original_columns:
            all_columns_map[c] = [ encoded_columns.index(e_c) for e_c in encoded_columns if e_c == c or e_c.startswith(c + '_') ]

        # List of list of the indexes of each sensitive features
        encoded_features = X_df_all.columns.values.tolist()
        sensitive_idxs = []
        sensitive_idxs_flat = []
        for sf in sensitive_features:
            sensitive_idxs.append(all_columns_map[sf])
            sensitive_idxs_flat.extend(all_columns_map[sf])
        all_idxs = [i for i in range(len(X_df_all.columns))]
        valid_idxs = [i for i in all_idxs if i not in sensitive_idxs_flat]

        # Datasets with one-hot encoded columns of each sensitive feature
        self.sensitive_dfs = [X_df_all.iloc[:, idxs] for idxs in sensitive_idxs]

        # Dataset with all features but the sensitive ones
        self.X_df = X_df_all.iloc[:, valid_idxs]

        self.columns_map = {}
        encoded_columns = self.X_df.columns.values.tolist()
        for c in num_cols + cat_cols:
            self.columns_map[c] = [ encoded_columns.index(e_c) for e_c in encoded_columns if e_c == c or e_c.startswith(c + '_') ]

    def get_num_cat_columns_sorted(self, X_df, sensitive_features):
        num_cols = []
        cat_cols = []

        sens_num_cols = []
        sens_cat_cols = []

        for c in X_df.columns:
            if c in sensitive_features:
                if X_df[c].dtype == 'object' or X_df[c].dtype.name == 'category':
                    sens_cat_cols.append(c)
                else:
                    sens_num_cols.append(c)
            else:
                if X_df[c].dtype == 'object' or X_df[c].dtype.name == 'category':
                    cat_cols.append(c)
                else:
                    num_cols.append(c)

        num_cols.sort()
        cat_cols.sort()
        sens_num_cols.sort()
        sens_cat_cols.sort()

        return num_cols, cat_cols, sens_num_cols, sens_cat_cols

    def scale_num_cols(self, X_df, num_cols):
        """
        X_df: features dataframe
        num_cols: name of all numerical columns to be scaled

        returns: feature dataframe with scaled numerical features
        """
        X_df_scaled = X_df.copy()
        scaler = MinMaxScaler()
        X_num = scaler.fit_transform(X_df_scaled[num_cols])

        for i, c in enumerate(num_cols):
            X_df_scaled[c] = X_num[:,i]

        return X_df_scaled

    def process_num_cat_columns(self, X_df, drop_first):
        """
        X_df: features dataframe

        returns: feature dataframe with scaled numerical features and one-hot encoded categorical features
        """
        num_cols = []
        cat_cols = []

        for c in X_df.columns:
            if X_df[c].dtype == 'object' or X_df[c].dtype.name == 'category':
                cat_cols.append(c)
            else:
                num_cols.append(c)

        # TODO: need to think about this drop_first
        X_df_encoded = pd.get_dummies(X_df, columns=cat_cols, drop_first=drop_first)

        cat_cols = list(set(X_df_encoded.columns) - set(num_cols))

        num_cols.sort()
        cat_cols.sort()

        X_df_encoded_scaled = self.scale_num_cols(X_df_encoded, num_cols)

        return X_df_encoded_scaled[num_cols + cat_cols]


    def process_labels(self, X_df, y_df, drop_first):
        X_processed = X_df.copy()
        if isinstance(y_df, str):
            prefix = y_df
            y_columns = [ c for c in X_processed.columns if c == prefix or c.startswith(prefix + '_') ]
            y_processed = X_df[y_columns]
            X_processed.drop(columns=y_columns, inplace=True)
        else:
            y_processed = pd.get_dummies(y_df, drop_first=drop_first)

        return X_processed, y_processed


    def prepare_dataset(self, X_df_original, y_df_original, drop_first, drop_first_labels, drop_columns=[]):
        """
        X_df_original: features dataframe
        y_df_original: labels dataframe

        returns:
            - feature dataframe with scaled numerical features and one-hot encoded categorical features
            - one hot encoded labels, with drop_first option
        """
        X_df = X_df_original.copy()

        X_processed = self.process_num_cat_columns(X_df, drop_first)

        X_processed, y_processed = self.process_labels(X_processed, y_df_original, drop_first_labels)

        return X_processed, y_processed
