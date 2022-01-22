# coding=utf-8
import pandas as pd
from typing import Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
sc = StandardScaler()


def fetch_uci(
    name: Optional[str] = None,
    need_pre: Optional[bool] = True
):
    """Fetch dataset from uci by name.
    UCI Data URL: http://archive.ics.uci.edu/ml/datasets.php
    :return: X, y
    """
    if name == 'German':
        path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
        df = pd.read_csv(path, header=None)
        # str to int
        for i in [1, 4, 7, 10, 12, 15, 17, 20]:
            df[i] = df[i].astype(int)
        if not need_pre:
            return df.iloc[:, :-1].values, df.iloc[:, -1].values  # X, y
        else:
            # Categorical to int
            for i in [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]:
                df[i] = LabelEncoder().fit_transform(df[i])
            X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
            X, y = sc.fit_transform(X), 2 * y - 3
            return X, y
    elif name == 'Ionosphere':
        path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
        df = pd.read_csv(path, header=None)
        if not need_pre:
            return df.iloc[:, :-1].values, df.iloc[:, -1].values
        else:
            df[34] = LabelEncoder().fit_transform(df[34])
            X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
            X, y = sc.fit_transform(X), 2 * y - 1
        return X, y
    elif name == 'Spambase':
        path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
        df = pd.read_csv(path, header=None)
        X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
        if need_pre:
            X, y = sc.fit_transform(X), 2 * y - 1
        return X, y
    elif name == 'Splice':
        path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/splice-junction-gene-sequences/splice.data'
        df = pd.read_csv(path, header=None)
        df.rename(columns={0: 60}, inplace=True)
        df[2] = df[2].map(lambda x: "-".join(list(x.strip())))
        df_new = df[2].str.split('-', expand=True)
        df_new = pd.concat([df_new, df[60]], axis=1)
        if not need_pre:
            return df_new.iloc[:, :-1].values, df_new.iloc[:, -1].values
        else:
            df_new = df_new[df_new[60].isin(['EI', 'IE'])]
            # Categorical to int
            for i in range(61):
                df_new[i] = LabelEncoder().fit_transform(df_new[i])
            X, y = df_new.iloc[:, :-1].values, df_new.iloc[:, -1].values
            X, y = sc.fit_transform(X), 2 * y - 1
            return X, y
    else:
        ValueError("Unknown UCI Dataset.")
