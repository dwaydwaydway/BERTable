import random
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import RobustScaler, MaxAbsScaler

import sys
sys.path.append('../..')
import utils


def main():
    cloumn_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                    'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
    numerical_column = ['age', 'fnlwgt', 'education_num',
                        'capital_gain', 'capital_loss', 'hours_per_week']
    scategorical_column = ['workclass', 'education', 'marital_status',
                           'occupation', 'relationship', 'race', 'sex', 'native_country']
    NA_form = ' ?'
    label_column = 'income'
    label_mapping = {' <=50K': 0, ' >50K': 1}

    df = pd.read_csv('/home/user/BERTable/datasets/adult/data/adult.data', header=None, names=cloumn_names)


    df = df.drop(columns=[label_column])
    df = df.applymap(lambda x: np.NaN if x == NA_form else x)
    # scaling & clip outliers
    for col in numerical_column:
        if col in {'capital_gain', 'capital_loss'}:
            scaler = MaxAbsScaler()
        else:
            scaler = RobustScaler()

        transformer = scaler.fit(df[col].to_numpy().reshape(-1, 1))
        df[col] = transformer.transform(df[col].to_numpy().reshape(-1, 1))
    df.to_csv('data/adult.csv', index=None, header=None)


if __name__ == "__main__":
    main()
