import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def extract_NA(df):
    nonNA = df.dropna(axis=0)
    NA = df[df.isna().any(axis=1)]
    return nonNA, NA

# Clip numerical data to [mean - n * std, mean + n * std] and min-max
# scale to [0, 1]


def clip_scaling(df, numerical_column, n=3, scaler=None):
    df_clip_scaling = pd.DataFrame()
    for column in df.columns:
        if column in numerical_column:
            df_clip_scaling[column] = df[column].clip(df[column].mean(
            ) - n * df[column].std(), df[column].mean() + n * df[column].std())
        else:
            df_clip_scaling[column] = df[column]
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(df_clip_scaling[numerical_column])
        df_clip_scaling[numerical_column] = scaler.transform(
            df[numerical_column])
        return df_clip_scaling, scaler
    else:
        df_clip_scaling[numerical_column] = scaler.transform(
            df_clip_scaling[numerical_column])
        return df_clip_scaling


def train_test_split(df, label_col, test_ratio=0.2):
    # Shuffle
    df = df.sample(frac=1)
    # Split into training/testing set
    df_train = df[int(len(df) * test_ratio):]
    df_test = df[:int(len(df) * test_ratio)]
    # Extract label column
    df_train_label, df_test_label = df_train[label_col], df_test[label_col]
    # Drop label column
    df_train = df_train.drop(columns=[label_col])
    df_test = df_test.drop(columns=[label_col])

    return (df_train, df_train_label), (df_test, df_test_label)
