import pandas as pd
import argparse
import yaml
import numpy as np
from utils import clip_scaling
from sklearn.model_selection import train_test_split

def extract_NA(df):
    nonNA = df.dropna(axis=0)
    NA = df[df.isna().any(axis=1)]
    return nonNA, NA

def main(config):
    df = pd.read_csv(config["original_file"], header=None)
    if config['exists_NA']:
        df = df.applymap(lambda x: np.NaN if x == config["NA_form"] else x)
    df, scaler = clip_scaling(df, config["param"]["numerical_column"], n=4)
    df.iloc[:, config["label_column"]] = df.iloc[:, config["label_column"]].map(
        lambda x: 0 if x == ' <=50K' else 1)
    if config['exists_NA']:
        df, NA = extract_NA(df)
        NA.to_csv(config["original_file"].split('.')[0] + "_NA.csv", header=None, index=False)
    df_train, df_test, _, _ = train_test_split(
        df.values.tolist(), np.ones(
            df.shape[0]), test_size=0.2, random_state=42)
    pd.DataFrame(df_train).to_csv(config["original_file"].split('.')[0] + "_train.csv", header=None, index=False)
    pd.DataFrame(df_test).to_csv(config["original_file"].split('.')[0] + "_test.csv", header=None, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="configuration yaml file for data preprocessing")
    args = parser.parse_args()
    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    main(config)
