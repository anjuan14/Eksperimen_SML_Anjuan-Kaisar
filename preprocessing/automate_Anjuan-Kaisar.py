import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(path):
    return pd.read_csv(path)

df = load_data("heart_disease_raw.csv")




def handle_missing_values(df):
    """
    Menangani missing values
    - Numerik: median
    - Kategorikal: modus
    """
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include="object").columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    return df


def remove_duplicates(df):
    return df.drop_duplicates()


def encode_categorical(df):
    cat_cols = df.select_dtypes(include="object").columns
    encoder = LabelEncoder()

    for col in cat_cols:
        df[col] = encoder.fit_transform(df[col])

    return df


def standardize_features(df):
    num_cols = df.select_dtypes(include="number").columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df


def binning_features(df):
    df["Age_Bin"] = pd.qcut(
        df["Age"],
        q=3,
        labels=["Low", "Medium", "High"]
    )

    df["Age_Bin"] = df["Age_Bin"].map({
        "Low": 0,
        "Medium": 1,
        "High": 2
    })

    return df


def preprocess_pipeline(input_path, output_path):
    df = load_data(input_path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = encode_categorical(df)
    df = standardize_features(df)
    df = binning_features(df)

    df.to_csv(output_path, index=False)
    print(f"Dataset preprocessed disimpan di: {output_path}")


if __name__ == "__main__":
    preprocess_pipeline(
        input_path="heart_disease_raw.csv",
        output_path="preprocessing/heart_disease_preprocessed_automate.csv"
    )
