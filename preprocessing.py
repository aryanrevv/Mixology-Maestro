"""
preprocessing.py

A small preprocessing pipeline for the cocktail dataset (data.csv).

Steps:
1. Load the raw data
2. Clean it (normalize text casing, drop duplicates, report missing values)
3. Impute missing values and scale numeric features
4. One-hot encode categorical features
5. Split into train/test sets and save the processed output
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

RAW_PATH = "data.csv"
TRAIN_OUTPUT_PATH = "train_processed.csv"
TEST_OUTPUT_PATH = "test_processed.csv"

NUMERIC_FEATURES = ["abv", "rating", "price_usd", "ingredient_count", "prep_time_min"]
CATEGORICAL_FEATURES = ["category", "base_spirit"]
ID_COLUMN = "cocktail_name"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns from {path}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize text casing/whitespace before de-duplicating, so
    # e.g. "whiskey" and "Whiskey" are treated as the same value.
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip().str.title()

    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate row(s)")

    missing = df.isna().sum()
    print("Missing values per column:")
    print(missing[missing > 0].to_string() if missing.any() else "  none")

    return df.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def to_dense(array):
    return array.toarray() if hasattr(array, "toarray") else array


def main():
    df = load_data(RAW_PATH)
    df = clean_data(df)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    ids = df[[ID_COLUMN]]

    X_train, X_test, ids_train, ids_test = train_test_split(
        X, ids, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    train_df = pd.DataFrame(to_dense(X_train_processed), columns=feature_names, index=ids_train.index)
    test_df = pd.DataFrame(to_dense(X_test_processed), columns=feature_names, index=ids_test.index)

    train_df = pd.concat([ids_train, train_df], axis=1)
    test_df = pd.concat([ids_test, test_df], axis=1)

    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
    test_df.to_csv(TEST_OUTPUT_PATH, index=False)

    print(f"Train set processed: {train_df.shape} -> {TRAIN_OUTPUT_PATH}")
    print(f"Test set processed:  {test_df.shape} -> {TEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
