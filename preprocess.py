import pandas as pd
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = "2022_public_lar_csv.csv"
COLS = [
    "action_taken",
    "applicant_age",
    "combined_loan_to_value_ratio",
    "debt_to_income_ratio",
    "derived_ethnicity",
    "derived_race",
    "derived_sex",
    "income",
    "interest_rate",
    "loan_amount",
    "loan_purpose",
    "loan_term",
    "loan_type",
    "occupancy_type",
    "property_value",
]
CATEG_COLS = [
    "action_taken",
    "applicant_age",
    "debt_to_income_ratio",
    "derived_ethnicity",
    "derived_race",
    "derived_sex",
    "loan_purpose",
    "loan_type",
    "occupancy_type",
]


def preprocess(df, col):
    """Preprocess a column in the DataFrame."""

    if col == "action_taken":
        # Transform (1, 2, 8) -> 0 and (3, 7) -> 1.
        df = df[df[col].isin([1, 2, 3, 7, 8])]
        df.loc[:, col] = df[col].replace({1: 0, 2: 0, 8: 0, 3: 1, 7: 1})
    elif col == "debt_to_income_ratio":
        # Replace "Exempt" and NaN with mode.
        df.loc[:, col] = df[col].replace("Exempt", None)
        mode = df[col].mode().values[0]
        df.loc[:, col] = df[col].fillna(mode)
    elif col in [
        "combined_loan_to_value_ratio",
        "income",
        "interest_rate",
        "loan_term",
        "property_value",
    ]:
        # Replace "Exempt" and NaN with median.
        df.loc[:, col] = df[col].replace("Exempt", None).astype(float)
        median = df[col].median()
        df.loc[:, col] = df[col].astype(float).fillna(median)
    return df


def main():
    """Preprocess the data and write to a new CSV file."""

    df = pd.read_csv(DATA_PATH)
    df = df[df["state_code"] == "MA"][COLS]

    # Preprocess each column.
    for col in COLS:
        df = preprocess(df, col)

    # Perform one-hot encoding on categorical columns.
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(df[CATEG_COLS])
    one_hot_df = pd.DataFrame(
        one_hot_encoded,
        columns=encoder.get_feature_names_out(CATEG_COLS),
    )
    df = pd.concat([df, one_hot_df], axis=1)
    df = df.drop(columns=CATEG_COLS, axis=1)

    # Write to a new CSV file.
    df.to_csv("data.csv", index=False)


if __name__ == "__main__":
    main()
