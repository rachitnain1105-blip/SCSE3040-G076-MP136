
from pathlib import Path
import pandas as pd

# Project root directory
ROOT = Path(__file__).resolve().parents[1]

# Input and output paths
DATA_PATH = ROOT / "data" / "raw" / "train.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "train_clean.csv"


def load_data():
    """Load the original training dataset."""
    df = pd.read_csv(DATA_PATH)

    print("\n========== DATASET OVERVIEW ==========")
    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nFirst 5 Rows:")
    print(df.head())

    return df


def inspect_data(df):
    """Inspect important data issues."""

    print("\n========== DATA INSPECTION ==========")

    if "days_from_tracking_page" in df.columns:
        print("\nPotential leakage column detected:")
        print("days_from_tracking_page")

    print("\nWeight by destination:")
    print(df.groupby("destination")["weight_kg"].describe())

    print("\nTransfers value counts:")
    print(df["transfers"].value_counts().sort_index())

    print("\nService categories:")
    print(df["service"].value_counts())

    print("\nDestination categories:")
    print(df["destination"].value_counts())


def clean_data(df):
    """Clean the training dataset."""

    cleaned_df = df.copy()

    print("\n========== DATA CLEANING ==========")
    print("Original rows:", len(cleaned_df))

    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()

    # Remove the target leakage column
    if "days_from_tracking_page" in cleaned_df.columns:
        cleaned_df = cleaned_df.drop(
            columns=["days_from_tracking_page"]
        )

    # Correct weight unit mix-up for Tier-2
    tier2_mask = cleaned_df["destination"] == "Tier-2"
    cleaned_df.loc[tier2_mask, "weight_kg"] = (
        cleaned_df.loc[tier2_mask, "weight_kg"] / 1000
    )

    # Remove unnecessary spaces from text columns
    for column in ["service", "destination"]:
        cleaned_df[column] = cleaned_df[column].astype(str).str.strip()

    # Convert date column into datetime
    cleaned_df["date"] = pd.to_datetime(
        cleaned_df["date"], errors="coerce"
    )

    # Sort records by date
    cleaned_df = cleaned_df.sort_values("date")

    print("\nCleaning Summary:")
    print("Original rows:", len(df))
    print("Cleaned rows:", len(cleaned_df))
    print("Remaining duplicates:", cleaned_df.duplicated().sum())
    print("Remaining columns:", cleaned_df.columns.tolist())

    return cleaned_df


if __name__ == "__main__":

    # Load dataset
    original_df = load_data()

    # Inspect dataset
    inspect_data(original_df)

    # Clean dataset
    cleaned_df = clean_data(original_df)

    # Create processed folder if it does not exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save cleaned dataset
    cleaned_df.to_csv(OUTPUT_PATH, index=False)

    print("\n========== OUTPUT ==========")
    print("Cleaned dataset saved successfully!")
    print("Location:", OUTPUT_PATH)
    print("Final shape:", cleaned_df.shape)
