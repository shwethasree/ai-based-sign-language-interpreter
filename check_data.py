
import pandas as pd
import os

# Check current working directory
print("Current folder:", os.getcwd())

# Find gestures.csv inside the project
csv_file = None

for root, dirs, files in os.walk("."):
    if "gestures.csv" in files:
        csv_file = os.path.join(root, "gestures.csv")
        break

if csv_file is None:
    print("ERROR: gestures.csv not found!")
else:
    print("CSV found at:", csv_file)

    # Read dataset
    df = pd.read_csv(csv_file)

    print("\nDataset shape:", df.shape)

    print("\nSign-wise sample count:")
    print(df["label"].value_counts())

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nFirst five rows:")
    print(df.head())

import os
import pandas as pd
import numpy as np

# ==========================================
# 1. DATASET PATHS
# ==========================================

PUBLIC_FILE = "data/processed/public_landmarks.csv"
CUSTOM_FILE = "data/collected/gestures.csv"

EXPECTED_FEATURES = 126
EXPECTED_COLUMNS = 127

# ==========================================
# 2. FUNCTION TO CHECK ONE DATASET
# ==========================================

def check_dataset(file_path, dataset_name):

    print("\n" + "=" * 55)
    print("CHECKING:", dataset_name)
    print("=" * 55)

    # Check file existence
    if not os.path.exists(file_path):

        print("ERROR: File does not exist!")
        print("Expected location:", file_path)
        return None

    if os.path.getsize(file_path) == 0:

        print("ERROR: File is empty!")
        return None

    print("File exists: YES")

    # Load CSV
    df = pd.read_csv(file_path)

    print("\nDataset shape:", df.shape)

    # ======================================
    # 3. CHECK COLUMNS
    # ======================================

    print("\nTotal columns:", len(df.columns))

    if "label" not in df.columns:

        print("ERROR: Label column missing!")
        return None

    feature_columns = [
        col for col in df.columns
        if col != "label"
    ]

    print("Feature columns:", len(feature_columns))

    if len(feature_columns) == EXPECTED_FEATURES:

        print("Feature count: CORRECT (126)")

    else:

        print(
            "WARNING: Expected 126 features, "
            "but found", len(feature_columns)
        )

    # ======================================
    # 4. CHECK FEATURE NAMES
    # ======================================

    expected_names = [
        f"f{i}" for i in range(126)
    ]

    if feature_columns == expected_names:

        print("Feature naming: CORRECT")

    else:

        print("WARNING: Feature naming is different.")

        print("First columns:", feature_columns[:10])

    # ======================================
    # 5. CHECK LABELS
    # ======================================

    print("\nUnique signs:")

    print(df["label"].value_counts())

    print("\nTotal sign categories:")

    print(df["label"].nunique())

    # Check empty labels
    empty_labels = df["label"].isna().sum()

    print("\nMissing labels:", empty_labels)

    # ======================================
    # 6. CHECK MISSING VALUES
    # ======================================

    print("\nMissing values:", df.isnull().sum().sum())

    # ======================================
    # 7. CHECK NUMERICAL VALUES
    # ======================================

    numeric_data = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    invalid_values = numeric_data.isna().sum().sum()

    print("Non-numeric or missing features:", invalid_values)

    finite_values = np.isfinite(
        numeric_data.to_numpy(dtype=float)
    )

    infinite_values = np.size(finite_values) - np.count_nonzero(
        finite_values
    )

    print("Infinite or invalid numerical values:", infinite_values)

    # ======================================
    # 8. CHECK DUPLICATE ROWS
    # ======================================

    print("\nDuplicate rows:", df.duplicated().sum())

    # ======================================
    # 9. DISPLAY FIRST FIVE ROWS
    # ======================================

    print("\nFirst five rows:")

    print(df.head())

    print("\nDataset checking completed.")

    return df


# ==========================================
# 10. CHECK BOTH DATASETS
# ==========================================

public_df = check_dataset(
    PUBLIC_FILE,
    "PUBLIC DATASET"
)

custom_df = check_dataset(
    CUSTOM_FILE,
    "CUSTOM DATASET"
)

# ==========================================
# 11. COMPARE BOTH DATASETS
# ==========================================

print("\n" + "=" * 55)
print("DATASET COMPARISON")
print("=" * 55)

if public_df is not None and custom_df is not None:

    public_features = [
        c for c in public_df.columns if c != "label"
    ]

    custom_features = [
        c for c in custom_df.columns if c != "label"
    ]

    if public_features == custom_features:

        print("Feature column names: MATCH")

    else:

        print("Feature column names: DO NOT MATCH")

    print("\nPublic dataset samples:", len(public_df))
    print("Custom dataset samples:", len(custom_df))

    print(
        "\nPublic feature count:",
        len(public_features)
    )

    print(
        "Custom feature count:",
        len(custom_features)
    )

    print("\nNOTE:")
    print(
        "Matching column names alone do not prove "
        "that landmark extraction and normalization "
        "are compatible."
    )

else:

    print("Both datasets must exist before comparison.")

print("\nCHECKING FINISHED!")
