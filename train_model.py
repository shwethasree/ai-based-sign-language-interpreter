 
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. DATASET PATHS
# ==========================================

PUBLIC_FILE = "data/processed/public_landmarks.csv"
CUSTOM_FILE = "data/collected/gestures.csv"

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "isl_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

EXPECTED_FEATURES = 126

FEATURE_COLUMNS = [
    f"f{i}" for i in range(EXPECTED_FEATURES)
]

EXPECTED_COLUMNS = FEATURE_COLUMNS + ["label"]


# ==========================================
# 2. LOAD AND VALIDATE DATASETS
# ==========================================

def load_dataset(file_path, dataset_name):

    print("\n" + "=" * 55)
    print("LOADING:", dataset_name)
    print("=" * 55)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"{dataset_name} not found: {file_path}"
        )

    if os.path.getsize(file_path) == 0:
        raise ValueError(
            f"{dataset_name} is empty!"
        )

    df = pd.read_csv(file_path)

    print("Dataset shape:", df.shape)

    # Check exact column structure
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"{dataset_name} has incorrect columns.\n"
            f"Expected 126 features named f0-f125 and label.\n"
            f"Found {len(df.columns)} columns."
        )

    # Convert features to numerical values
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Check missing values
    if df.isnull().any().any():
        raise ValueError(
            f"{dataset_name} contains missing or "
            "non-numeric values. Check the CSV."
        )

    # Check infinite values
    if not np.isfinite(
        df[FEATURE_COLUMNS].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"{dataset_name} contains infinite values."
        )

    # Clean labels
    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    if (df["label"] == "").any():
        raise ValueError(
            f"{dataset_name} contains empty labels."
        )

    print("Feature count: 126")
    print("Label column: Available")

    print("\nSign-wise sample count:")
    print(df["label"].value_counts())

    print("\nTotal samples:", len(df))
    print("Total sign categories:", df["label"].nunique())

    return df


# ==========================================
# 3. LOAD BOTH DATASETS
# ==========================================

public_df = load_dataset(
    PUBLIC_FILE,
    "PUBLIC DATASET"
)

custom_df = load_dataset(
    CUSTOM_FILE,
    "CUSTOM DATASET"
)


# ==========================================
# 4. COMBINE BOTH DATASETS
# ==========================================

print("\n" + "=" * 55)
print("COMBINING DATASETS")
print("=" * 55)

df = pd.concat(
    [public_df, custom_df],
    ignore_index=True
)

# Shuffle the combined dataset
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("Public samples:", len(public_df))
print("Custom samples:", len(custom_df))
print("Combined samples:", len(df))

print("\nCombined sign categories:")
print(df["label"].value_counts())

print("\nTotal unique signs:", df["label"].nunique())


# ==========================================
# 5. SEPARATE FEATURES AND LABELS
# ==========================================

X = df[FEATURE_COLUMNS]
y = df["label"]

print("\nFeature matrix shape:", X.shape)
print("Label matrix shape:", y.shape)


# ==========================================
# 6. CHECK SAMPLE COUNTS
# ==========================================

class_counts = y.value_counts()

if (class_counts < 2).any():

    raise ValueError(
        "Every sign must have at least 2 samples "
        "for stratified training and testing.\n"
        + str(class_counts[class_counts < 2])
    )


# ==========================================
# 7. SPLIT TRAINING AND TESTING DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 8. CREATE RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 9. TRAIN THE MODEL
# ==========================================

print("\n" + "=" * 55)
print("TRAINING MODEL")
print("=" * 55)

model.fit(
    X_train,
    y_train
)

print("Model training completed successfully!")


# ==========================================
# 10. EVALUATE THE MODEL
# ==========================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n" + "=" * 55)
print("MODEL EVALUATION")
print("=" * 55)

print(
    f"\nModel Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_
    )
)


# ==========================================
# 11. SAVE TRAINED MODEL
# ==========================================

joblib.dump(
    model,
    MODEL_FILE
)

print("\n" + "=" * 55)
print("MODEL SAVED")
print("=" * 55)

print("Model location:", MODEL_FILE)

print("\nRecognizable signs:")

for sign in model.classes_:
    print("-", sign)

print("\nTotal recognizable signs:", len(model.classes_))

print("\nTRAINING PROCESS COMPLETED!")
