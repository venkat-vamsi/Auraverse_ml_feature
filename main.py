import os
import pandas as pd

from stage1_panic_detection.stage1 import train_stage1, test_stage1
from stage2_cause_identification.stage2 import train_stage2, test_stage2


TRAIN_FOLDER = "data/train"
TEST_FOLDER = "data/test"
OUTPUT_FOLDER = "data/output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def load_csv_folder(folder_path):
    dfs = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            full_path = os.path.join(folder_path, file)
            dfs.append(pd.read_csv(full_path))
    return pd.concat(dfs, ignore_index=True)


def run_pipeline():
    print("\n========== AURAVERSE PANIC ML MODEL ==========\n")

    # -------------------------------
    # Load data based on folder split
    # -------------------------------
    train_df = load_csv_folder(TRAIN_FOLDER)
    test_df = load_csv_folder(TEST_FOLDER)

    print("Train rows:", len(train_df))
    print("Test rows:", len(test_df))

    # -------------------------------
    # Stage 1 — Panic Detection
    # -------------------------------
    print("\n🔹 Training Stage 1 (Isolation Forest)")
    model1, scaler1 = train_stage1(train_df)

    print("🔹 Testing Stage 1")
    train_panic = test_stage1(train_df, model1, scaler1)
    test_panic = test_stage1(test_df, model1, scaler1)

    print("Train panic rows:", len(train_panic))
    print("Test panic rows:", len(test_panic))
    print("Test panic %:", round((len(test_panic) / len(test_df)) * 100, 2), "%")

    # -------------------------------
    # Stage 2 — Cause Identification
    # -------------------------------
    print("\n🔹 Training Stage 2 (KMeans Cause Identification)")
    model2, scaler2 = train_stage2(train_panic)

    print("🔹 Testing Stage 2")
    final_test = test_stage2(
        test_panic,
        model2,
        scaler2,
        save_path="data/output/test_panic_with_cause.csv"
    )

    print("\nTest Cause Distribution:")
    print(final_test["cause_name"].value_counts())

    print("\nPipeline Complete.\n")


if __name__ == "__main__":
    run_pipeline()