import pandas as pd

file_path = "data/ai4i2020.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET SHAPE ==========")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n========== COLUMN NAMES ==========")
for column in df.columns:
    print(column)

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== MACHINE FAILURE DISTRIBUTION ==========")
if "Machine failure" in df.columns:
    print(df["Machine failure"].value_counts())

print("\n========== DATASET INFORMATION ==========")
print(df.info())