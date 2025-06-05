import os
import re
import pandas as pd

# Load the CSV
df = pd.read_csv(r"C:\Users\Akash tiwari\PycharmProjects\PythonProject\Web_Logs_Error_Prediction_Using_LGBM\Data\Logs\system-11.csv")

# Separate errors and non-errors
error_df = df[df['server-up'] == 1]
non_error_df = df[df['server-up'] == 2]

# Number of errors
n_errors = len(error_df)

# Select remaining rows from non-errors to make total = 30
n_non_errors = 30 - n_errors

# Sample non-errors randomly
non_error_sample = non_error_df.sample(n=n_non_errors, random_state=42)

# Combine and shuffle
sample_df = pd.concat([error_df, non_error_sample]).sample(frac=1, random_state=42)

# Preview or save
print(sample_df.head())
sample_df.to_csv(r"C:\Users\Akash tiwari\PycharmProjects\PythonProject\Web_Logs_Error_Prediction_Using_LGBM\Data\sampled_30_rows.csv", index=False)