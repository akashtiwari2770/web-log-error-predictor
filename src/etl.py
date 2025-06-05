import os
import re
import pandas as pd

#  Set paths
data_dir = r"C:\Users\Akash tiwari\PycharmProjects\PythonProject\Web_Logs_Error_Prediction_Using_LGBM\Data\logs"
master_file = os.path.join(data_dir, "system_logs.csv")


#  Required columns
expected_columns = [
    'sys-mem-free', 'sys-mem-cache', 'sys-mem-buffered', 'sys-mem-available',
    'sys-mem-total', 'sys-fork-rate', 'sys-interrupt-rate', 'sys-context-switch-rate',
    'sys-thermal', 'disk-io-time', 'disk-bytes-read', 'disk-bytes-written',
    'disk-io-read', 'disk-io-write', 'cpu-iowait', 'cpu-system', 'cpu-user', 'server-up']


#  System file numbers to include
target_numbers = {1, 2, 8, 13, 14, 18, 19}

#  Step 1: Get list of matching system-<number>.csv files
pattern = re.compile(r"system-(\d+)\.csv")
system_files = [
    (int(m.group(1)), f)
    for f in os.listdir(data_dir)
    if (m := pattern.match(f)) and int(m.group(1)) in target_numbers
]
system_files.sort(key=lambda x: x[0])  # Sort by number

#  Step 2: Use first valid file for schema
base_number, base_file = system_files[0]
base_path = os.path.join(data_dir, base_file)
master_columns = list(pd.read_csv(base_path).columns) + ["source_file"]

#  Step 3: Read and align files
dataframes = []
for _, file in system_files:
    df = pd.read_csv(os.path.join(data_dir, file))
    df["source_file"] = file
    for col in master_columns:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[master_columns]
    dataframes.append(df)

#  Step 4: Combine with existing master if exists
combined_df = pd.concat(dataframes, ignore_index=True)
if os.path.exists(master_file):
    existing_df = pd.read_csv(master_file)
    combined_df = pd.concat([existing_df, combined_df], ignore_index=True)

#  Step 5: Save to master file
combined_df.to_csv(master_file, index=False)

#  Confirmation
print(f" Master file saved at: {master_file}")
print(f" Appended {len(dataframes)} files: {[f for _, f in system_files]}")
