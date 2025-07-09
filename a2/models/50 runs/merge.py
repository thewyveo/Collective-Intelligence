import pandas as pd
import os

# Folder containing the CSV files
folder_path = "z50s"  # Change this

# Optional: filter only .csv files
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# List to hold DataFrames
df_list = []

# Loop through and read each CSV
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    df["source_file"] = file  # Optional: tag origin
    df_list.append(df)

# Concatenate all into one DataFrame
merged_df = pd.concat(df_list, ignore_index=True)

# Save if needed
merged_df.to_csv("merged_output.csv", index=False)

print(f"✅ Merged {len(csv_files)} CSV files into 'merged_output.csv'")