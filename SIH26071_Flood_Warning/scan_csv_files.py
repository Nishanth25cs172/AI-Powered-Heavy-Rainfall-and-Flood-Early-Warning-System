import os
import pandas as pd

root_dir = r"c:\Users\kpoob\OneDrive\Documents\AIML\SIH26071_Flood_Warning_Project"

csv_files = []
for dirpath, _, filenames in os.walk(root_dir):
    for f in filenames:
        if f.lower().endswith('.csv'):
            csv_files.append(os.path.join(dirpath, f))

print(f"Total CSV files found: {len(csv_files)}\n")

for idx, file_path in enumerate(csv_files, 1):
    filename = os.path.basename(file_path)
    print("=" * 70)
    print(f"FILE #{idx}")
    print(f"Exact Path: {file_path}")
    print(f"Filename:   {filename}")
    try:
        df = pd.read_csv(file_path)
        print(f"Rows:       {len(df):,}")
        print(f"Columns:    {len(df.columns)}")
        print("Column Names:")
        for col_idx, col_name in enumerate(df.columns, 1):
            print(f"  {col_idx}. {col_name}")
    except Exception as e:
        print(f"Error reading file: {e}")
    print("=" * 70)
    print()
