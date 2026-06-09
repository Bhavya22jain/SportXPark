import glob
import os
import csv

nba_dir = "data/nba"
for f in glob.glob(os.path.join(nba_dir, "*.csv")):
    print(f"\nFile: {os.path.basename(f)}")
    try:
        with open(f, mode='r', encoding='utf-8') as fh:
            reader = csv.reader(fh)
            header = next(reader)
            print(f"Columns: {header}")
            row1 = next(reader, None)
            row2 = next(reader, None)
            print("Row 1:", row1)
            print("Row 2:", row2)
    except Exception as e:
        print(f"Error reading: {e}")
