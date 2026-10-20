
from pathlib import Path
import pandas as pd

folder = Path("data/raw")
files = sorted(folder.glob("*.csv"))

print("CSV files found:", len(files))
print("\nFiles and date coverage:")

all_dates = []

for file in files:
    try:
        df = pd.read_csv(file, usecols=["date"], low_memory=False)
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if not dates.empty:
            first_date = dates.min().date()
            last_date = dates.max().date()
            all_dates.extend(dates.dt.normalize().tolist())
            print(f"{file.name}: {first_date} to {last_date}")
        else:
            print(f"{file.name}: No valid dates found")
    except Exception as e:
        print(f"{file.name}: ERROR - {e}")

if all_dates:
    print("\nTotal distinct dates:", len(set(all_dates)))
    print("Earliest date:", min(all_dates).date())
    print("Latest date:", max(all_dates).date())

print("\nCheck complete.")
