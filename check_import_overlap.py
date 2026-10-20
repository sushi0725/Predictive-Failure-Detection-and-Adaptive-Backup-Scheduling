
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, "src")
from sqlalchemy import create_engine, text
from config import DATABASE_URL
DATA_DIR = Path("data/raw")
engine = create_engine(DATABASE_URL)
try:
    with engine.connect() as conn:
        db_dates = {
            row[0].isoformat()
            for row in conn.execute(
                text("SELECT DISTINCT date FROM smart_telemetry WHERE date IS NOT NULL")
            )
        }
    csv_dates = set()
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    print(f"CSV files found: {len(csv_files)}")
    for file in csv_files:
        try:
            dates = pd.read_csv(file, usecols=["date"])["date"].dropna().astype(str)
            csv_dates.update(d[:10] for d in dates)
        except Exception as exc:
            print(f"Could not inspect {file.name}: {exc}")
    print(f"\nCSV date range: {min(csv_dates) if csv_dates else 'none'} to {max(csv_dates) if csv_dates else 'none'}")
    print(f"Distinct CSV dates: {len(csv_dates)}")
    print(f"Distinct database dates: {len(db_dates)}")
    print(f"Dates present in both: {len(csv_dates & db_dates)}")
    print(f"New dates not in database: {len(csv_dates - db_dates)}")
    print(f"Database dates absent from CSVs: {sorted(db_dates - csv_dates)}")
finally:
    engine.dispose()
