
import sys
sys.path.insert(0, "src")
from sqlalchemy import create_engine, text
from config import DATABASE_URL

tables = [
    "storage_nodes",
    "disks",
    "smart_telemetry",
    "features",
    "risk_scores",
    "model_drift_monitoring"
]

engine = create_engine(DATABASE_URL)

try:
    with engine.connect() as connection:
        print("\nExisting database row counts:\n")
        for table in tables:
            count = connection.execute(
                text(f"SELECT COUNT(*) FROM {table}")
            ).scalar_one()
            print(f"{table}: {count:,}")
finally:
    engine.dispose()
