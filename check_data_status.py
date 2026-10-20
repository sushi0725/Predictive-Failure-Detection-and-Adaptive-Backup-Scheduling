
import sys
sys.path.insert(0, "src")
from sqlalchemy import create_engine, text
from config import DATABASE_URL

engine = create_engine(DATABASE_URL)

queries = {
    "Telemetry date coverage": """
        SELECT MIN(date), MAX(date), COUNT(DISTINCT date)
        FROM smart_telemetry
    """,
    "Telemetry failure labels": """
        SELECT failure, COUNT(*)
        FROM smart_telemetry
        GROUP BY failure
        ORDER BY failure
    """,
    "Feature label distribution": """
        SELECT fails_within_horizon, COUNT(*)
        FROM features
        GROUP BY fails_within_horizon
        ORDER BY fails_within_horizon
    """,
    "Risk score date coverage": """
        SELECT MIN(timestamp), MAX(timestamp), COUNT(*)
        FROM risk_scores
    """
}

try:
    with engine.connect() as connection:
        for title, query in queries.items():
            print(f"\n--- {title} ---")
            result = connection.execute(text(query))
            for row in result:
                print(row)
finally:
    engine.dispose()
