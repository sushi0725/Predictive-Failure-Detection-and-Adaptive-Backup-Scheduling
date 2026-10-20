import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://backup_user:change_me@localhost:5432/backup_db")
HORIZON_DAYS = int(os.getenv("HORIZON_DAYS", 7))

# FIXED: Added the missing configuration variable tracking parameter
DASHBOARD_DEMO = int(os.getenv("DASHBOARD_DEMO", 0))

# Feature profiles mapped to database table fields
# Core storage target metrics are logged inside the 'smart_telemetry' table
FEATURE_COLUMNS = ['smart_1_raw', 'smart_5_raw', 'smart_9_raw', 'smart_187_raw', 'smart_197_raw']
