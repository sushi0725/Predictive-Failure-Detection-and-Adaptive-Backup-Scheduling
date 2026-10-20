
import sys
sys.path.insert(0, "src")
import psycopg
from config import DATABASE_URL

dsn = DATABASE_URL.replace(
    "postgresql+psycopg://",
    "postgresql://",
    1
)

try:
    with psycopg.connect(dsn) as conn:
        print("PostgreSQL connection successful!")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            tables = cur.fetchall()
            print("\nDatabase tables:")
            if tables:
                for table in tables:
                    print("-", table[0])
            else:
                print("No tables found in the public schema.")
except Exception as e:
    print("Database connection failed.")
    print("Error:", e)
