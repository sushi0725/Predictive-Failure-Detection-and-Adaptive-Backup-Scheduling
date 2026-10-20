
import sys
sys.path.insert(0, "src")
from sqlalchemy import create_engine, text
from config import DATABASE_URL

engine = create_engine(DATABASE_URL)

query = text("""
SELECT
    tc.table_name,
    kcu.column_name,
    ccu.column_name AS referenced_column
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
    AND tc.constraint_schema = kcu.constraint_schema
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
    AND ccu.constraint_schema = tc.constraint_schema
WHERE tc.constraint_type = 'FOREIGN KEY'
  AND ccu.table_name = 'features'
ORDER BY tc.table_name
""")

try:
    with engine.connect() as connection:
        rows = connection.execute(query).fetchall()
        print("Foreign keys referencing features:")
        if rows:
            for row in rows:
                print(row)
        else:
            print("No foreign keys reference features.")
finally:
    engine.dispose()
