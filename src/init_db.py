"""
Create (or rebuild) the database tables.

    python src/init_db.py            # create anything that is missing; safe to re-run
    python src/init_db.py --reset    # DELETE every table and start again (asks first)

Before this works you must have created the empty database and user once
(see README.md, "Create the database").
"""

import argparse

import config
from db_utils import get_connection

DB_DIR = config.ROOT / "db"


def run_sql_file(conn, filename: str) -> None:
    conn.execute((DB_DIR / filename).read_text(encoding="utf-8"))
    print(f"  ran db/{filename}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reset", action="store_true", help="drop all project tables first")
    args = ap.parse_args()

    if args.reset:
        answer = input("This DELETES all project tables and data. Type 'yes' to continue: ")
        if answer.strip().lower() != "yes":
            raise SystemExit("Cancelled.")

    with get_connection() as conn:
        if args.reset:
            run_sql_file(conn, "drop_all.sql")
        run_sql_file(conn, "schema.sql")
        run_sql_file(conn, "views.sql")
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' ORDER BY table_name"
        ).fetchall()
    print("Tables now in the database:", ", ".join(t[0] for t in tables))


if __name__ == "__main__":
    main()
