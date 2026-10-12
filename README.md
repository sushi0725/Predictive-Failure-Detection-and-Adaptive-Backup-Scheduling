# Predictive Failure Detection & Adaptive Backup Scheduling

DBMS course project (BACSE202). The system watches hard-drive health counters, predicts which drives
will fail soon, and tightens their backup schedule in proportion to the risk instead of using one fixed schedule.

```
Backblaze CSVs --(Step 1)--> PostgreSQL --(Step 2)--> features --(Steps 3,5: ML model)--> risk_scores
                                                                                              |
                                    dashboard  <--(Step 7)--  backup_events  <--(Step 4)-- decision_layer
```

## What is in this folder

| Path | What it is | Owner |
| --- | --- | --- |
| `db/schema.sql`, `db/views.sql` | The tables and views | Data/DB |
| `db/queries.sql` | Ready-made queries for exploring and for the report | Data/DB |
| `src/load_backblaze.py` | Step 1: load the raw CSVs into PostgreSQL | Data/DB |
| `src/build_features.py` | Step 2: clean, window and label the data | Data/DB |
| `src/dataset.py` | Loads training data and does the correct time-based split | Data/DB, used by ML |
| `src/db_utils.py` | `write_risk_scores()` (ML writes here), `load_latest_risk_scores()` (dashboard reads here) | Data/DB |
| `src/demo_score_drives.py` | **Placeholder** model so the chain works before the real model exists | replace |
| `src/decision_layer.py` | Step 4: risk score to backup action (rules) | Decision/dashboard |
| `src/run_decision_cycle.py` | Runs the decision layer on the database and logs `backup_events` | Data/DB |
| `src/dashboard.py` | Step 7: Streamlit dashboard | Decision/dashboard |
| `tools/make_fixture_data.py` | Makes a small FAKE dataset for testing | Data/DB |
| `tests/` | Tests for the feature maths (no database needed) | Data/DB |
| `docs/` | ER diagram, team contract | everyone |

Full explanations: `docs/TEAM_CONTRACT.md` and `docs/ER_DIAGRAM.md`.

## One-time setup (about 15 minutes)

1. Install Python 3.13, PostgreSQL 18 (it includes pgAdmin and psql) and Git.
2. In this folder, create the virtual environment and install the packages:

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows PowerShell:  .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Create the empty database and user. Open **SQL Shell (psql)** from the Windows Start menu
   (or `psql -U postgres -h localhost` elsewhere), log in as `postgres`, then run:

   ```sql
   CREATE USER backup_user WITH PASSWORD 'change_me';
   CREATE DATABASE backup_db OWNER backup_user;
   ```

4. Copy `.env.example` to `.env` (same folder) and put your password in it. Never commit `.env`.
5. Create the tables:

   ```bash
   python src/init_db.py
   ```

## Try everything on FAKE data first (2 minutes)

```bash
python tools/make_fixture_data.py                     # fake Backblaze-style zip in data/raw/
python src/load_backblaze.py --data data/raw/fixture_Q1_2026.zip --healthy-sample 1
python src/build_features.py
python src/demo_score_drives.py                       # placeholder model -> risk_scores
python src/run_decision_cycle.py                      # decisions -> backup_events
streamlit run src/dashboard.py                        # opens in your browser
pytest                                                # feature-maths tests
```

If the dashboard shows a yellow "DEMO DATA" warning, it could not read the database (check `.env`).
Fake data proves the plumbing works. **Never quote numbers from it in the report.**

## Real data

1. Download `data_Q1_2026.zip` from <https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data>
   into `data/raw/` (do not unzip it).
2. Trial run on the newest day only, to catch any column surprise cheaply:

   ```bash
   python src/load_backblaze.py --data data/raw/data_Q1_2026.zip --max-files 1
   ```

3. Full quarter (keeps every failed drive + 2% of healthy ones; takes a while):

   ```bash
   python src/init_db.py --reset         # wipe the fake data first
   python src/load_backblaze.py --data data/raw/data_Q1_2026.zip
   python src/build_features.py
   ```

Adding a second quarter later (needed for the drift / "accuracy decay" experiments): always load the
**newest quarter first**, then the older one, so drives that fail in the newer quarter keep their older history.
If Q1 is already loaded, run `python src/init_db.py --reset`, then load Q2, then Q1, then `build_features.py` again.

Why a sample? A quarter has ~31 million rows. Because healthy drives are down-sampled, failures are far more
common in our table than in reality, so model scores are **not true probabilities**. State this in the report.

## Sharing the database with teammates

The database itself is not in git (too big). Either each person runs the loader, or share a dump file
through Google Drive / OneDrive:

```bash
pg_dump -U backup_user -h localhost -Fc backup_db > backup_db.dump           # export
pg_restore -U backup_user -h localhost -d backup_db --clean backup_db.dump   # import (after init_db.py)
```

## Common errors

| Message | Cause and fix |
| --- | --- |
| `password authentication failed` | The password in `.env` does not match the one you set in `CREATE USER`. |
| `connection refused` | PostgreSQL is not running (start the service) or uses another port. |
| `relation "drives" does not exist` | Run `python src/init_db.py`. |
| `ModuleNotFoundError: No module named ...` | The virtual environment is not active; activate it, then `pip install -r requirements.txt`. |
| `CSV is missing expected columns` | The file layout differs from what the loader expects; send the first line of the CSV to the team. |
| `'psql' is not recognized` (Windows) | Use **SQL Shell (psql)** from the Start menu, or pgAdmin's Query Tool. |

## Team rules

- Work on your own branch (`feature/<name>`), open a Pull Request, merge into `main` after one teammate looks at it.
- Never commit `.env`, data files, `.venv`, or `__pycache__`/`*.pyc` (the `.gitignore` already blocks them).
- If you change a table, change `db/schema.sql` **and** tell the team (see `docs/TEAM_CONTRACT.md`).
