import os
import sys
import pandas as pd
import numpy as np
import psycopg
from sqlalchemy import create_engine
from config import DATABASE_URL, FEATURE_COLUMNS

def get_engine():
    return create_engine(DATABASE_URL)

def read_telemetry():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM smart_telemetry;", engine)
    return df

def build_features_matrix(df):
    if df.empty:
        return pd.DataFrame()
    print("[-] Compiling sliding temporal lifecycle indicators feature matrices...")
    
    df = df.sort_values(by=['disk_id', 'timestamp'])
    df['feature_date'] = df['timestamp']
    
    df['hours_in_service'] = df['smart_9_raw']
    df['temperature_c'] = df['smart_194_raw'] if 'smart_194_raw' in df.columns else 35
    
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[f"{col}_delta"] = df.groupby('disk_id')[col].diff().fillna(0)
            
    return df

def write_features(df):
    if df.empty:
        return
    
    psycopg_dsn = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")
    with psycopg.connect(psycopg_dsn) as conn:
        with conn.cursor() as cur:
            # Dynamically fetch destination columns to avoid UndefinedColumn mismatches
            cur.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'features' AND table_schema = 'public';
            """)
            db_cols = [row[0] for row in cur.fetchall() if row[0] != 'feature_id']
            
            # Keep only the columns present on both sides
            matched_cols = [c for c in df.columns if c in db_cols]
            
            # Clean floating numbers to integers for strict database columns (e.g. converting 0.0 to 0)
            for c in matched_cols:
                if 'raw' in c or 'id' in c or 'failure' in c:
                    try:
                        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(np.int64)
                    except Exception:
                        pass

            cleaned_df = df[matched_cols].copy()
            
            print(f"[-] Streaming {len(cleaned_df):,} feature records into destination schemas...")
            cur.execute("TRUNCATE TABLE features RESTART IDENTITY CASCADE;")
            
            # FIX: Use FORMAT text and DELIMITER tab to avoid column shifting errors
            with cur.copy(f"COPY features ({', '.join(matched_cols)}) FROM STDIN WITH (FORMAT text, DELIMITER E'\\t', NULL 'nan')") as copy:
                for row in cleaned_df.itertuples(index=False):
                    row_strings = [str(x).replace('\t', ' ').replace('\n', ' ') for x in row]
                    copy.write_row(row_strings)
            conn.commit()

def main():
    print("\n================ STARTING FEATURE ENGINEERING PIPELINE ==================")
    try:
        raw_df = read_telemetry()
        print(f"[+] Loaded {len(raw_df):,} raw telemetry log records from database storage views.")
        
        feats_df = build_features_matrix(raw_df)
        write_features(feats_df)
        print("[+] Feature engineering module completed successfully. Pipeline aligned.\n")
    except Exception as e:
        print(f"[!] Critical feature processing failure: {e}")
        raise e

if __name__ == "__main__":
    main()
