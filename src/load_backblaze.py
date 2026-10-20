import os
import sys
import glob
import argparse
import pandas as pd
import psycopg
from config import DATABASE_URL

def load_raw_csv_files(cur, data_dir, sample_rate):
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        print(f"[!] Warning: No raw CSV telemetry log files located in target path directory: {data_dir}")
        return
    
    print(f"[-] Located {len(csv_files)} logs inside system storage. Initializing data streaming pipelines...")
    
    for file_idx, file in enumerate(csv_files):
        print(f"[-] Processing log window stream target file [{file_idx + 1}/{len(csv_files)}]: {os.path.basename(file)}")
        try:
            with open(file, 'r', encoding='utf-8') as f:
                first_line = f.readline()
            delim = '\t' if '\t' in first_line or '   ' in first_line else ','
            
            df = pd.read_csv(file, delimiter=delim)
            df = df.dropna(subset=['serial_number', 'date'])
            
            all_smart_cols = [
                'smart_1_raw', 'smart_3_raw', 'smart_4_raw', 'smart_5_raw', 'smart_7_raw', 
                'smart_9_raw', 'smart_10_raw', 'smart_12_raw', 'smart_187_raw', 'smart_188_raw', 
                'smart_189_raw', 'smart_190_raw', 'smart_192_raw', 'smart_193_raw', 'smart_194_raw', 
                'smart_197_raw', 'smart_198_raw', 'smart_199_raw', 'smart_240_raw', 'smart_241_raw', 'smart_242_raw'
            ]
            
            core_cols = ['date', 'serial_number', 'model', 'capacity_bytes', 'failure']
            
            for col in core_cols + all_smart_cols:
                if col not in df.columns:
                    df[col] = 0
            
            cur.execute("DROP TABLE IF EXISTS temp_load;")
            staging_columns = ", ".join([f"{c} TEXT" for c in core_cols + all_smart_cols])
            cur.execute(f"CREATE TEMP TABLE temp_load ({staging_columns}, node_id TEXT, datacenter TEXT, cluster_id TEXT, vault_id TEXT, pod_id TEXT);")
            
            extra_payload = {}
            for extra in ['node_id', 'datacenter', 'cluster_id', 'vault_id', 'pod_id']:
                if extra not in df.columns:
                    extra_payload[extra] = '0' if extra != 'datacenter' else 'dc-local'
            
            if extra_payload:
                extra_df = pd.DataFrame(extra_payload, index=df.index)
                df = pd.concat([df, extra_df], axis=1)
            
            final_export_cols = core_cols + all_smart_cols + ['node_id', 'datacenter', 'cluster_id', 'vault_id', 'pod_id']
            cleaned_df = df[final_export_cols].astype(str).copy()
            
            with cur.copy("COPY temp_load FROM STDIN WITH (FORMAT text, DELIMITER E'\\t', NULL 'nan')") as copy:
                for row in cleaned_df.itertuples(index=False):
                    row_strings = [str(x).replace('\t', ' ').replace('\n', ' ') for x in row]
                    copy.write_row(row_strings)
                    
            # 1. FIXED: node_id explicitly cast to integer to eliminate type mismatch crashes
            cur.execute("""
                INSERT INTO storage_nodes (node_id, datacenter, cluster_id, vault_id, pod_id)
                SELECT DISTINCT 
                    node_id::integer, 
                    datacenter, 
                    NULLIF(cluster_id, '')::numeric::integer, 
                    NULLIF(vault_id, '')::numeric::integer, 
                    NULLIF(pod_id, '')::numeric::integer 
                FROM temp_load
                ON CONFLICT (node_id) DO NOTHING;
            """)
            
            cur.execute("""
                INSERT INTO disks (serial_number, model, capacity_bytes, node_id, status)
                SELECT DISTINCT 
                    serial_number, 
                    model, 
                    NULLIF(capacity_bytes, '')::numeric::bigint, 
                    node_id::integer, 
                    'active' 
                FROM temp_load
                ON CONFLICT (serial_number) DO NOTHING;
            """)
            
            insert_fields = "disk_id, serial_number, model, timestamp, date, record_date, failure, " + ", ".join(all_smart_cols)
            select_fields = """
                d.disk_id, t.serial_number, t.model, t.date::timestamp, t.date::date, t.date::timestamp, t.failure::numeric::integer,
            """ + ", ".join([f"NULLIF(t.{c}, '')::numeric::bigint" for c in all_smart_cols])
            
            cur.execute(f"""
                INSERT INTO smart_telemetry ({insert_fields})
                SELECT {select_fields}
                FROM temp_load t
                JOIN disks d ON t.serial_number = d.serial_number
                WHERE t.failure::numeric::integer = 1 OR random() <= %s;
            """, (sample_rate,))
            
        except Exception as ex:
            print(f"[!] Critical file level execution exception: {ex}")
            raise ex

def main():
    parser = argparse.ArgumentParser(description="Telemetry Framework Logs Ingestion Script Driver")
    parser.add_argument("--data", required=True, help="Directory folder location of target raw files logs")
    parser.add_argument("--healthy-sample", type=float, default=0.05, help="Downward sampling multiplier tracking ratios")
    args = parser.parse_args()

    print(f"[-] Launching dataset ingestion module connecting to host configurations...")
    psycopg_dsn = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")

    with psycopg.connect(psycopg_dsn) as conn:
        with conn.cursor() as cur:
            load_raw_csv_files(cur, args.data, args.healthy_sample)
            conn.commit()
    print("[+] Ingestion data pipelines completed successfully. Core tables populated.\n")

if __name__ == "__main__":
    main()
