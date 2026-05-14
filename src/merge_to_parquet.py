#!/usr/bin/env python3
"""Memory-efficient CSV -> merged Parquet pipeline using DuckDB.

Reads `train_transaction.csv` and `train_identity.csv`, performs a left
join on `TransactionID`, sorts by `TransactionDT`, and writes a single
compressed Parquet file to the given output path.

This approach uses DuckDB which handles large data efficiently and
streams data to disk, avoiding loading the full joined table into RAM.
"""
import argparse
import os
import sys

try:
    import duckdb
except Exception as e:
    print("duckdb is required. Install with: pip install duckdb", file=sys.stderr)
    raise


def ensure_dir_for_file(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def main():
    p = argparse.ArgumentParser(description='Merge Kaggle fraud CSVs to a single Parquet')
    p.add_argument('--transactions', '-t', required=True, help='Path to train_transaction.csv')
    p.add_argument('--identity', '-i', required=True, help='Path to train_identity.csv')
    p.add_argument('--out', '-o', default='data/processed/merged.parquet', help='Output Parquet file path')
    p.add_argument('--compression', '-c', default='SNAPPY', choices=['SNAPPY','ZSTD','GZIP','NONE'], help='Parquet compression')
    args = p.parse_args()

    transactions_path = os.path.abspath(args.transactions)
    identity_path = os.path.abspath(args.identity)
    out_path = os.path.abspath(args.out)

    ensure_dir_for_file(out_path)

    con = duckdb.connect(database=':memory:')

    # Create DuckDB tables from CSVs. DuckDB will stream and is efficient.
    print('Reading CSV headers and registering tables in DuckDB...')
    con.execute("DROP TABLE IF EXISTS transactions")
    con.execute("DROP TABLE IF EXISTS identity")

    # Use read_csv_auto which infers types robustly. Setting sample_size to -1 uses full file for inference,
    # but that may be slow on huge files; default inference is usually fine. We keep defaults to be safe.
    tx_esc = transactions_path.replace("'", "''")
    id_esc = identity_path.replace("'", "''")
    con.execute("CREATE TABLE transactions AS SELECT * FROM read_csv_auto('" + tx_esc + "')")
    con.execute("CREATE TABLE identity AS SELECT * FROM read_csv_auto('" + id_esc + "')")

    # Ensure TransactionDT exists
    res = con.execute("PRAGMA show_tables").fetchall()
    print('Tables registered:', res)

    # Perform left join and write directly to Parquet ordered by TransactionDT.
    # Using COPY from a SELECT allows DuckDB to stream to parquet without materializing
    # everything in Python memory.
    sql = (
        "COPY ("
        "SELECT * FROM transactions t LEFT JOIN identity i USING(TransactionID) "
        "ORDER BY TransactionDT"
        ") TO '" + out_path.replace("'","''") + "' (FORMAT PARQUET, COMPRESSION '" + args.compression + "')"
    )

    print('Writing merged Parquet to', out_path)
    con.execute(sql)

    con.close()
    print('Done.')


if __name__ == '__main__':
    main()
