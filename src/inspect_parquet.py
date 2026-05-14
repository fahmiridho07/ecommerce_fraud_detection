#!/usr/bin/env python3
import os
import sys
try:
    import duckdb
except Exception:
    print('duckdb not installed. Run: pip install duckdb', file=sys.stderr)
    raise

def human(n):
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024.0:
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}PB"

def main(p='data/processed/merged.parquet'):
    p = os.path.abspath(p)
    if not os.path.exists(p):
        print('File not found:', p)
        sys.exit(2)
    print('PATH:', p)
    size = os.path.getsize(p)
    print('SIZE_BYTES:', size, '(', human(size), ')')
    con = duckdb.connect()
    # Read a small sample to infer columns and types
    q = "SELECT * FROM read_parquet('{}') LIMIT 5".format(p.replace("'","''"))
    df = con.execute(q).fetchdf()
    print('\nCOLUMNS:')
    for c, t in zip(df.columns, df.dtypes):
        print(c, t)
    print('\nFIRST 5 ROWS:')
    for r in df.itertuples(index=False, name=None):
        print(r)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('path', nargs='?', default='data/processed/merged.parquet')
    args = ap.parse_args()
    main(args.path)
