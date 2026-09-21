"""Consistent online SQLite backup, including committed WAL data."""
import argparse
import sqlite3
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',default='data/opbench.sqlite3')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    source,target=Path(args.db).resolve(),Path(args.output).resolve()
    if not source.is_file():p.error('source database does not exist')
    if target.exists():p.error('output already exists; choose a new backup path')
    target.parent.mkdir(parents=True,exist_ok=True)
    src=sqlite3.connect(source.as_uri()+'?mode=ro',uri=True)
    dst=sqlite3.connect(target)
    try:src.backup(dst)
    finally:dst.close();src.close()
    print(f'Backup saved: {target}')


if __name__=='__main__':main()
