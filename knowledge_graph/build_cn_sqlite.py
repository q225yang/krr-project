import gzip
import sqlite3
from tqdm import tqdm

INPUT_GZ = "conceptnet-assertions-5.7.0.csv.gz"
DB_PATH = "conceptnet_en.db"

# ConceptNet CSV format is tab-separated:
# uri \t rel \t start \t end \t json
# Example json contains {"weight":..., "surfaceText":...}

import json

def is_en(concept_uri: str) -> bool:
    return concept_uri.startswith("/c/en/")

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS edges (
        start TEXT,
        rel TEXT,
        end TEXT,
        weight REAL,
        surfaceText TEXT
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_start ON edges(start)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_end ON edges(end)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_rel ON edges(rel)")
    conn.commit()

    # Speed pragmas for bulk insert
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA temp_store=MEMORY;")

    batch = []
    BATCH_SIZE = 50000

    with gzip.open(INPUT_GZ, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading lines"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue
            _uri, rel_uri, start_uri, end_uri, meta_json = parts

            # Keep only English-English edges
            if not (is_en(start_uri) and is_en(end_uri)):
                continue

            # relation label like "/r/UsedFor" -> "UsedFor"
            rel = rel_uri.split("/")[-1]

            try:
                meta = json.loads(meta_json)
            except Exception:
                meta = {}

            weight = float(meta.get("weight", 1.0))
            surface = meta.get("surfaceText", "")

            batch.append((start_uri, rel, end_uri, weight, surface))

            if len(batch) >= BATCH_SIZE:
                cur.executemany(
                    "INSERT INTO edges(start, rel, end, weight, surfaceText) VALUES (?,?,?,?,?)",
                    batch
                )
                conn.commit()
                batch.clear()

    if batch:
        cur.executemany(
            "INSERT INTO edges(start, rel, end, weight, surfaceText) VALUES (?,?,?,?,?)",
            batch
        )
        conn.commit()

    conn.close()
    print(f"Done. Wrote SQLite DB to: {DB_PATH}")

if __name__ == "__main__":
    main()