"""Phase 0.1: inventory CFPB zips in ./data (streamed, no extraction)."""
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"
NARR = "Consumer complaint narrative"
ID = "Complaint ID"

headers, per_file = {}, []
ids = pd.Series(dtype="int64")
total = nonempty = 0
for zp in sorted(DATA.glob("*.zip")):
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        assert len(names) == 1, (zp, names)
        with z.open(names[0]) as f:
            df = pd.read_csv(f, dtype=str, keep_default_na=False, usecols=None)
    headers[zp.name] = list(df.columns)
    ne = df[NARR].str.strip().ne("")
    per_file.append({"file": zp.name, "rows": len(df), "nonempty_narrative": int(ne.sum()),
                     "date_min": df["Date received"].min(), "date_max": df["Date received"].max()})
    total += len(df)
    nonempty += int(ne.sum())
    ids = pd.concat([ids, df.loc[ne, ID].astype("int64")], ignore_index=True)
    print(per_file[-1], file=sys.stderr)

first = next(iter(headers.values()))
out = {
    "files": per_file,
    "headers_identical": all(h == first for h in headers.values()),
    "columns": first,
    "total_rows": total,
    "nonempty_narrative": nonempty,
    "nonempty_unique_ids": int(ids.nunique()),
    "nonempty_duplicate_id_rows": int(ids.duplicated().sum()),
}
print(json.dumps(out, indent=1))
