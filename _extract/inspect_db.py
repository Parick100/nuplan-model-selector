import sqlite3, os, sys, struct, json
from collections import Counter

DB = r"C:\Users\lenovo\Downloads\nuplan-v1.1_mini\2021.10.11.08.31.07_veh-50_01750_01948.db"
con = sqlite3.connect(DB)
cur = con.cursor()

print("=== size MB:", round(os.path.getsize(DB)/1e6, 1))
print("=== tables ===")
tabs = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
for t in tabs:
    try:
        n = cur.execute(f"SELECT COUNT(*) FROM '{t}'").fetchone()[0]
    except Exception as e:
        n = f"ERR {e}"
    print(f"  {t:32s} {n}")

def cols(t):
    return [r[1] for r in cur.execute(f"PRAGMA table_info('{t}')")]
for t in ("lidar_pc", "track", "category", "ego_pose", "log", "scenario_tag"):
    if t in tabs:
        print(f"--- cols {t}: {cols(t)}")

if "log" in tabs:
    print("=== log rows ===")
    c = cols("log")
    for r in cur.execute("SELECT * FROM log LIMIT 5"):
        print(dict(zip(c, r)))

if "lidar_pc" in tabs:
    print("=== lidar_pc ===")
    n, tmin, tmax = cur.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM lidar_pc").fetchone()
    print("  count:", n, "duration_s:", round((tmax - tmin)/1e6, 1))
    ts = [r[0] for r in cur.execute("SELECT timestamp FROM lidar_pc ORDER BY timestamp LIMIT 200")]
    d = [round((ts[i+1]-ts[i])/1e6, 4) for i in range(len(ts)-1)]
    print("  dt(s) counts:", Counter(d).most_common(5))

if "ego_pose" in tabs:
    print("=== ego_pose ===")
    n, tmin, tmax = cur.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM ego_pose").fetchone()
    print("  count:", n, "duration_s:", round((tmax-tmin)/1e6, 1))

# decode track_objects blob from lidar_pc
print("=== track_objects decode attempt ===")
row = cur.execute("SELECT token, timestamp, track_objects FROM lidar_pc ORDER BY timestamp LIMIT 1").fetchone()
if row:
    tok, ts, blob = row
    print("  token:", tok, "len(blob):", len(blob) if blob else None)
    print("  head bytes:", blob[:24].hex() if blob else None)
    obj = None
    try:
        import msgpack
        from io import BytesIO
        def _ext(code, data):
            if code == 1:  # numpy scalar/array
                try:
                    import numpy as np
                    return np.load(BytesIO(data), allow_pickle=False)
                except Exception:
                    return data
            return data
        obj = msgpack.unpackb(blob, raw=False, ext_hook=_ext)
        print("  msgpack OK, type:", type(obj), "keys:", len(obj))
    except Exception as e:
        print("  msgpack failed:", repr(e))

    if obj:
        k0 = list(obj.keys())[0]
        print("  sample entry:", json.dumps({k: (str(v) if not isinstance(v, (int, float, str)) else v) for k, v in list(obj[k0].items())[:12]}, default=str)[:600])

# category distribution across a few frames
print("=== category counts (10 sampled frames) ===")
cnt = Counter()
frames = [r[0] for r in cur.execute("SELECT token FROM lidar_pc ORDER BY timestamp LIMIT 600")]
catmap = {}
if "category" in tabs and "track" in tabs:
    try:
        for r in cur.execute("SELECT token, name FROM category"):
            catmap[r[0]] = r[1]
    except Exception as e:
        print("  ", e)
    print("  category table names:", sorted(set(catmap.values()))[:30])

print("=== track table sample ===")
if "track" in tabs:
    c = cols("track")
    for r in cur.execute("SELECT * FROM track LIMIT 3"):
        d = dict(zip(c, r))
        d = {k: (f"<blob {len(v)}B>" if isinstance(v, bytes) else v) for k, v in d.items()}
        print("  ", json.dumps(d, default=str)[:400])
con.close()
