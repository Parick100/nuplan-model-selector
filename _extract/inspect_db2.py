import sqlite3, os, json
from collections import Counter, defaultdict
import numpy as np

DB = r"C:\Users\lenovo\Downloads\nuplan-v1.1_mini\2021.10.11.08.31.07_veh-50_01750_01948.db"
con = sqlite3.connect(DB); cur = con.cursor()

def cols(t): return [r[1] for r in cur.execute(f"PRAGMA table_info('{t}')")]
print("lidar_box cols:", cols("lidar_box"))
print("scene cols:", cols("scene"))
print("scenario_tag cols:", cols("scenario_tag"))

print("\n=== category ===")
for r in cur.execute("SELECT token, name, description FROM category"):
    print("  ", r[1], "|", r[2])

print("\n=== scenario_tag types ===")
tc = Counter()
for (t,) in cur.execute("SELECT type FROM scenario_tag"):
    tc[str(t)] += 1
for k, v in tc.most_common(60):
    print(f"   {v:6d}  {k[:90]}")

print("\n=== scenario_tag sample rows ===")
for r in cur.execute("SELECT * FROM scenario_tag LIMIT 5"):
    print("  ", [x.decode(errors='replace') if isinstance(x, bytes) else x for x in r])

# lidar_box: agent states
print("\n=== lidar_box ===")
n = cur.execute("SELECT COUNT(*) FROM lidar_box").fetchone()[0]
print("  rows:", n)
for r in cur.execute("SELECT * FROM lidar_box LIMIT 2"):
    d = dict(zip(cols("lidar_box"), r))
    print("  ", {k: (f"<blob {len(v)}B>" if isinstance(v, bytes) else v) for k, v in d.items()})

# track -> category mapping
print("\n=== tracks per category ===")
t2c = {}
for tok, ctok in cur.execute("SELECT token, category_token FROM track"):
    t2c[tok] = ctok
c2n = {r[0]: r[1] for r in cur.execute("SELECT token, name FROM category")}
cc = Counter(c2n.get(c, "?") for c in t2c.values())
print("  distinct tracks by category:", cc.most_common())

# frame ordering
frames = [r[0] for r in cur.execute("SELECT token FROM lidar_pc ORDER BY timestamp")]
fidx = {t: i for i, t in enumerate(frames)}
T = len(frames)
print("\n  total lidar_pc frames:", T)

# build per-track presence and position series
pres = defaultdict(dict)
for lpc, trk, x, y in cur.execute("SELECT lidar_pc_token, track_token, x, y FROM lidar_box"):
    i = fidx.get(lpc)
    if i is not None:
        pres[trk][i] = (x, y)

print("  distinct tracks with boxes:", len(pres))
lens = sorted((len(v) for v in pres.values()), reverse=True)
print("  track length (frames present): top20", lens[:20])
print("  tracks with >=200 frames (>=10s @20Hz):", sum(1 for l in lens if l >= 200))
print("  tracks with >=100 frames (>=5s):", sum(1 for l in lens if l >= 100))

# yield of valid (agent, t0) windows: need 40 past + 160 future contiguous
H, F = 40, 160
stride = 10  # 0.5s at 20Hz
valid_any, valid_strict = 0, 0
per_cat = Counter()
for trk, d in pres.items():
    if len(d) < H + F: continue
    ks = sorted(d)
    ks_set = set(ks)
    for i in range(0, T - H - F, stride):
        win = range(i, i + H + F)
        if all(w in ks_set for w in win):
            valid_any += 1
            per_cat[c2n.get(t2c.get(trk), "?")] += 1
print(f"\n  valid (agent,t0) windows @stride 0.5s, contiguous 10s: {valid_any}")
print("  by category:", per_cat.most_common())

# unique t0 count
print("\n=== images/camera ===")
for r in cur.execute("SELECT COUNT(*) FROM camera"): print("  cameras:", r[0])
con.close()
