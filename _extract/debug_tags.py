import sqlite3, numpy as np
from collections import Counter

DB = r"C:\Users\lenovo\Downloads\nuplan-v1.1_mini\2021.10.11.08.31.07_veh-50_01750_01948.db"
con = sqlite3.connect(DB)
frames = [t for (t,) in con.execute("SELECT token FROM lidar_pc ORDER BY timestamp")]
fidx = {t: i for i, t in enumerate(frames)}

tag_frames = Counter()
types = Counter()
for lpc, typ in con.execute("SELECT lidar_pc_token, type FROM scenario_tag"):
    i = fidx.get(lpc)
    if i is not None:
        tag_frames[i] += 1
        types[str(typ)] += 1

print("frames with >=1 tag :", len(tag_frames), "of", len(frames))
print("total tag rows      :", sum(tag_frames.values()))
tot = np.zeros(len(frames), dtype=int)
for i, c in tag_frames.items():
    tot[i] = c
print("tag rows per frame  : mean %.2f  max %d" % (tot.mean(), tot.max()))

# 10 Hz 帧视角
dec = np.arange(0, len(frames), 2)
tag10 = tot[dec]
print("10 Hz frames with tag:", int((tag10 > 0).sum()), "of", len(dec))

d = np.load(r"C:\Users\lenovo\Desktop\Homework\Term5\Intro of ML\项目\code\data\cache\samples.npz")
t0 = d["t0"]
print("\nkept samples        :", len(t0))
hit = [int(tag10[t]) for t in t0]
print("of which tagged     :", sum(1 for h in hit if h > 0))
print("tag counts at t0    :", Counter(hit).most_common(6))

# 手工复算一个
print("\nsample t0 (10Hz idx) :", t0[:8].tolist())
print("-> 20Hz idx          :", [int(dec[t]) for t in t0[:8]])
print("-> tag rows at those :", [int(tot[dec[t]]) for t in t0[:8]])

print("\ntop tag types overall:")
for k, v in types.most_common(8):
    print(f"   {v:5d}  {k}")
con.close()
