"""把缓存从「每 log 一个压缩 npz」转成「每 log 一个目录 + 未压缩 .npy」。

**为什么必须转**
训练时 batch 会跨 log 乱序，而压缩 npz 每次都要**解压整个文件**（单个最大 ~38 MB）。
实测 6 s/step（约 300× 于计算量），完全是数据侧瓶颈。

未压缩 .npy + `mmap_mode='r'` 是零拷贝：OS 页缓存负责换页，随机取一条样本只碰它所在的页。
代价是磁盘从 0.96 GB 涨到约 1.8 GB（WSL 剩余 885 GB，可忽略）。

    /home/parick/ENTER/envs/diffusion_planner/bin/python code/data/convert_cache.py
"""
import argparse
import json
import os

import numpy as np

KEYS = ("hist", "neigh", "neigh_mask", "future", "tags", "t0", "track_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/parick/dsh_ml/cache")
    ap.add_argument("--force", action="store_true",
                    help="即使 npy 目录已存在也重新转换（重建缓存后必须加，否则会跳过）")
    a = ap.parse_args()

    man = json.load(open(os.path.join(a.root, "manifest.json"), encoding="utf-8"))
    logs = man["logs"]
    print(f"{len(logs)} 个 log，开始转换 …")

    done = 0
    for e in logs:
        npz = os.path.join(a.root, e["file"])
        if not os.path.exists(npz):
            print(f"  !! 缺 {npz}")
            continue
        stem = os.path.basename(npz)[:-4]
        outdir = os.path.join(a.root, "npy", stem)
        if not a.force and os.path.isdir(outdir) and all(
                os.path.exists(os.path.join(outdir, k + ".npy")) for k in KEYS):
            done += 1
            continue
        os.makedirs(outdir, exist_ok=True)
        z = np.load(npz)
        for k in KEYS:
            if k in z.files:
                np.save(os.path.join(outdir, k + ".npy"), z[k])
        z.close()
        done += 1
        if done % 16 == 0:
            print(f"  {done}/{len(logs)}")

    # 更新 manifest 的文件指向
    for e in logs:
        stem = os.path.basename(e["file"])[:-4]
        e["file_npz"] = e.pop("file")
        e["file"] = os.path.join("npy", stem)
    with open(os.path.join(a.root, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, ensure_ascii=False, indent=2)

    tot = sum(os.path.getsize(os.path.join(dp, f))
              for dp, _, fs in os.walk(os.path.join(a.root, "npy")) for f in fs)
    print(f"\n完成 {done}/{len(logs)}   npy 目录总大小 {tot/1024**3:.2f} GB")
    print("manifest 已更新为指向 .npy 目录（原字段移入 file_npz）")


if __name__ == "__main__":
    main()
