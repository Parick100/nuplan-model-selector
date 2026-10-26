import os, glob, shutil
from PIL import Image

SRC = r"C:\Users\lenovo\AppData\Roaming\dsh-desktop\harness\attachments\v1\objects"
OUT = r"C:\Users\lenovo\Desktop\Homework\Term5\Intro of ML\项目\pre1_ppt\assets"
os.makedirs(OUT, exist_ok=True)

names = {
    "near_ped":        r"51\5117181737d4169013227260fd8251f114497691e62dd8b8368841d388443fc2",
    "unprotect_turn":  r"ea\ea4122b88ae2bedc16482420cdb8d8b64ffc909a4c700abbf8c4d85b48eb0ffc",
    "multiple_vehicle":r"0e\0e0e644bf4f9dfdcbf77d615d657b5e3505cc8fc4b4187f52c57c78be87d40f1",
}

for name, rel in names.items():
    p = os.path.join(SRC, rel)
    print("==", name, os.path.exists(p))
    if not os.path.exists(p):
        continue
    im = Image.open(p)
    n = getattr(im, "n_frames", 1)
    print("   format:", im.format, "size:", im.size, "frames:", n, "mode:", im.mode)
    # save several frames as png
    picks = [0, n // 3, n // 2, (2 * n) // 3, n - 1] if n > 1 else [0]
    picked = []
    for i in sorted(set(picks)):
        im.seek(i)
        fr = im.convert("RGB")
        op = os.path.join(OUT, f"{name}_f{i:03d}.png")
        fr.save(op)
        picked.append((i, os.path.basename(op), fr.size))
    for t in picked:
        print("   frame", t)
print("--- files written ---")
for f in sorted(os.listdir(OUT)):
    print("  ", f, os.path.getsize(os.path.join(OUT, f)))
