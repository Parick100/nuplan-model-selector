#!/bin/bash
# 只读探测：找出 WSL 里可用的 Python 与关键依赖。不修改任何既有文件。
echo "=== python 候选 ==="
ls -d /home/parick/ENTER/envs/*/ 2>/dev/null
ls /home/parick/ENTER/bin/python* 2>/dev/null
ls /home/parick/miniconda3/bin/python /home/parick/anaconda3/bin/python 2>/dev/null

for p in /usr/bin/python3 /home/parick/ENTER/bin/python \
         /home/parick/ENTER/envs/*/bin/python \
         /home/parick/miniconda3/bin/python /home/parick/anaconda3/bin/python; do
  [ -x "$p" ] || continue
  echo "--- $p"
  "$p" -V 2>&1
  "$p" - <<'PY' 2>&1 | sed 's/^/    /'
import importlib
for m in ("numpy", "torch", "scipy", "sqlalchemy", "msgpack", "shapely", "pandas"):
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "?"))
    except Exception:
        print(m, "MISSING")
try:
    import torch
    print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
except Exception as e:
    print("torch probe:", e)
PY
done

echo "=== nuplan-devkit 可导入? ==="
for p in /usr/bin/python3 /home/parick/ENTER/envs/*/bin/python; do
  [ -x "$p" ] || continue
  printf '%s -> ' "$p"
  "$p" -c "import nuplan,os;print('OK',os.path.dirname(nuplan.__file__))" 2>&1 | tail -1
done

echo "=== 数据集 ==="
ls /home/parick/nuplan/dataset/nuplan-v1.1/splits/mini/*.db 2>/dev/null | wc -l
du -sh /home/parick/nuplan/dataset/nuplan-v1.1/splits/mini 2>/dev/null

echo "=== CPU / 内存 ==="
nproc
free -g | head -2
