import importlib, sys
print("python", sys.version.split()[0])
for m in ["torch","numpy","msgpack","sqlalchemy","hydra","shapely","geopandas","fiona",
          "h5py","matplotlib","timm","einops","scipy","pandas","sklearn","zarr","pyarrow"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:12s} {getattr(mod, '__version__', '?')}")
    except Exception:
        print(f"  {m:12s} MISSING")
try:
    import torch
    print("cuda available:", torch.cuda.is_available(), "| count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("vram GB:", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))
except Exception as e:
    print("torch err:", e)
import platform
print("os:", platform.platform())
try:
    import os
    print("cpu count:", os.cpu_count())
except Exception:
    pass
