import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "vastai_full_download_2026-05-20_0830UTC" / "project" / "models" / "calibrator.pkl"

with open(P, "rb") as f:
    c = pickle.load(f)

print(f"type: {type(c)}")
if hasattr(c, "__dict__"):
    for k, v in c.__dict__.items():
        print(f"  {k}: type={type(v).__name__}", end="")
        if hasattr(v, "shape"):
            print(f" shape={v.shape}", end="")
        try:
            print(f" value={v}", end="")
        except Exception:
            pass
        print()
elif isinstance(c, dict):
    for k, v in c.items():
        print(f"  {k}: {v}")
else:
    print(repr(c)[:2000])
