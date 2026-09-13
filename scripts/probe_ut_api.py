import uncertainty_toolbox as ut
import uncertainty_toolbox.metrics as m
import os

print("version:", getattr(ut, "__version__", "?"))
print("top-level:", [x for x in dir(ut) if not x.startswith("_")])
print()
print("metrics module:", [x for x in dir(m) if not x.startswith("_")])
print()
p = os.path.dirname(m.__file__)
print("metrics files:", sorted(f for f in os.listdir(p) if f.endswith(".py")))
print()
# 找与拒识/选择性预测相关的名字
keys = [x for x in dir(m) if not x.startswith("_")]
hits = [k for k in keys if any(t in k.lower() for t in
        ("risk", "cover", "aurc", "absten", "select", "reject", "retention"))]
print("候选(拒识相关):", hits)
