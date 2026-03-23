import json, sys

data = json.load(sys.stdin)
print(f"{'NODE':<50} {'CAPACITY':>10} {'ALLOCATABLE':>12}")
print("-" * 75)
for node in data["items"]:
    name = node["metadata"]["name"]
    cap = int(node["status"].get("capacity", {}).get("nvidia.com/gpu", 0))
    alloc = int(node["status"].get("allocatable", {}).get("nvidia.com/gpu", 0))
    if cap > 0:
        print(f"{name:<50} {cap:>10} {alloc:>12}")
