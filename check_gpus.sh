#!/bin/bash
kubectl get nodes -o json | python3 -c '
import json, sys

data = json.load(sys.stdin)
print(f"{'NODE':<40} {'CAPACITY':>10} {'ALLOCATABLE':>12} {'USED':>6} {'FREE':>6}")
print("-" * 80)

for node in data['items']:
    name = node['metadata']['name']
    capacity = int(node['status'].get('capacity', {}).get('nvidia.com/gpu', 0))
    allocatable = int(node['status'].get('allocatable', {}).get('nvidia.com/gpu', 0))
    
    # get used from allocated resources via kubectl describe is hard from json
    # so just show capacity vs allocatable (allocatable = capacity - reserved by system)
    print(f"{name:<40} {capacity:>10} {allocatable:>12} {'?':>6} {allocatable:>6}")

'