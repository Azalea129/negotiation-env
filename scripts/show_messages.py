"""Show visible_message vs action for each turn to verify natural-language separation."""
import json
import sys
from pathlib import Path

data_dir = sys.argv[1] if len(sys.argv) > 1 else "data_smoke2"
ep_file = Path(data_dir) / "episodes.jsonl"

with ep_file.open(encoding="utf-8") as f:
    ep = json.loads(f.readline())

print("Episode:", ep["episode_id"])
print("Deal:", ep["deal_reached"], "| Price:", ep.get("deal_price"))
print()

for turn in ep["turns"]:
    print(f"Turn {turn['round']} | {turn['role'].upper()}")
    print(f"  [visible_message] {turn['message']}")
    print(f"  [action]          {turn['action']}  price={turn['price']}")
    print()
