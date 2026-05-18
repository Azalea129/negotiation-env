"""Print a full episode with context, belief_gt, and intention_gt for review."""
import json
import sys
from pathlib import Path

data_dir = sys.argv[2] if len(sys.argv) > 2 else "data_smoke2"
episodes_file = Path(data_dir) / "episodes.jsonl"
with episodes_file.open(encoding="utf-8") as f:
    episodes = [json.loads(line) for line in f]

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ep = episodes[idx]

SEP = "=" * 70
DIV = "-" * 70

print(SEP)
print("Episode :", ep["episode_id"])
print("Deal    :", ep["deal_reached"], " | Price:", ep.get("deal_price", "n/a"))
print("Buyer reservation :", round(ep["buyer_reservation"], 1))
print("Seller reservation:", round(ep["seller_reservation"], 1))
print(SEP)

for turn in ep["turns"]:
    role = turn["role"].upper()
    print()
    print(DIV)
    print("  TURN", turn["round"], "--", role)
    print(DIV)

    ctx = turn["context"]
    if ctx:
        print()
        print("[CONTEXT / history]")
        for line in ctx.strip().split("\n"):
            print("  " + line)
    else:
        print()
        print("[CONTEXT] (first turn -- no history)")

    print()
    print("[BELIEF GT]")
    for sent in turn["belief_gt"].replace(". ", ".\n").strip().split("\n"):
        if sent.strip():
            print("  " + sent.strip())

    print()
    print("[INTENTION GT]")
    for sent in turn["intention_gt"].replace(". ", ".\n").strip().split("\n"):
        if sent.strip():
            print("  " + sent.strip())

    print()
    print("[MESSAGE]")
    print("  " + turn["message"])

print()
print(SEP)
n = len(ep["turns"])
br = round(ep.get("buyer_reward", 0), 1)
sr = round(ep.get("seller_reward", 0), 1)
print(f"Total turns: {n}  |  Buyer reward: {br}  |  Seller reward: {sr}")
