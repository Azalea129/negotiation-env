import sys
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, '.')

from negotiation_project import Condition, NegotiationConfig, NegotiationEnv, sample_types

config = NegotiationConfig(condition=Condition.FREE_FORM, max_rounds=8)
buyer_type, seller_type = sample_types(seed=42)
print(f"Buyer: {buyer_type}")
print(f"Seller: {seller_type}")

env = NegotiationEnv(config)
result = env.run(buyer_type=buyer_type, seller_type=seller_type)

print(f"\n결과: {'DEAL' if result.deal_reached else 'NO DEAL'} ({result.termination})")
if result.deal_reached:
    print(f"가격: ${result.deal_price:.0f} | 라운드: {result.deal_round}")
    print(f"Total welfare: {result.total_welfare:.1f}")

print("\n--- TRANSCRIPT ---")
for turn in result.turns:
    print(f"\n[Round {turn.round} | {turn.speaker.upper()} | action={turn.action}]")
    print(turn.visible_message)
