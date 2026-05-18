"""
Concordia-based Stage 1 data collection for LASH.

Runs mixed-motive game scenarios where each agent produces structured BDI CoT,
capturing (context, belief_gt, intention_gt, action) training pairs in exactly
the same format as the A2A negotiation env (lash/data_collector.py).

Supported standalone games (no Concordia install required):
  - PrisonersDilemmaGame   : 2-player repeated cooperation/defection
  - PublicGoodsGame        : N-player contribution game
  - ResourceAllocationGame : N-player sealed-bid auction
  - HagglingGame           : 2-player bilateral price negotiation (Stage 1 variant)
  - StagHuntGame           : 2-player coordination game (held-out evaluation candidate)

Concordia integration (requires `pip install concordia`):
  - ConcordiaScenarioRunner : wraps existing Concordia scenarios with LASH agents
    Usage: ConcordiaScenarioRunner.run(concordia_game_master, agents)

Data flow per turn:
  observation (c_t) → LLM → <belief> B_t^GT + <intention> I_t^GT + [ACTION]
  → TurnData saved to training_pairs.jsonl
"""

from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from .config import LASHConfig
from .data_collector import append_episode, append_training_pairs
from .types import EpisodeData, TurnData


# ── Structured CoT prompts ──────────────────────────────────────────────────

_AGENT_SYSTEM = """\
You are {agent_name}, a player in {game_name}.

GAME RULES:
{game_rules}

YOUR PRIVATE INFORMATION:
{private_info}

REASONING FORMAT — required every turn:

<belief>
Based on {opponent_description}'s past actions and messages, estimate:
- Their likely strategy type (cooperative / selfish / mixed / adaptive)
- Their private information or valuation if relevant
- How their strategy might shift this round
</belief>
<intention>
Given your belief, decide your action:
- What will you do and why?
- How does this exploit or respond to opponents' estimated strategies?
</intention>
{message_section}
{action_format}

{visibility_note}
"""

_MSG_SECTION_GAME = """\
<message>
[What you say out loud to the other players — natural language only, 1 to 3 sentences.
 Do NOT directly reveal your private information or explicitly state your action choice.]
</message>"""

_VISIBILITY_GAME_WITH_MSG = (
    "The <belief>, <intention>, and action line are PRIVATE — NOT shown to other players.\n"
    "Only your <message> block is visible to opponents."
)
_VISIBILITY_GAME_NO_MSG = (
    "The <belief> and <intention> sections are PRIVATE — NOT shown to other players.\n"
    "The action line is forwarded directly to opponents."
)


def _build_system_prompt(config: LASHConfig, **kwargs) -> str:
    """Fill _AGENT_SYSTEM with message_section and visibility_note from config."""
    use_msg = config.natural_language_message
    return _AGENT_SYSTEM.format(
        message_section=_MSG_SECTION_GAME if use_msg else "",
        visibility_note=_VISIBILITY_GAME_WITH_MSG if use_msg else _VISIBILITY_GAME_NO_MSG,
        **kwargs,
    )


# ── Generic strategic game agent ───────────────────────────────────────────

class StrategicGameAgent:
    """
    LLM-backed agent for any mixed-motive game.
    Produces structured BDI CoT at each turn, mirroring lash/agents.py.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        config: LASHConfig,
        agent_idx: int = 0,
        natural_language_message: bool = True,
    ):
        self.natural_language_message = natural_language_message
        self.name = name
        self.agent_idx = agent_idx
        self.config = config

        client_kwargs: dict = {}
        if config.api_base:
            client_kwargs["base_url"] = config.api_base
        if config.api_key:
            client_kwargs["api_key"] = config.api_key
        self.client = OpenAI(**client_kwargs)

        self.history: list[dict] = [{"role": "system", "content": system_prompt}]
        self.context_lines: list[str] = []   # running c_t for this agent

    def act(self, observation: str) -> dict:
        """
        Observe game state and generate structured CoT + action.

        Returns:
          raw_cot:         full model output
          belief_text:     B_t^GT
          intention_text:  I_t^GT
          visible_message: action shown to other players
          numeric_action:  parsed float (None if not applicable)
          context:         c_t snapshot before this turn
        """
        context_snapshot = "\n".join(self.context_lines)

        self.history.append({"role": "user", "content": observation})
        completion = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=self.history,
        )
        raw = completion.choices[0].message.content
        self.history.append({"role": "assistant", "content": raw})

        belief, intention = _extract_cot(raw)
        action_text = _extract_action_text(raw)  # action tag line — private, game engine only
        if self.natural_language_message:
            message = _extract_message(raw)   # natural language from <message> tag
        else:
            message = action_text             # ablation: bare action tag forwarded directly

        self.context_lines.append(f"[{self.name}]: {message}")

        return {
            "raw_cot": raw,
            "belief_text": belief,
            "intention_text": intention,
            "visible_message": message,
            "action_text": action_text,
            "numeric_action": _parse_number(action_text),
            "context": context_snapshot,
        }


def _extract_cot(raw: str) -> tuple[str, str]:
    b = re.search(r"<belief>(.*?)</belief>", raw, re.DOTALL | re.IGNORECASE)
    i = re.search(r"<intention>(.*?)</intention>", raw, re.DOTALL | re.IGNORECASE)
    return (b.group(1).strip() if b else ""), (i.group(1).strip() if i else "")


def _extract_message(raw: str) -> str:
    """Extract natural-language <message> tag content; fall back to _strip_cot."""
    m = re.search(r"<message>(.*?)</message>", raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return _strip_cot(raw)


def _extract_action_text(raw: str) -> str:
    """Strip all CoT/message tags; return only the action line(s) for game engine parsing."""
    s = re.sub(r"<belief>.*?</belief>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<intention>.*?</intention>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<message>.*?</message>", "", s, flags=re.DOTALL | re.IGNORECASE)
    return s.strip()


def _strip_cot(raw: str) -> str:
    s = re.sub(r"<belief>.*?</belief>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<intention>.*?</intention>", "", s, flags=re.DOTALL | re.IGNORECASE)
    return s.strip()


def _parse_number(text: str) -> Optional[float]:
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(m.group()) if m else None


# ── Base game class ─────────────────────────────────────────────────────────

class BaseGame(ABC):
    """Abstract base for all LASH Stage 1 game scenarios."""

    def __init__(self, config: LASHConfig):
        self.config = config

    @abstractmethod
    def run(self, seed: Optional[int] = None) -> EpisodeData:
        """Run one episode and return EpisodeData with all TurnData."""
        ...

    def _new_episode_id(self) -> str:
        return str(uuid.uuid4())[:8]

    def _make_turn(
        self,
        turn_idx: int,
        round_idx: int,
        agent: StrategicGameAgent,
        resp: dict,
        action_str: str,
    ) -> TurnData:
        return TurnData(
            turn=turn_idx,
            round=round_idx,
            role=agent.name,
            context=resp["context"],
            raw_cot=resp["raw_cot"],
            belief_text=resp["belief_text"],
            intention_text=resp["intention_text"],
            visible_message=resp["visible_message"],
            action=action_str,
            price=resp["numeric_action"],
        )


# ── Prisoner's Dilemma ──────────────────────────────────────────────────────

_PD_RULES = """\
- Each round both players simultaneously choose COOPERATE or DEFECT.
- Payoffs per round:
    Both cooperate   → each gets 3
    You defect, they cooperate → you get 5, they get 0
    You cooperate, they defect → you get 0, they get 5
    Both defect      → each gets 1
- Game lasts {n_rounds} rounds. Maximise total payoff."""

_PD_ACTION_FORMAT = """\
ACTION FORMAT: respond with exactly one word on the last line:
  COOPERATE
  DEFECT"""


class PrisonersDilemmaGame(BaseGame):
    """
    2-player repeated Prisoner's Dilemma.

    Strategy reasoning: the key ToM task is estimating whether the opponent
    is a tit-for-tat player, always-defect, or conditional cooperator.
    """

    def __init__(self, config: LASHConfig, n_rounds: int = 6):
        super().__init__(config)
        self.n_rounds = n_rounds

    def run(self, seed: Optional[int] = None) -> EpisodeData:
        names = ["Alice", "Bob"]
        turns: list[TurnData] = []
        payoffs = [0.0, 0.0]
        turn_idx = 0

        agents = [
            StrategicGameAgent(
                name=names[i],
                system_prompt=_build_system_prompt(self.config,
                    agent_name=names[i],
                    game_name="Prisoner's Dilemma",
                    game_rules=_PD_RULES.format(n_rounds=self.n_rounds),
                    private_info="No private information — this is a symmetric game.",
                    opponent_description=names[1 - i],
                    action_format=_PD_ACTION_FORMAT,
                ),
                config=self.config,
                agent_idx=i,
                natural_language_message=self.config.natural_language_message,
            )
            for i in range(2)
        ]

        last_actions = [None, None]

        for rnd in range(self.n_rounds):
            observation = self._build_obs(rnd, last_actions, names)

            resps = [agents[i].act(observation) for i in range(2)]
            actions = [_parse_pd_action(r["raw_cot"]) for r in resps]

            for i in range(2):
                turns.append(self._make_turn(turn_idx + i, rnd, agents[i], resps[i], actions[i]))
                # Share opponent's natural-language message for next round's context
                agents[1 - i].context_lines.append(f"[{agents[i].name}]: {resps[i]['visible_message']}")

            p0, p1 = _pd_payoff(actions[0], actions[1])
            payoffs[0] += p0
            payoffs[1] += p1
            last_actions = actions
            turn_idx += 2

        return EpisodeData(
            episode_id=self._new_episode_id(),
            buyer_type=None,     # not applicable; use None
            seller_type=None,
            deal_reached=True,
            deal_price=None,
            deal_round=self.n_rounds - 1,
            termination="complete",
            buyer_surplus=payoffs[0],
            seller_surplus=payoffs[1],
            total_welfare=sum(payoffs),
            buyer_reward=payoffs[0],
            seller_reward=payoffs[1],
            turns=turns,
        )

    def _build_obs(self, rnd: int, last_actions: list, names: list) -> str:
        lines = [f"=== PRISONER'S DILEMMA — Round {rnd + 1}/{self.n_rounds} ==="]
        if rnd > 0:
            lines.append(f"Last round: {names[0]}={last_actions[0]}, {names[1]}={last_actions[1]}")
        lines.append("\nYour move for this round:")
        return "\n".join(lines)


def _parse_pd_action(text: str) -> str:
    upper = text.upper()
    if "COOPERATE" in upper:
        return "COOPERATE"
    if "DEFECT" in upper:
        return "DEFECT"
    return "DEFECT"   # conservative default


def _pd_payoff(a0: str, a1: str) -> tuple[float, float]:
    if a0 == "COOPERATE" and a1 == "COOPERATE":
        return 3.0, 3.0
    if a0 == "DEFECT" and a1 == "COOPERATE":
        return 5.0, 0.0
    if a0 == "COOPERATE" and a1 == "DEFECT":
        return 0.0, 5.0
    return 1.0, 1.0


# ── Public Goods Game ───────────────────────────────────────────────────────

_PGG_RULES = """\
- Each round, every player receives an endowment of {endowment} tokens.
- Each player privately decides how much to contribute to the public pool (0 to {endowment}).
- The pool is multiplied by {multiplier} and split equally among all {n_players} players.
- You keep uncontributed tokens. Game lasts {n_rounds} rounds. Maximise total tokens."""

_PGG_ACTION_FORMAT = """\
ACTION FORMAT: last line must be exactly:
  CONTRIBUTE: X
where X is an integer from 0 to {endowment}."""


class PublicGoodsGame(BaseGame):
    """
    N-player repeated Public Goods Game.

    ToM task: estimate each opponent's contribution tendency to decide
    whether to free-ride or maintain cooperation.
    """

    def __init__(
        self,
        config: LASHConfig,
        n_players: int = 4,
        n_rounds: int = 6,
        endowment: int = 10,
        multiplier: float = 2.0,
    ):
        super().__init__(config)
        self.n_players = n_players
        self.n_rounds = n_rounds
        self.endowment = endowment
        self.multiplier = multiplier

    def run(self, seed: Optional[int] = None) -> EpisodeData:
        names = [f"Player{i+1}" for i in range(self.n_players)]
        turns: list[TurnData] = []
        total_tokens = [0.0] * self.n_players
        turn_idx = 0
        history: list[dict] = []   # [{name: contribution}]

        agents = [
            StrategicGameAgent(
                name=names[i],
                system_prompt=_build_system_prompt(self.config,
                    agent_name=names[i],
                    game_name="Public Goods Game",
                    game_rules=_PGG_RULES.format(
                        endowment=self.endowment,
                        multiplier=self.multiplier,
                        n_players=self.n_players,
                        n_rounds=self.n_rounds,
                    ),
                    private_info="No private information — symmetric game.",
                    opponent_description="the other players",
                    action_format=_PGG_ACTION_FORMAT.format(endowment=self.endowment),
                ),
                config=self.config,
                agent_idx=i,
                natural_language_message=self.config.natural_language_message,
            )
            for i in range(self.n_players)
        ]

        for rnd in range(self.n_rounds):
            observation = self._build_obs(rnd, history, names)
            resps = [agents[i].act(observation) for i in range(self.n_players)]
            contributions = [
                max(0, min(self.endowment, int(r["numeric_action"] or 0)))
                for r in resps
            ]
            # Share each agent's natural-language message with all others
            for i in range(self.n_players):
                for j in range(self.n_players):
                    if i != j:
                        agents[j].context_lines.append(
                            f"[{agents[i].name}]: {resps[i]['visible_message']}"
                        )

            pool = sum(contributions) * self.multiplier
            share = pool / self.n_players

            for i in range(self.n_players):
                turns.append(self._make_turn(
                    turn_idx + i, rnd, agents[i], resps[i],
                    f"CONTRIBUTE:{contributions[i]}"
                ))
                total_tokens[i] += (self.endowment - contributions[i]) + share

            history.append({names[i]: contributions[i] for i in range(self.n_players)})
            turn_idx += self.n_players

        return EpisodeData(
            episode_id=self._new_episode_id(),
            buyer_type=None,
            seller_type=None,
            deal_reached=True,
            deal_price=None,
            deal_round=self.n_rounds - 1,
            termination="complete",
            buyer_surplus=total_tokens[0],
            seller_surplus=sum(total_tokens[1:]) / max(1, self.n_players - 1),
            total_welfare=sum(total_tokens),
            buyer_reward=total_tokens[0],
            seller_reward=total_tokens[-1],
            turns=turns,
        )

    def _build_obs(self, rnd: int, history: list[dict], names: list) -> str:
        lines = [f"=== PUBLIC GOODS GAME — Round {rnd + 1}/{self.n_rounds} ==="]
        if history:
            lines.append("Past contributions:")
            for r_idx, record in enumerate(history):
                row = ", ".join(f"{n}={record[n]}" for n in names)
                lines.append(f"  Round {r_idx + 1}: {row}")
        lines.append(f"\nYour endowment this round: {self.endowment}")
        lines.append("Decide your contribution:")
        return "\n".join(lines)


# ── Resource Allocation (Sealed-Bid Auction) ────────────────────────────────

_RA_RULES = """\
- {n_items} items are auctioned simultaneously in a sealed-bid format.
- Each player submits one bid per item. Highest bidder wins each item; ties broken randomly.
- Winners pay their bid. Your profit = private_value - bid (if you win), else 0.
- Budget limit: {budget} total across all bids."""

_RA_ACTION_FORMAT = """\
ACTION FORMAT: last line must list bids for all {n_items} items:
  BIDS: X1, X2, ..., X{n_items}
where each Xi is a non-negative integer and sum(Xi) <= {budget}."""


class ResourceAllocationGame(BaseGame):
    """
    N-player sealed-bid multi-item auction.

    ToM task: estimate opponents' private valuations from their bidding
    patterns to calibrate own bids optimally.
    """

    def __init__(
        self,
        config: LASHConfig,
        n_players: int = 3,
        n_items: int = 3,
        n_rounds: int = 4,
        budget: int = 30,
        value_range: tuple[int, int] = (5, 20),
    ):
        super().__init__(config)
        self.n_players = n_players
        self.n_items = n_items
        self.n_rounds = n_rounds
        self.budget = budget
        self.value_range = value_range

    def run(self, seed: Optional[int] = None) -> EpisodeData:
        import random
        rng = random.Random(seed)

        names = [f"Bidder{i+1}" for i in range(self.n_players)]
        turns: list[TurnData] = []
        total_profits = [0.0] * self.n_players
        turn_idx = 0
        bid_history: list[dict] = []

        # Private valuations per player per item (hidden from others)
        valuations = [
            [rng.randint(*self.value_range) for _ in range(self.n_items)]
            for _ in range(self.n_players)
        ]

        agents = [
            StrategicGameAgent(
                name=names[i],
                system_prompt=_build_system_prompt(self.config,
                    agent_name=names[i],
                    game_name="Multi-Item Sealed-Bid Auction",
                    game_rules=_RA_RULES.format(
                        n_items=self.n_items,
                        budget=self.budget,
                    ),
                    private_info=(
                        f"Your private values for items 1-{self.n_items}: "
                        + ", ".join(str(v) for v in valuations[i])
                    ),
                    opponent_description="other bidders",
                    action_format=_RA_ACTION_FORMAT.format(
                        n_items=self.n_items, budget=self.budget
                    ),
                ),
                config=self.config,
                agent_idx=i,
                natural_language_message=self.config.natural_language_message,
            )
            for i in range(self.n_players)
        ]

        for rnd in range(self.n_rounds):
            observation = self._build_obs(rnd, bid_history, names)
            resps = [agents[i].act(observation) for i in range(self.n_players)]
            all_bids = [_parse_bids(r["raw_cot"], self.n_items, self.budget) for r in resps]
            # Share each agent's natural-language message with all others
            for i in range(self.n_players):
                for j in range(self.n_players):
                    if i != j:
                        agents[j].context_lines.append(
                            f"[{agents[i].name}]: {resps[i]['visible_message']}"
                        )

            round_record = {}
            for item_idx in range(self.n_items):
                item_bids = [all_bids[p][item_idx] for p in range(self.n_players)]
                winner = item_bids.index(max(item_bids))
                profit = valuations[winner][item_idx] - item_bids[winner]
                total_profits[winner] += profit
                round_record[f"item{item_idx+1}"] = {
                    names[p]: all_bids[p][item_idx] for p in range(self.n_players)
                }

            for i in range(self.n_players):
                turns.append(self._make_turn(
                    turn_idx + i, rnd, agents[i], resps[i],
                    "BIDS:" + ",".join(str(b) for b in all_bids[i])
                ))

            bid_history.append(round_record)
            turn_idx += self.n_players

        return EpisodeData(
            episode_id=self._new_episode_id(),
            buyer_type=None,
            seller_type=None,
            deal_reached=True,
            deal_price=None,
            deal_round=self.n_rounds - 1,
            termination="complete",
            buyer_surplus=total_profits[0],
            seller_surplus=sum(total_profits[1:]) / max(1, self.n_players - 1),
            total_welfare=sum(total_profits),
            buyer_reward=total_profits[0],
            seller_reward=total_profits[-1],
            turns=turns,
        )

    def _build_obs(self, rnd: int, history: list[dict], names: list) -> str:
        lines = [f"=== AUCTION — Round {rnd + 1}/{self.n_rounds} | Items: {self.n_items} | Budget: {self.budget} ==="]
        if history:
            lines.append("Past bids (all items):")
            for r_idx, record in enumerate(history):
                lines.append(f"  Round {r_idx + 1}:")
                for item_key, bids in record.items():
                    row = ", ".join(f"{n}={bids[n]}" for n in names)
                    lines.append(f"    {item_key}: {row}")
        lines.append("\nSubmit your bids for this round:")
        return "\n".join(lines)


def _parse_bids(text: str, n_items: int, budget: int) -> list[int]:
    m = re.search(r"BIDS:\s*([\d,\s]+)", text, re.IGNORECASE)
    if m:
        parts = [int(x.strip()) for x in m.group(1).split(",") if x.strip().isdigit()]
        if len(parts) == n_items:
            total = sum(parts)
            if total > budget:
                # Scale down proportionally
                parts = [int(b * budget / total) for b in parts]
            return parts
    # Fallback: spread budget evenly
    base = budget // n_items
    return [base] * n_items


# ── Haggling Game ───────────────────────────────────────────────────────────

_HAGGLING_RULES = """\
- This is a bilateral price negotiation between a BUYER and a SELLER.
- The item's price is unknown to both parties — each has a private reservation price.
  BUYER's reservation price: maximum price they're willing to pay (private).
  SELLER's reservation price: minimum price they'll accept (private).
- The buyer makes the first offer. Players alternate: offer → accept/counter → ...
- A deal is struck when one party replies ACCEPT to the other's offer.
- If no deal after {max_rounds} rounds, both walk away with nothing.
- The negotiation zone (ZOPA) exists — a mutually beneficial deal is possible."""

_HAGGLING_BUYER_PRIVATE = "Your maximum willingness to pay: {max_price}. Do NOT reveal this."
_HAGGLING_SELLER_PRIVATE = "Your minimum acceptable price: {min_price}. Do NOT reveal this."

_HAGGLING_ACTION_FORMAT = """\
ACTION FORMAT: last line must be exactly one of:
  OFFER: X      (propose price X)
  ACCEPT        (accept the counterparty's last offer)
  REJECT        (walk away — ends the negotiation with no deal)"""


class HagglingGame(BaseGame):
    """
    2-player bilateral price negotiation (Stage 1 lightweight variant).

    Differs from A2ANegotiation (env.py) in that it has no patience discount,
    simpler reward structure, and is self-contained — designed to inject
    negotiation-context CoT diversity into Stage 1 training data.

    ToM task: estimate opponent's reservation price from their offers/counter-offers
    to decide when to push harder vs. concede to close the deal.
    """

    def __init__(
        self,
        config: LASHConfig,
        max_rounds: int = 4,
        price_range: tuple[float, float] = (50.0, 200.0),
    ):
        super().__init__(config)
        self.max_rounds = max_rounds
        self.price_range = price_range

    def run(self, seed: Optional[int] = None) -> EpisodeData:
        import random
        rng = random.Random(seed)

        lo, hi = self.price_range
        min_price = rng.uniform(lo, hi * 0.6)
        max_price = rng.uniform(min_price * 1.1, hi)

        rules = _HAGGLING_RULES.format(max_rounds=self.max_rounds)

        buyer = StrategicGameAgent(
            name="Buyer",
            system_prompt=_build_system_prompt(self.config,
                agent_name="Buyer",
                game_name="Haggling",
                game_rules=rules,
                private_info=_HAGGLING_BUYER_PRIVATE.format(max_price=round(max_price, 1)),
                opponent_description="Seller",
                action_format=_HAGGLING_ACTION_FORMAT,
            ),
            config=self.config,
            agent_idx=0,
            natural_language_message=self.config.natural_language_message,
        )
        seller = StrategicGameAgent(
            name="Seller",
            system_prompt=_build_system_prompt(self.config,
                agent_name="Seller",
                game_name="Haggling",
                game_rules=rules,
                private_info=_HAGGLING_SELLER_PRIVATE.format(min_price=round(min_price, 1)),
                opponent_description="Buyer",
                action_format=_HAGGLING_ACTION_FORMAT,
            ),
            config=self.config,
            agent_idx=1,
            natural_language_message=self.config.natural_language_message,
        )

        turns: list[TurnData] = []
        turn_idx = 0
        deal_reached = False
        deal_price: Optional[float] = None
        deal_round: Optional[int] = None
        last_offer: Optional[float] = None
        active_agents = [buyer, seller]

        for rnd in range(self.max_rounds):
            for agent_idx, agent in enumerate(active_agents):
                is_buyer = agent_idx == 0
                if rnd == 0 and agent_idx == 1:
                    # Seller skips first half-round (buyer opens)
                    continue

                if rnd == 0 and agent_idx == 0:
                    obs = (
                        f"=== HAGGLING — Round {rnd + 1}/{self.max_rounds} ===\n"
                        f"You go first. Make your opening offer."
                    )
                else:
                    obs = (
                        f"=== HAGGLING — Round {rnd + 1}/{self.max_rounds} ===\n"
                        f"{'Buyer' if not is_buyer else 'Seller'}'s last move: "
                        f"{'OFFER: ' + str(round(last_offer, 1)) if last_offer is not None else 'no offer yet'}\n"
                        f"Your response:"
                    )

                resp = agent.act(obs)
                action_text = resp["action_text"].upper()  # private — game engine only

                if "ACCEPT" in action_text and last_offer is not None:
                    deal_reached = True
                    deal_price = last_offer
                    deal_round = rnd
                    turns.append(self._make_turn(turn_idx, rnd, agent, resp, "ACCEPT"))
                    turn_idx += 1
                    # Share final natural-language message with other agent
                    other = active_agents[1 - agent_idx]
                    other.context_lines.append(f"[{agent.name}]: {resp['visible_message']}")
                    break
                elif "REJECT" in action_text:
                    turns.append(self._make_turn(turn_idx, rnd, agent, resp, "REJECT"))
                    turn_idx += 1
                    break
                else:
                    price = resp["numeric_action"]
                    if price is not None:
                        last_offer = price
                    turns.append(self._make_turn(
                        turn_idx, rnd, agent, resp,
                        f"OFFER:{round(last_offer, 1)}" if last_offer else "OFFER:unknown"
                    ))
                    turn_idx += 1

                    # Share natural-language message (not raw action) with other agent
                    other = active_agents[1 - agent_idx]
                    other.context_lines.append(f"[{agent.name}]: {resp['visible_message']}")

            else:
                continue
            break

        # Compute rewards
        if deal_reached and deal_price is not None:
            buyer_surplus = max(0.0, max_price - deal_price)
            seller_surplus = max(0.0, deal_price - min_price)
        else:
            buyer_surplus = 0.0
            seller_surplus = 0.0

        return EpisodeData(
            episode_id=self._new_episode_id(),
            buyer_type=None,
            seller_type=None,
            deal_reached=deal_reached,
            deal_price=deal_price,
            deal_round=deal_round,
            termination="accept" if deal_reached else "reject" if any(
                t.action == "REJECT" for t in turns
            ) else "max_rounds",
            buyer_surplus=buyer_surplus,
            seller_surplus=seller_surplus,
            total_welfare=buyer_surplus + seller_surplus,
            buyer_reward=buyer_surplus,
            seller_reward=seller_surplus,
            turns=turns,
        )


# ── Stag Hunt Game ───────────────────────────────────────────────────────────

_SH_RULES = """\
- Each round, both players simultaneously and independently choose: STAG or HARE.
- Payoffs per round:
    Both choose STAG  → each gets 4  (best joint outcome, but requires mutual trust)
    Both choose HARE  → each gets 2  (safe, guaranteed)
    You choose STAG, they choose HARE → you get 0, they get 2
    You choose HARE, they choose STAG → you get 2, they get 0
- Unlike Prisoner's Dilemma, there is NO incentive to defect on a cooperator.
  The only risk is coordination failure — choosing STAG when your partner plays HARE.
- Game lasts {n_rounds} rounds. Maximise total payoff."""

_SH_ACTION_FORMAT = """\
ACTION FORMAT: respond with exactly one word on the last line:
  STAG
  HARE"""


class StagHuntGame(BaseGame):
    """
    2-player repeated Stag Hunt (coordination game).

    Unlike Prisoner's Dilemma, defection has no temptation payoff vs. a cooperator —
    the only issue is whether both agents can coordinate on the (STAG, STAG) equilibrium
    despite risk. This tests whether agents can infer coordination intent from history.

    Intended as a held-out evaluation game: train on PD/PGG/RA/Haggling,
    then test zero-shot ToM transfer on StagHunt (different belief structure).
    """

    def __init__(self, config: LASHConfig, n_rounds: int = 6):
        super().__init__(config)
        self.n_rounds = n_rounds

    def run(self, seed: Optional[int] = None) -> EpisodeData:
        names = ["Alpha", "Beta"]
        turns: list[TurnData] = []
        payoffs = [0.0, 0.0]
        turn_idx = 0
        last_actions: list[Optional[str]] = [None, None]

        agents = [
            StrategicGameAgent(
                name=names[i],
                system_prompt=_build_system_prompt(self.config,
                    agent_name=names[i],
                    game_name="Stag Hunt",
                    game_rules=_SH_RULES.format(n_rounds=self.n_rounds),
                    private_info="No private information — this is a symmetric coordination game.",
                    opponent_description=names[1 - i],
                    action_format=_SH_ACTION_FORMAT,
                ),
                config=self.config,
                agent_idx=i,
                natural_language_message=self.config.natural_language_message,
            )
            for i in range(2)
        ]

        for rnd in range(self.n_rounds):
            obs = self._build_obs(rnd, last_actions, names)
            resps = [agents[i].act(obs) for i in range(2)]
            actions = [_parse_sh_action(r["raw_cot"]) for r in resps]

            for i in range(2):
                turns.append(self._make_turn(turn_idx + i, rnd, agents[i], resps[i], actions[i]))
                # Share opponent's natural-language message for next round's context
                agents[1 - i].context_lines.append(f"[{agents[i].name}]: {resps[i]['visible_message']}")

            p0, p1 = _sh_payoff(actions[0], actions[1])
            payoffs[0] += p0
            payoffs[1] += p1
            last_actions = actions
            turn_idx += 2

        return EpisodeData(
            episode_id=self._new_episode_id(),
            buyer_type=None,
            seller_type=None,
            deal_reached=True,
            deal_price=None,
            deal_round=self.n_rounds - 1,
            termination="complete",
            buyer_surplus=payoffs[0],
            seller_surplus=payoffs[1],
            total_welfare=sum(payoffs),
            buyer_reward=payoffs[0],
            seller_reward=payoffs[1],
            turns=turns,
        )

    def _build_obs(self, rnd: int, last_actions: list, names: list) -> str:
        lines = [f"=== STAG HUNT — Round {rnd + 1}/{self.n_rounds} ==="]
        if last_actions[0] is not None:
            lines.append(f"Last round: {names[0]}={last_actions[0]}, {names[1]}={last_actions[1]}")
        lines.append("\nYour choice for this round:")
        return "\n".join(lines)


def _parse_sh_action(text: str) -> str:
    upper = text.upper()
    if "STAG" in upper:
        return "STAG"
    if "HARE" in upper:
        return "HARE"
    return "HARE"   # conservative default


def _sh_payoff(a0: str, a1: str) -> tuple[float, float]:
    if a0 == "STAG" and a1 == "STAG":
        return 4.0, 4.0
    if a0 == "HARE" and a1 == "HARE":
        return 2.0, 2.0
    if a0 == "STAG" and a1 == "HARE":
        return 0.0, 2.0
    return 2.0, 0.0   # a0==HARE, a1==STAG


# ── Concordia integration (optional) ───────────────────────────────────────

try:
    import concordia  # noqa: F401
    _CONCORDIA_AVAILABLE = True
except ImportError:
    _CONCORDIA_AVAILABLE = False


class ConcordiaScenarioRunner:
    """
    Wraps an existing Concordia GameMaster with LASH-structured agents.
    Requires `pip install concordia`.

    Usage:
        runner = ConcordiaScenarioRunner(config)
        episode = runner.run(game_master, agent_configs)
    """

    def __init__(self, config: LASHConfig):
        if not _CONCORDIA_AVAILABLE:
            raise ImportError(
                "Concordia not installed. Run: pip install concordia\n"
                "For standalone games, use PrisonersDilemmaGame, "
                "PublicGoodsGame, or ResourceAllocationGame instead."
            )
        self.config = config

    def run(self, game_master, agent_configs: list[dict]) -> EpisodeData:
        """
        Inject LASH-structured agents into a Concordia GameMaster and run.

        agent_configs: list of dicts with keys:
          name, role, private_info, game_rules, opponent_description, action_format
        """
        agents = [
            StrategicGameAgent(
                name=cfg["name"],
                system_prompt=_build_system_prompt(self.config, **cfg),
                config=self.config,
                agent_idx=i,
                natural_language_message=self.config.natural_language_message,
            )
            for i, cfg in enumerate(agent_configs)
        ]

        turns: list[TurnData] = []
        turn_idx = 0

        # Run Concordia simulation, hooking into each agent's act()
        for step in game_master.steps():
            for agent in agents:
                if step.actor == agent.name:
                    resp = agent.act(step.observation)
                    game_master.receive_action(agent.name, resp["visible_message"])
                    turns.append(TurnData(
                        turn=turn_idx,
                        round=step.round,
                        role=agent.name,
                        context=resp["context"],
                        raw_cot=resp["raw_cot"],
                        belief_text=resp["belief_text"],
                        intention_text=resp["intention_text"],
                        visible_message=resp["visible_message"],
                        action=resp["visible_message"],
                        price=resp["numeric_action"],
                    ))
                    turn_idx += 1

        return EpisodeData(
            episode_id=str(uuid.uuid4())[:8],
            buyer_type=None,
            seller_type=None,
            deal_reached=True,
            deal_price=None,
            deal_round=None,
            termination="concordia_complete",
            buyer_surplus=0.0,
            seller_surplus=0.0,
            total_welfare=0.0,
            buyer_reward=0.0,
            seller_reward=0.0,
            turns=turns,
        )


# ── Multi-game collection helper ────────────────────────────────────────────

GAME_REGISTRY: dict[str, type[BaseGame]] = {
    "prisoners_dilemma": PrisonersDilemmaGame,
    "public_goods": PublicGoodsGame,
    "resource_allocation": ResourceAllocationGame,
    "haggling": HagglingGame,
    "stag_hunt": StagHuntGame,
}


def run_multitask_collection(
    config: LASHConfig,
    n_episodes_per_game: int,
    output_dir: str,
    games: Optional[list[str]] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> dict[str, int]:
    """
    Run N episodes for each game type and save to output_dir.

    Returns dict of {game_name: training_pairs_collected}.
    """
    import traceback

    games = games or list(GAME_REGISTRY.keys())
    stats: dict[str, int] = {}

    for game_name in games:
        game_cls = GAME_REGISTRY[game_name]
        game = game_cls(config)
        pairs_total = 0

        if verbose:
            print(f"\n── {game_name} ({n_episodes_per_game} episodes) ──")

        for i in range(n_episodes_per_game):
            ep_seed = (seed + i) if seed is not None else None
            try:
                ep = game.run(seed=ep_seed)
                append_episode(ep, output_dir)
                n = append_training_pairs(ep, output_dir)
                pairs_total += n
                if verbose:
                    print(
                        f"  [{i+1:>3}/{n_episodes_per_game}] {ep.episode_id}"
                        f"  welfare={ep.total_welfare:>7.1f}  pairs={n}"
                    )
            except KeyboardInterrupt:
                print("\nInterrupted.")
                stats[game_name] = pairs_total
                return stats
            except Exception:
                print(f"  [{i+1:>3}/{n_episodes_per_game}] ERROR — skipping")
                traceback.print_exc()

        stats[game_name] = pairs_total

    return stats
