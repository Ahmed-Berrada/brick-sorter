"""REASON: inspection outcome -> bowl, with re-inspection of uncertain bricks. Pure logic, no I/O."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    action: str  # "place" | "reinspect"
    bowl: str | None
    outcome: str  # plain | circle | cross | uncertain
    reason: str
    forced: bool = False  # bowl chosen by the uncertainty fallback


MEANING = {"plain": "good", "circle": "minor defect", "cross": "defective"}


def decide(outcome: str, attempt: int, rules: dict) -> Decision:
    """`attempt` is the 1-based number of the inspection just made for this brick."""
    if outcome in ("plain", "circle", "cross"):
        bowl = rules[outcome]
        return Decision("place", bowl, outcome, f"{outcome} ({MEANING[outcome]}) -> {bowl} bowl")
    if attempt < rules["max_uncertain"]:
        return Decision("reinspect", None, "uncertain",
                        f"uncertain (inspection {attempt}/{rules['max_uncertain']}) -> look again")
    fb = rules["fallback"]
    return Decision("place", fb, "uncertain",
                    f"still uncertain after {attempt} inspections -> {fb} bowl (safe default)", forced=True)


def required_poses(rules: dict) -> list[str]:
    return ["home"] + sorted({rules["plain"], rules["circle"], rules["cross"], rules["fallback"]})
