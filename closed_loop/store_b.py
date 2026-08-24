"""Store B wrapper: same JSONL as DecisionMemory, ranked by later outcome quality."""

from __future__ import annotations

import json

from agent_loop.memory import DecisionMemory


class OutcomeMemory(DecisionMemory):
    """cycle() calls like() / append(). Wins first, then losses as warnings."""

    def like(self, class_name: str, limit: int = 3) -> list[dict]:
        if not class_name or class_name == "good_print":
            return []
        hits: list[dict] = []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        for line in lines:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("class") == class_name:
                hits.append(item)
        hits.sort(
            key=lambda row: (
                float(row.get("outcome_quality") or 0.5),
                str(row.get("timestamp") or row.get("layer") or ""),
            ),
            reverse=True,
        )
        wins = [h for h in hits if float(h.get("outcome_quality") or 0.5) >= 0.7]
        losses = [h for h in hits if float(h.get("outcome_quality") or 0.5) < 0.4]
        ordered = wins + losses + [h for h in hits if h not in wins and h not in losses]
        return ordered[: max(1, int(limit))]
