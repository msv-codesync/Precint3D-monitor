"""Store B from the abstract: past cycles on disk. No embeddings required to retrieve by class."""

from __future__ import annotations

import json
from pathlib import Path


class DecisionMemory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def append(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def like(self, class_name: str, limit: int = 3) -> list[dict]:
        if not class_name or class_name == "good_print":
            return []
        hits: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("class") == class_name:
                hits.append(item)
        return hits[-limit:]
