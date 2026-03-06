from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
import pandas as pd

@dataclass
class StepLogger:
    rows: list[dict] = field(default_factory=list)

    def log(self, **kwargs):
        self.rows.append(dict(kwargs))

    def to_csv(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(path, index=False)

def write_summary(path: str | Path, summary: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
