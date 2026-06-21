"""Result record serialization and loading."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.benchmark.harness import Metrics

VALID_QUALITY_NOTES = ("coherent", "minor_degradation", "incoherent")


@dataclass
class ScenarioInfo:
    """Metadata that identifies one experiment run."""

    engine: str
    model: str
    quant_level: str
    prompt_tokens: int
    max_new_tokens: int
    seed: int


@dataclass
class OutputInfo:
    """Generated text and quality assessment for one run."""

    generated_text: str
    n_output_tokens: int
    quality_note: str

    def __post_init__(self) -> None:
        if self.quality_note not in VALID_QUALITY_NOTES:
            raise ValueError(
                f"quality_note must be one of {VALID_QUALITY_NOTES}, "
                f"got '{self.quality_note}'"
            )


@dataclass
class ResultRecord:
    """Complete result for one experiment run."""

    scenario: ScenarioInfo
    metrics: Metrics
    output: OutputInfo
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


def save_result(record: ResultRecord, results_dir: str) -> Path:
    """Serialize record to JSON and write it to results_dir."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{record.scenario.engine}_{record.scenario.quant_level}.json"
    out_path = out_dir / fname
    if out_path.exists():
        print(f"[persistence] Warning: overwriting {out_path}")
    data = {
        "scenario": asdict(record.scenario),
        "metrics": asdict(record.metrics),
        "output": asdict(record.output),
        "timestamp": record.timestamp,
    }
    out_path.write_text(json.dumps(data, indent=2))
    return out_path


def load_results(results_dir: str) -> list[ResultRecord]:
    """Load all *.json result files from results_dir."""
    records: list[ResultRecord] = []
    for path in sorted(Path(results_dir).glob("*.json")):
        data = json.loads(path.read_text())
        records.append(
            ResultRecord(
                scenario=ScenarioInfo(**data["scenario"]),
                metrics=Metrics(**data["metrics"]),
                output=OutputInfo(**data["output"]),
                timestamp=data.get("timestamp", ""),
            )
        )
    return records
