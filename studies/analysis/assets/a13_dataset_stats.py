"""A13 — dataset & phase-label statistics per task (from data_provenance.json)."""

from __future__ import annotations

from pathlib import Path

from studies.analysis.common import io as cio
from studies.analysis.common import registry
from studies.analysis.dataset import AnalysisDataset
from studies.analysis.render.tables import Table, save_table


def _provenance_for(dataset: AnalysisDataset, task: str) -> dict:
    for (t, name, seed, stage), run in dataset.train_runs.items():
        if t == task:
            path = run.path / "metadata" / "data_provenance.json"
            if path.is_file():
                data = cio.read_json(path)
                if isinstance(data, dict):
                    return data
    return {}


def generate(dataset: AnalysisDataset) -> list[Path]:
    rows = []
    for task in registry.tasks():
        prov = _provenance_for(dataset, task)
        p = prov.get("provenance", {}) if prov else {}
        state_keys = p.get("state_keys", [])
        state_dim = p.get("state_dim", sum(int(k.get("dim", 0)) for k in state_keys))
        rows.append(
            [
                task,
                str(state_dim),
                str(p.get("action_dim", "--")),
                str(p.get("normalization_strategy", "--")),
                str(p.get("sequence_length", "--")),
                str(p.get("schema_version", "--")),
                str(len(state_keys)),
            ]
        )
    table = Table(
        headers=[
            "Task",
            "State dim",
            "Action dim",
            "Normalization",
            "Seq. len",
            "Schema",
            "State keys",
        ],
        rows=rows,
        caption="Per-task structured-state schema and normalization (data\\_provenance.json).",
        notes=(
            "Phase-label distribution statistics are produced by the ingestion "
            "cache; the frozen phase definitions (6 regimes) are identical "
            "across tasks.",
        ),
    )
    return save_table(table, "tables/A13_dataset_stats")
