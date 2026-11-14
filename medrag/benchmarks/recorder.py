"""
Prediction Recorder and Experiment Artifact Storage.
Handles streaming JSONL writes, failure recovery, resuming interrupted runs,
and saving complete experiment manifests and metrics.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Set
import yaml

from ..config_schema import ExperimentConfig
from .manifest import ExperimentManifest
from .schema import PredictionRecord, RetrievalTrace


class PredictionRecorder:
    """
    Manages experiment file system layout and atomic streaming writes.
    Layout:
        runs/
            <experiment_id>/
                config.yaml
                manifest.json
                predictions.jsonl
                retrieval_traces.jsonl
                metrics.json
                logs/
                    run.log
    """

    def __init__(self, run_dir: str, experiment_id: str) -> None:
        self.experiment_id = experiment_id
        self.run_dir = os.path.join(run_dir, experiment_id)
        self.logs_dir = os.path.join(self.run_dir, "logs")

        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.config_path = os.path.join(self.run_dir, "config.yaml")
        self.manifest_path = os.path.join(self.run_dir, "manifest.json")
        self.predictions_path = os.path.join(self.run_dir, "predictions.jsonl")
        self.traces_path = os.path.join(self.run_dir, "retrieval_traces.jsonl")
        self.metrics_path = os.path.join(self.run_dir, "metrics.json")
        self.log_path = os.path.join(self.logs_dir, "run.log")

        self.completed_ids: Set[str] = self._load_completed_ids()

    def _load_completed_ids(self) -> Set[str]:
        """Loads already-evaluated example IDs to enable resume/recovery."""
        completed: Set[str] = set()
        if os.path.exists(self.predictions_path):
            with open(self.predictions_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            eid = record.get("example_id")
                            if eid:
                                completed.add(eid)
                        except json.JSONDecodeError:
                            continue
        return completed

    def is_completed(self, example_id: str) -> bool:
        """Checks whether an example has already been evaluated in this experiment."""
        return example_id in self.completed_ids

    def save_config(self, config: ExperimentConfig) -> None:
        """Saves configuration as YAML (ensures credentials are not written)."""
        data = config.model_dump()
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    def save_manifest(self, manifest: ExperimentManifest) -> None:
        """Saves experiment manifest JSON."""
        manifest.save(self.manifest_path)

    def record_prediction(self, record: PredictionRecord) -> None:
        """Appends a prediction record to predictions.jsonl with immediate flush."""
        with open(self.predictions_path, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
            f.flush()
        self.completed_ids.add(record.example_id)

    def record_retrieval_traces(self, traces: List[RetrievalTrace]) -> None:
        """Appends retrieval traces to retrieval_traces.jsonl with immediate flush."""
        if not traces:
            return
        with open(self.traces_path, "a", encoding="utf-8") as f:
            for t in traces:
                f.write(t.model_dump_json() + "\n")
            f.flush()

    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Saves computed metrics to metrics.json."""
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    def load_all_predictions(self) -> List[PredictionRecord]:
        """Reads all evaluated prediction records from predictions.jsonl."""
        records: List[PredictionRecord] = []
        if not os.path.exists(self.predictions_path):
            return records
        with open(self.predictions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(PredictionRecord.model_validate_json(line))
                    except Exception:
                        continue
        return records
