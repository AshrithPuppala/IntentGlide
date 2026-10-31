"""CSV benchmark logger; processing time and end-to-end latency stay distinct."""
from __future__ import annotations

import csv
from pathlib import Path


FIELDS = [
    "timestamp", "mode", "raw_x", "raw_y", "output_x", "output_y",
    "controller_state", "target_id", "confidence", "processing_time_ms",
    "end_to_end_latency_ms", "task_success", "wrong_target",
]


class BenchmarkLogger:
    def __init__(self, output_path: str = "evaluation/results/benchmark.csv"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()

    def log(self, **row):
        self._writer.writerow({key: row.get(key) for key in FIELDS})
        self._file.flush()

    def close(self):
        if not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
