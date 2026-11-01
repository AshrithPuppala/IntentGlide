"""Append-only local interaction outcomes; no per-frame telemetry is written."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid


class SessionLogger:
    def __init__(self, directory: Path | str = "data/interaction_sessions", profile_id: str = "local-anonymous"):
        self.directory = Path(directory)
        self.session_id = "run-" + uuid.uuid4().hex[:12]
        self.profile_id = profile_id
        self.path = self.directory / f"{self.session_id}.jsonl"

    def log(self, event: str, **fields):
        self.directory.mkdir(parents=True, exist_ok=True)
        record = {"schema_version": 1, "session_id": self.session_id,
                  "profile_id": self.profile_id, "event": event,
                  "logged_at_utc": datetime.now(timezone.utc).isoformat(), **fields}
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

