"""Small local preference store for dwell and magnetic assistance."""
from __future__ import annotations

import json
from pathlib import Path


class UserPreferences:
    DEFAULTS = {"dwell_seconds": 3.0, "magnetic_profile": "balanced"}
    PROFILES = {"gentle", "balanced", "strong"}

    def __init__(self, path: Path | None = None):
        self.path = path or Path(__file__).resolve().parents[1] / "data" / "user_preferences.json"
        self.values = dict(self.DEFAULTS)
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            dwell = float(loaded.get("dwell_seconds", 3.0))
            profile = str(loaded.get("magnetic_profile", "balanced"))
            self.values["dwell_seconds"] = min(8.0, max(2.0, dwell))
            if profile in self.PROFILES:
                self.values["magnetic_profile"] = profile
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    def save(self, *, dwell_seconds: float, magnetic_profile: str) -> None:
        self.values = {
            "dwell_seconds": min(8.0, max(2.0, float(dwell_seconds))),
            "magnetic_profile": magnetic_profile if magnetic_profile in self.PROFILES else "balanced",
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.values, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)
