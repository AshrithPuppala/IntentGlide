"""Safe local profile persistence, reset, and explicit export."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from personalization.user_profile import UserProfile


class ProfileManager:
    def __init__(self, directory: Path | str = "data/user_profiles", profile_id: str = "default"):
        self.directory = Path(directory)
        if Path(profile_id).name != profile_id or profile_id in {"", ".", ".."}:
            raise ValueError("profile_id must be a simple local filename")
        self.profile_id = profile_id
        self.path = self.directory / f"{profile_id}.json"

    def load(self) -> UserProfile:
        if not self.path.exists():
            profile = UserProfile()
            self.save(profile)
            return profile
        try:
            values = json.loads(self.path.read_text(encoding="utf-8"))
            profile = UserProfile.from_dict(values)
            # Keep the generated internal identifier; filename is only a local slot name.
            return profile
        except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
            return UserProfile()

    def save(self, profile: UserProfile) -> Path:
        profile.validate()
        self.directory.mkdir(parents=True, exist_ok=True)
        data = json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n"
        fd, temp_name = tempfile.mkstemp(prefix="profile-", suffix=".tmp", dir=self.directory)
        try:
            with open(fd, "w", encoding="utf-8", newline="\n", closefd=True) as handle:
                handle.write(data)
            Path(temp_name).replace(self.path)
        except Exception:
            Path(temp_name).unlink(missing_ok=True)
            raise
        return self.path

    def reset(self) -> UserProfile:
        profile = UserProfile()
        self.save(profile)
        return profile

    def export(self, destination: Path | str, profile: UserProfile | None = None) -> Path:
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps((profile or self.load()).to_dict(), indent=2) + "\n", encoding="utf-8")
        return dest
