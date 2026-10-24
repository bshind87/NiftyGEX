"""Code provenance: the git commit that produced an output."""
from __future__ import annotations

import subprocess
from functools import lru_cache

from niftygex.config import REPO_ROOT


@lru_cache(maxsize=1)
def git_provenance() -> dict[str, object]:
    """HEAD commit and whether the working tree has uncommitted or untracked changes (None outside a repo)."""
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"git_commit": commit, "git_dirty": None if status is None else bool(status)}
