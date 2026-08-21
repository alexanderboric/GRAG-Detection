"""Plain-git-backed versioning for GraphRAG output folders - no DVC. Each repo_root is its own
small, isolated, disposable git repo (e.g. one per dataset's output folder), not a shared/synced
store of large files, so git alone is enough and DVC's dedup/remote sync would be pure overhead.
"""

import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class VersionStoreConfig:
    """repo_root: git repo root. tracked_folders: paths (relative to repo_root) snapshotted on
    every save()."""
    repo_root: str | Path
    tracked_folders: list[str | Path] = field(default_factory=list)


_store: "CorpusVersionStore | None" = None


def init(config: VersionStoreConfig) -> "CorpusVersionStore":
    """Initializes the global CorpusVersionStore. Call once at startup before save()/restore()."""
    global _store
    _store = CorpusVersionStore(config)
    return _store


def get_store() -> "CorpusVersionStore":
    if _store is None:
        raise RuntimeError("Call version.init(config) first.")
    return _store


def save(label: str = "") -> str:
    return get_store().save(label)


def restore(tag: str | None = None) -> str:
    return get_store().restore(tag)


def list_versions() -> list[dict]:
    return get_store().list_versions()


class CorpusVersionStore:
    """Wraps plain git for atomic save/restore of a set of tracked folders. Prefer the
    module-level init()/save()/restore() over instantiating this directly."""

    def __init__(self, config: VersionStoreConfig):
        self.root = Path(config.repo_root).resolve()
        self.tracked_folders: list[Path] = [Path(f).resolve() for f in config.tracked_folders]
        if not self.tracked_folders:
            raise ValueError("config.tracked_folders must contain at least one path.")

        if not (self.root / ".git").exists():
            print(f"Initializing git repository at {self.root}...")
            try:
                subprocess.run(["git", "init"], cwd=self.root, check=True, capture_output=True)
            except FileNotFoundError:
                print("ERROR: git is not installed. Please install git first.")
                raise

        for folder in self.tracked_folders:
            if not folder.exists():
                print(f"Creating tracked folder: {folder}...")
                folder.mkdir(parents=True, exist_ok=True)

    def save(self, label: str = "") -> str:
        """Snapshots all tracked folders and returns the version tag ("YYYYMMDDTHHMMSS[_label]"),
        e.g. "20260518T143012_pre_ingest" - pass it to restore() to roll back to this point."""
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        tag = (f"{ts}_{label}" if label else ts).replace(" ", "_")

        rel_folders = [str(f.relative_to(self.root)) for f in self.tracked_folders]
        self._run(["git", "add"] + rel_folders)

        commit_msg = f"snapshot: {tag}"
        result = self._run(["git", "commit", "-m", commit_msg], check=False)
        if result.returncode not in (0, 1):  # 1 = "nothing to commit", not an error
            result.check_returncode()

        self._run(["git", "tag", "-a", tag, "-m", commit_msg])  # annotated, so list_versions() can read the message back
        return tag

    def restore(self, tag: str | None = None) -> str:
        """Restores all tracked folders to `tag` (latest if None), in place - never moves HEAD or
        detaches, since callers may restore repeatedly (e.g. once per detector run) and shouldn't
        accumulate detached-HEAD states."""
        if tag is None:
            tag = self._latest_tag()
            if tag is None:
                raise RuntimeError("No snapshots found. Call save() first.")

        rel_folders = [str(f.relative_to(self.root)) for f in self.tracked_folders]
        self._run(["git", "checkout", tag, "--"] + rel_folders)
        self._run(["git", "clean", "-fd", "--"] + rel_folders)  # remove files created since the tag
        return tag

    def list_versions(self) -> list[dict]:
        """Returns every snapshot newest-first, as {"tag", "date", "message"}."""
        result = self._run(
            ["git", "tag", "--sort=-creatordate", "--format=%(refname:short)|%(creatordate:iso)|%(subject)"],
            capture=True,
        )
        versions = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("|", 2)
            if len(parts) == 3:
                versions.append({"tag": parts[0], "date": parts[1].strip(), "message": parts[2].strip()})
        return versions

    def _run(self, cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd, cwd=str(self.root), check=check, text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )

    def _latest_tag(self) -> str | None:
        result = self._run(["git", "describe", "--tags", "--abbrev=0"], check=False, capture=True)
        return result.stdout.strip() or None


if __name__ == "__main__":
    # Manual CLI for testing save/restore/list without wiring up a caller.
    import sys
    import version as v

    config = VersionStoreConfig(repo_root=".", tracked_folders=["data/input", "data/output"])
    v.init(config)

    if len(sys.argv) < 2:
        print("Usage: python version.py save [label]")
        print("       python version.py restore [tag]")
        print("       python version.py list")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "save":
        tag = v.save(label=sys.argv[2] if len(sys.argv) > 2 else "snapshot")
        print(f"Saved: {tag}")
    elif cmd == "restore":
        restored = v.restore(sys.argv[2] if len(sys.argv) > 2 else None)
        print(f"Restored: {restored}")
    elif cmd == "list":
        for ver in v.list_versions():
            print(f"{ver['tag']:40s}  {ver['date']}  {ver['message']}")
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
