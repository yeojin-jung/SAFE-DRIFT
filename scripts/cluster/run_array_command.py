#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def manifest_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if data.get("prepared_shared_selector_cache_only"):
        return True
    runs = data.get("runs")
    if not isinstance(runs, list) or not runs:
        return False
    return all(int(run.get("train_return_code", 1)) == 0 for run in runs)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if not isinstance(item, dict) or not isinstance(item.get("command"), list):
                raise ValueError(f"Invalid command item at {path}:{line_no}")
            items.append(item)
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one command from a JSONL Slurm array manifest.")
    parser.add_argument("--commands-file", required=True)
    parser.add_argument("--task-id", type=int, default=None)
    args = parser.parse_args()

    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    commands_file = Path(args.commands_file).resolve()
    items = load_jsonl(commands_file)
    if task_id < 0 or task_id >= len(items):
        raise IndexError(f"Task id {task_id} is outside 0..{len(items) - 1} for {commands_file}")

    item = items[task_id]
    name = str(item.get("name") or f"task_{task_id}")
    done_file_raw = item.get("done_file")
    if done_file_raw and manifest_complete(Path(str(done_file_raw))):
        print(f"[array-runner] {name}: already complete at {done_file_raw}; skipping.")
        return 0

    command = [str(part) for part in item["command"]]
    print(f"[array-runner] commands_file={commands_file}")
    print(f"[array-runner] task_id={task_id}")
    print(f"[array-runner] name={name}")
    print(f"[array-runner] command={shlex.join(command)}")
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
