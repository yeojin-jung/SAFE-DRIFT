#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one command from a SAFE-DRIFT pipeline manifest.")
    parser.add_argument("manifest")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--name")
    selector.add_argument("--index", type=int)
    selector.add_argument("--first-kind")
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--status-file", required=True)
    return parser.parse_args()


def select_entry(entries: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    if args.name is not None:
        matches = [entry for entry in entries if entry.get("name") == args.name]
        if len(matches) != 1:
            raise ValueError(f"Expected one manifest entry named {args.name!r}, found {len(matches)}.")
        return matches[0]
    if args.index is not None:
        return entries[args.index]
    return next(entry for entry in entries if entry.get("kind") == args.first_kind)


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = select_entry(list(manifest["run_commands"]), args)
    command = [str(part) for part in entry["command"]]
    log_path = Path(args.log_file)
    status_path = Path(args.status_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.setdefault("USE_TF", "0")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    started_at = time.time()
    with log_path.open("a", encoding="utf-8") as log_handle:
        print(f"[manifest runner] name={entry['name']}", file=log_handle, flush=True)
        print(f"[manifest runner] command={shlex.join(command)}", file=log_handle, flush=True)
        completed = subprocess.run(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
    status = {
        "name": entry["name"],
        "kind": entry.get("kind"),
        "return_code": int(completed.returncode),
        "started_at_unix": started_at,
        "finished_at_unix": time.time(),
        "wallclock_seconds": time.time() - started_at,
        "log_file": str(log_path.resolve()),
        "command": command,
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
