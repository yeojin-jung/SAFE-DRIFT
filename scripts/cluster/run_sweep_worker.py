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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def manifest_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if payload.get("prepared_shared_selector_cache_only"):
        return True
    runs = payload.get("runs")
    return bool(runs) and all(int(run.get("train_return_code", 1)) == 0 for run in runs)


def safe_slug(value: Any) -> str:
    return (
        "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in str(value)
        ).strip("_")
        or "item"
    )


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_entries(
    entries: list[dict[str, Any]],
    *,
    phase: str,
    log_dir: Path,
    status_dir: Path,
    stop_on_error: bool,
) -> list[dict[str, Any]]:
    results = []
    environment = os.environ.copy()
    environment.setdefault("USE_TF", "0")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("HF_HOME", "/lambda/nfs/mem/safe-drift/hf_cache")
    environment.setdefault(
        "HUGGINGFACE_HUB_CACHE",
        "/lambda/nfs/mem/safe-drift/hf_cache/hub",
    )
    environment.setdefault("MPLCONFIGDIR", "/lambda/nfs/mem/safe-drift/mplconfig")

    for index, entry in enumerate(entries):
        name = str(entry.get("name") or f"{phase}_{index}")
        done_file = Path(str(entry.get("done_file") or ""))
        if done_file and manifest_complete(done_file):
            result = {
                "name": name,
                "phase": phase,
                "index": index,
                "return_code": 0,
                "skipped_complete": True,
            }
            results.append(result)
            save_json(status_dir / f"{index:04d}_{safe_slug(name)}.json", result)
            continue

        command = [str(part) for part in entry["command"]]
        log_path = log_dir / phase / f"{index:04d}_{safe_slug(name)}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with log_path.open("a", encoding="utf-8") as log_handle:
            print(f"[worker] phase={phase} index={index} name={name}", file=log_handle)
            print(f"[worker] command={shlex.join(command)}", file=log_handle)
            log_handle.flush()
            completed = subprocess.run(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
                check=False,
            )
        result = {
            "name": name,
            "phase": phase,
            "index": index,
            "return_code": int(completed.returncode),
            "skipped_complete": False,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "wallclock_seconds": time.time() - started,
            "log_file": str(log_path),
        }
        results.append(result)
        save_json(status_dir / f"{index:04d}_{safe_slug(name)}.json", result)
        if completed.returncode != 0 and stop_on_error:
            break
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one two-phase SAFE-DRIFT sweep shard with a shared barrier."
    )
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--cache-prep-file", type=Path, required=True)
    parser.add_argument("--runs-file", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    worker_root = args.state_dir / f"worker_{args.worker_id}"
    status_dir = worker_root / "statuses"
    log_dir = worker_root / "logs"
    worker_status_path = worker_root / "worker_status.json"
    save_json(
        worker_status_path,
        {
            "worker_id": args.worker_id,
            "phase": "cache_prep",
            "started_at_unix": time.time(),
        },
    )
    prep_results = run_entries(
        load_jsonl(args.cache_prep_file),
        phase="cache_prep",
        log_dir=log_dir,
        status_dir=status_dir / "cache_prep",
        stop_on_error=True,
    )
    prep_ok = all(int(result["return_code"]) == 0 for result in prep_results)
    save_json(
        args.state_dir / "barrier" / f"worker_{args.worker_id}.json",
        {"worker_id": args.worker_id, "success": prep_ok, "results": prep_results},
    )
    if not prep_ok:
        save_json(
            worker_status_path,
            {"worker_id": args.worker_id, "phase": "failed_cache_prep"},
        )
        return 1

    save_json(
        worker_status_path,
        {"worker_id": args.worker_id, "phase": "waiting_for_cache_barrier"},
    )
    barrier_dir = args.state_dir / "barrier"
    while True:
        barrier_files = [
            barrier_dir / f"worker_{worker}.json"
            for worker in range(args.worker_count)
        ]
        if all(path.exists() for path in barrier_files):
            barrier_payloads = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in barrier_files
            ]
            if not all(bool(payload.get("success")) for payload in barrier_payloads):
                save_json(
                    worker_status_path,
                    {"worker_id": args.worker_id, "phase": "failed_peer_cache_prep"},
                )
                return 1
            break
        time.sleep(args.poll_seconds)

    save_json(
        worker_status_path,
        {"worker_id": args.worker_id, "phase": "runs"},
    )
    run_results = run_entries(
        load_jsonl(args.runs_file),
        phase="runs",
        log_dir=log_dir,
        status_dir=status_dir / "runs",
        stop_on_error=False,
    )
    failures = [
        result for result in run_results if int(result.get("return_code", 1)) != 0
    ]
    save_json(
        worker_status_path,
        {
            "worker_id": args.worker_id,
            "phase": "complete" if not failures else "complete_with_failures",
            "run_count": len(run_results),
            "failure_count": len(failures),
            "finished_at_unix": time.time(),
        },
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
