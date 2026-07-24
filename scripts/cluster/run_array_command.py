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


def run_item(
    *,
    item: dict[str, Any],
    commands_file: Path,
    item_index: int,
) -> int:
    name = str(item.get("name") or f"task_{item_index}")
    done_file_raw = item.get("done_file")
    if done_file_raw and manifest_complete(Path(str(done_file_raw))):
        print(f"[array-runner] {name}: already complete at {done_file_raw}; skipping.")
        return 0

    command = [str(part) for part in item["command"]]
    print(f"[array-runner] commands_file={commands_file}")
    print(f"[array-runner] item_index={item_index}")
    print(f"[array-runner] name={name}")
    print(f"[array-runner] command={shlex.join(command)}")
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def run_items_in_parallel(
    *,
    items: list[dict[str, Any]],
    commands_file: Path,
    start: int,
    end: int,
) -> int:
    visible_devices_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_devices = [
        device.strip()
        for device in visible_devices_raw.split(",")
        if device.strip() and device.strip() != "NoDevFiles"
    ]
    if not visible_devices:
        visible_devices = [str(index) for index in range(end - start)]
    if len(visible_devices) < end - start:
        raise RuntimeError(
            f"Parallel batch needs {end - start} GPUs but CUDA_VISIBLE_DEVICES="
            f"{visible_devices_raw!r} exposes {len(visible_devices)}."
        )

    processes: list[tuple[int, str, subprocess.Popen[Any]]] = []
    for slot, item_index in enumerate(range(start, end)):
        item = items[item_index]
        name = str(item.get("name") or f"task_{item_index}")
        done_file_raw = item.get("done_file")
        if done_file_raw and manifest_complete(Path(str(done_file_raw))):
            print(f"[array-runner] {name}: already complete at {done_file_raw}; skipping.")
            continue
        command = [str(part) for part in item["command"]]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = visible_devices[slot]
        print(f"[array-runner] commands_file={commands_file}")
        print(f"[array-runner] item_index={item_index}")
        print(f"[array-runner] gpu={visible_devices[slot]}")
        print(f"[array-runner] name={name}")
        print(f"[array-runner] command={shlex.join(command)}")
        processes.append(
            (
                item_index,
                name,
                subprocess.Popen(command, env=env),
            )
        )

    failed = []
    for item_index, name, process in processes:
        return_code = int(process.wait())
        if return_code != 0:
            failed.append((item_index, name, return_code))
    if failed:
        for item_index, name, return_code in failed:
            print(
                f"[array-runner] item_index={item_index} name={name} "
                f"failed with return code {return_code}.",
                file=sys.stderr,
            )
        return failed[0][2]
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one command batch from a JSONL Slurm array manifest.")
    parser.add_argument("--commands-file", required=True)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of consecutive manifest commands executed by each Slurm array task.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run the command batch concurrently, assigning one visible GPU per command.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    commands_file = Path(args.commands_file).resolve()
    items = load_jsonl(commands_file)
    batch_count = (len(items) + args.batch_size - 1) // args.batch_size
    if task_id < 0 or task_id >= batch_count:
        raise IndexError(f"Task id {task_id} is outside 0..{batch_count - 1} for {commands_file}")

    start = task_id * args.batch_size
    end = min(len(items), start + args.batch_size)
    print(
        f"[array-runner] task_id={task_id} batch_size={args.batch_size} "
        f"item_range={start}:{end}"
    )
    if args.parallel:
        return run_items_in_parallel(
            items=items,
            commands_file=commands_file,
            start=start,
            end=end,
        )
    for item_index in range(start, end):
        return_code = run_item(
            item=items[item_index],
            commands_file=commands_file,
            item_index=item_index,
        )
        if return_code != 0:
            print(
                f"[array-runner] item_index={item_index} failed with return code {return_code}.",
                file=sys.stderr,
            )
            return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
