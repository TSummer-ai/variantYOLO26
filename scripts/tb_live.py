#!/usr/bin/env python
"""常驻的 TensorBoard 增量刷新器（给没有原生 TB 写入的训练补曲线）。

与 csv_to_tensorboard.py 的区别：**每个 run 只创建一次 SummaryWriter 并复用**。
之前用 shell 循环每 30 秒新建一次 SummaryWriter，会导致 protobuf/libstdc++ 层
SIGABRT（已实测崩溃一次），这个版本避免了该问题，并且异常不会让进程退出。

用法: python tb_live.py [--interval 30]
"""

from __future__ import annotations

import os

import argparse
import csv
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

ROOT = Path(os.environ.get("YOLO_ROOT", Path(__file__).resolve().parent.parent))
BASE = Path(os.environ.get("RUNS_DIR", ROOT / "results/results_csv"))
SKIP = {"epoch", "time"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    writers: dict[str, SummaryWriter] = {}
    last: dict[str, int] = {}
    for run_dir in BASE.iterdir():
        state = run_dir / "tb/.tb_state"
        if state.exists():
            try:
                last[run_dir.name] = int(state.read_text().strip() or -1)
            except ValueError:
                pass

    while True:
        try:
            for run_dir in sorted(BASE.iterdir()):
                csv_path = run_dir / "results.csv"
                if not csv_path.exists():
                    continue
                if any(run_dir.glob("events.out.tfevents.*")):
                    continue  # Ultralytics 自己写了原生 event，不要重复转换（否则 TB 里会重复一条曲线）
                name = run_dir.name
                rows = list(csv.DictReader(open(csv_path)))
                new = [r for r in rows if int(r["epoch"]) > last.get(name, -1)]
                if not new:
                    continue
                if name not in writers:
                    writers[name] = SummaryWriter(log_dir=str(run_dir / "tb"))
                for r in new:
                    step = int(r["epoch"])
                    for k, v in r.items():
                        if k in SKIP or v in ("", "None", None):
                            continue
                        try:
                            writers[name].add_scalar(k, float(v), step)  # 与原生 TB tag 命名一致
                        except (TypeError, ValueError):
                            continue
                writers[name].flush()
                last[name] = int(new[-1]["epoch"])
                (run_dir / "tb/.tb_state").write_text(str(last[name]))
                print(f"[{time.strftime('%T')}] {name}: +{len(new)} -> epoch {last[name]}", flush=True)
        except Exception as e:  # 任何异常都不能让常驻进程退出
            print(f"[{time.strftime('%T')}] scan error: {type(e).__name__}: {e}", flush=True)
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
