#!/usr/bin/env python
"""把 Ultralytics 的 results.csv 转成 TensorBoard event 文件（支持增量刷新、不重复写点）。

已存在的实验（baseline / SAR 系列）本来没有装 tensorboard，所以没有 events 文件；
这个脚本把它们补进 TensorBoard，并且可以反复调用：用 .tb_state 记录已写入的 epoch，
只追加新行，因此曲线不会出现重复点。

用法:
    python csv_to_tensorboard.py                      # 转换 runs/visdrone 下所有实验
    python csv_to_tensorboard.py --runs base_n_640 a_dhcd_w10
"""

from __future__ import annotations

import os

import argparse
import csv
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

ROOT = Path(os.environ.get("YOLO_ROOT", Path(__file__).resolve().parent.parent))
SKIP = {"epoch", "time"}


def convert(run_dir: Path, force: bool = False) -> int:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return 0
    tb_dir = run_dir / "tb"
    tb_dir.mkdir(exist_ok=True)
    state_file = tb_dir / ".tb_state"
    last = -1
    if state_file.exists() and not force:
        try:
            last = int(state_file.read_text().strip() or -1)
        except ValueError:
            last = -1

    rows = list(csv.DictReader(open(csv_path)))
    new_rows = [r for r in rows if int(r["epoch"]) > last]
    if not new_rows:
        return 0

    writer = SummaryWriter(log_dir=str(tb_dir))
    # 每个 run 用 tag 前缀区分（TensorBoard 里按 runs/visdrone/<name>/tb 分组，前缀让同名指标可读）
    for r in new_rows:
        step = int(r["epoch"])
        for k, v in r.items():
            if k in SKIP or v in ("", "None", None):
                continue
            try:
                writer.add_scalar(k, float(v), step)  # tag 名与 Ultralytics 原生 TB 写法保持一致
            except (TypeError, ValueError):
                continue
    writer.close()
    state_file.write_text(str(int(new_rows[-1]["epoch"])))
    return len(new_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="实验名（runs/visdrone/<name>）；默认全部")
    ap.add_argument("--force", action="store_true", help="忽略 state，整表重写")
    a = ap.parse_args()

    base = Path(os.environ.get("RUNS_DIR", ROOT / "results/results_csv"))
    names = a.runs or sorted(p.name for p in base.iterdir() if (p / "results.csv").exists())
    total = 0
    for n in names:
        if a.force:  # 清掉旧 events，避免重复点
            for f in (base / n / "tb").glob("events.out.tfevents.*"):
                f.unlink()
        k = convert(base / n, force=a.force)
        if k:
            print(f"  {n}: +{k} epoch")
        total += k
    print(f"共写入 {total} 个 epoch 的点" if total else "无新增（已是最新）")


if __name__ == "__main__":
    main()
