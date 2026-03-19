#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#运行示例 python count_cv_stage_distribution_to_csv.py /export/SHELAB/sikaiyue/sleep/TinyUStaging_sky/demo/datasets/processed_SKY/dcsm/views/10_CV
import sys
import os
from pathlib import Path
from collections import Counter
import csv

# 可按你的数据标签习惯修改顺序
STAGE_ORDER = ["W", "N1", "N2", "N3", "R", "REM", "UNK", "UNKNOWN"]


def count_epochs_from_ids(file_path):
    counts = Counter()
    total = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                start, duration, stage = line.split(",")
                duration = int(duration)

                n_epochs = duration // 30
                counts[stage] += n_epochs
                total += n_epochs

                if duration % 30 != 0:
                    print(f"[警告] {file_path} 第{line_num}行 duration={duration} 不是30的整数倍")

            except Exception as e:
                print(f"[错误] 解析文件 {file_path} 第{line_num} 行失败: {line}")
                print(f"       错误信息: {e}")

    return counts, total


def find_ids_files_followlinks(subset_dir):
    ids_files = []
    for root, dirs, files in os.walk(subset_dir, followlinks=True):
        for fname in files:
            if fname.endswith(".ids"):
                ids_files.append(os.path.join(root, fname))
    return sorted(ids_files)


def print_stage_stats(title, counts, total, n_files):
    print(f"\n--- {title} ---")
    print(f"文件数: {n_files}")
    print(f"总epoch数: {total}")

    if total == 0:
        print("无数据")
        return

    printed = set()
    for stage in STAGE_ORDER:
        if stage in counts:
            pct = counts[stage] / total * 100
            print(f"  {stage:8s}: {counts[stage]:6d} ({pct:6.2f}%)")
            printed.add(stage)

    for stage in sorted(counts.keys()):
        if stage not in printed:
            pct = counts[stage] / total * 100
            print(f"  {stage:8s}: {counts[stage]:6d} ({pct:6.2f}%)")


def count_subset(subset_dir):
    ids_files = find_ids_files_followlinks(subset_dir)
    subset_counts = Counter()
    subset_total = 0

    for fp in ids_files:
        c, t = count_epochs_from_ids(fp)
        subset_counts.update(c)
        subset_total += t

    return subset_counts, subset_total, len(ids_files)


def get_all_stage_names(rows):
    stage_names = set()
    for row in rows:
        stage_names.update(row["counts"].keys())

    ordered = [s for s in STAGE_ORDER if s in stage_names]
    remaining = sorted(stage_names - set(ordered))
    return ordered + remaining


def save_csv(rows, out_csv):
    stage_names = get_all_stage_names(rows)

    header = [
        "split",
        "subset",
        "n_files",
        "total_epochs",
    ]

    for stage in stage_names:
        header.append(f"{stage}_count")
        header.append(f"{stage}_pct")

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for row in rows:
            split = row["split"]
            subset = row["subset"]
            n_files = row["n_files"]
            total = row["total"]
            counts = row["counts"]

            line = [split, subset, n_files, total]

            for stage in stage_names:
                c = counts.get(stage, 0)
                pct = (c / total * 100) if total > 0 else 0.0
                line.extend([c, f"{pct:.4f}"])

            writer.writerow(line)


def main():
    if len(sys.argv) not in [2, 3]:
        print("用法:")
        print("  python count_cv_stage_distribution_to_csv.py <10_CV目录> [输出csv路径]")
        sys.exit(1)

    root = Path(sys.argv[1])
    if not root.exists():
        print(f"目录不存在: {root}")
        sys.exit(1)

    if len(sys.argv) == 3:
        out_csv = Path(sys.argv[2])
    else:
        out_csv = root / "stage_distribution_summary.csv"

    split_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("split_")])

    if not split_dirs:
        print(f"在 {root} 下没有找到 split_* 目录")
        sys.exit(1)

    overall = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }
    overall_total = {
        "train": 0,
        "val": 0,
        "test": 0,
    }
    overall_files = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    csv_rows = []

    for split_dir in split_dirs:
        print("\n" + "=" * 70)
        print(f"{split_dir.name}")
        print("=" * 70)

        for subset in ["train", "val", "test"]:
            subset_dir = split_dir / subset

            # 兼容 validation 命名
            if not subset_dir.exists() and subset == "val":
                alt_dir = split_dir / "validation"
                if alt_dir.exists():
                    subset_dir = alt_dir

            counts, total, n_files = count_subset(str(subset_dir))
            print_stage_stats(f"{split_dir.name} / {subset}", counts, total, n_files)

            overall[subset].update(counts)
            overall_total[subset] += total
            overall_files[subset] += n_files

            csv_rows.append({
                "split": split_dir.name,
                "subset": subset,
                "n_files": n_files,
                "total": total,
                "counts": dict(counts),
            })

    print("\n" + "#" * 70)
    print("所有折汇总")
    print("#" * 70)

    for subset in ["train", "val", "test"]:
        print_stage_stats(
            f"ALL FOLDS / {subset}",
            overall[subset],
            overall_total[subset],
            overall_files[subset]
        )

        csv_rows.append({
            "split": "ALL_FOLDS",
            "subset": subset,
            "n_files": overall_files[subset],
            "total": overall_total[subset],
            "counts": dict(overall[subset]),
        })

    save_csv(csv_rows, out_csv)
    print(f"\nCSV已保存到: {out_csv}")


if __name__ == "__main__":
    main()