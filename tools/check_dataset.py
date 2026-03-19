import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np


CSV_EXPECTS_DATE_COL = True


def _check_csv(path: Path) -> list[str]:
    problems: list[str] = []
    try:
        df = pd.read_csv(path, nrows=5)
    except Exception as e:
        return [f"无法读取 CSV: {e}"]

    if CSV_EXPECTS_DATE_COL and "date" not in df.columns:
        problems.append("缺少 `date` 列（时间戳列名必须是 date）")

    return problems


def _check_npz(path: Path) -> list[str]:
    problems: list[str] = []
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        return [f"无法读取 NPZ: {e}"]
    if "data" not in data.files:
        problems.append("NPZ 缺少 key `data`（PEMS 数据读取需要 data['data']）")
    return problems


def _check_text_lines(path: Path) -> list[str]:
    problems: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            line = f.readline().strip()
    except Exception as e:
        return [f"无法读取文本: {e}"]
    if not line:
        problems.append("文件为空")
        return problems
    parts = line.split(",")
    try:
        _ = [float(x) for x in parts]
    except Exception:
        problems.append("第一行不是逗号分隔的纯数字（Solar 数据应为纯数值矩阵文本）")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Check iTransformer dataset folder structure and file formats.")
    parser.add_argument("--root", type=str, default="./dataset", help="dataset 根目录（默认 ./dataset）")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[FAIL] dataset 根目录不存在: {root}")
        return 2

    # Minimal expectations for common scripts in this repo
    expected = {
        "traffic": root / "traffic" / "traffic.csv",
        "electricity": root / "electricity" / "electricity.csv",
        "weather": root / "weather" / "weather.csv",
        "exchange_rate": root / "exchange_rate" / "exchange_rate.csv",
        "ETTh1": root / "ETT-small" / "ETTh1.csv",
        "ETTh2": root / "ETT-small" / "ETTh2.csv",
        "ETTm1": root / "ETT-small" / "ETTm1.csv",
        "ETTm2": root / "ETT-small" / "ETTm2.csv",
        "PEMS03": root / "PEMS" / "PEMS03.npz",
        "PEMS04": root / "PEMS" / "PEMS04.npz",
        "PEMS07": root / "PEMS" / "PEMS07.npz",
        "PEMS08": root / "PEMS" / "PEMS08.npz",
        "Solar": root / "solar" / "solar_AL.txt",
    }

    ok = True
    print(f"Dataset root: {root}")
    for name, path in expected.items():
        if not path.exists():
            ok = False
            print(f"[MISS] {name}: {path}")
            continue

        problems: list[str] = []
        if path.suffix.lower() == ".csv":
            problems = _check_csv(path)
        elif path.suffix.lower() == ".npz":
            problems = _check_npz(path)
        elif path.suffix.lower() == ".txt":
            problems = _check_text_lines(path)

        if problems:
            ok = False
            print(f"[BAD ] {name}: {path}")
            for p in problems:
                print(f"       - {p}")
        else:
            print(f"[ OK ] {name}: {path}")

    if not ok:
        print("\n至少有一个数据文件缺失或格式不符合要求。你可以只下载/解压你要跑的那几个数据集。")
        return 1

    print("\n全部检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


