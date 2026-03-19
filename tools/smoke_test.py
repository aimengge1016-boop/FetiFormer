import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root / "dataset" / "_smoke"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    csv_path = dataset_dir / "smoke.csv"
    if not csv_path.exists():
        n = 600
        rng = np.random.default_rng(2023)
        dates = pd.date_range("2020-01-01", periods=n, freq="H")
        a = rng.normal(size=n)
        b = rng.normal(size=n)
        c = rng.normal(size=n)
        ot = 0.5 * a - 0.2 * b + 0.1 * c + rng.normal(scale=0.05, size=n)
        df = pd.DataFrame(
            {
                "date": dates.astype(str),
                "A": a,
                "B": b,
                "C": c,
                "OT": ot,
            }
        )
        df.to_csv(csv_path, index=False)

    cmd = [
        sys.executable,
        "-u",
        "run.py",
        "--is_training",
        "1",
        "--root_path",
        "./dataset/_smoke/",
        "--data_path",
        "smoke.csv",
        "--model_id",
        "smoke_96_24",
        "--model",
        "iTransformer",
        "--data",
        "custom",
        "--features",
        "M",
        "--seq_len",
        "96",
        "--pred_len",
        "24",
        "--label_len",
        "48",
        "--e_layers",
        "1",
        "--d_layers",
        "1",
        "--enc_in",
        "4",
        "--dec_in",
        "4",
        "--c_out",
        "4",
        "--d_model",
        "64",
        "--n_heads",
        "4",
        "--d_ff",
        "64",
        "--batch_size",
        "32",
        "--learning_rate",
        "0.001",
        "--train_epochs",
        "1",
        "--patience",
        "1",
        "--num_workers",
        "0",
        "--itr",
        "1",
        "--des",
        "Smoke",
    ]

    env = os.environ.copy()
    # Optional: user can override; we won't force CPU here because run.py uses torch.cuda.is_available()
    env.setdefault("CUDA_VISIBLE_DEVICES", env.get("CUDA_VISIBLE_DEVICES", "0"))

    print(f"Project root: {project_root}")
    print(f"Smoke CSV: {csv_path}")
    print("Running:\n  " + " ".join(cmd))
    print("\nIf this fails, it is almost always a Python version / torch install issue on Windows.\n")

    p = subprocess.run(cmd, cwd=str(project_root), env=env)
    if p.returncode != 0:
        return p.returncode

    print("\nSmoke test finished.")
    print("Outputs:")
    print(f"- checkpoints/: {project_root / 'checkpoints'}")
    print(f"- results/: {project_root / 'results'}")
    print(f"- result_long_term_forecast.txt: {project_root / 'result_long_term_forecast.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


