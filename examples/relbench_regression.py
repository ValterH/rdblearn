#!/usr/bin/env python3
"""
Run RDBLearn regressor on RelBench regression tasks and store results.

This script mirrors `examples/relbench_classification.py` but evaluates
regression tasks using Mean Absolute Error (MAE).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger
from sklearn.metrics import mean_absolute_error
from tabpfn import TabPFNRegressor

from rdblearn.constants import TABPFN_DEFAULT_CONFIG
from rdblearn.datasets import RDBDataset
from rdblearn.estimator import RDBLearnRegressor

logger.enable("rdblearn")

DEFAULT_MODEL_PATH = "tabpfn-v2-regressor-v2_default.ckpt"

# (relbench_dataset_name, relbench_task_name) for forecasting regression only
# (exclude autocomplete/link prediction tasks)
REGRESSION_TASKS = [
    # event
    ("rel-event", "user-attendance"),
    # f1
    ("rel-f1", "driver-position"),
    # avito
    ("rel-avito", "ad-ctr"),
    # trial
    ("rel-trial", "site-success"),
    ("rel-trial", "study-adverse"),
    # hm
    ("rel-hm", "item-sales"),
    # stack
    ("rel-stack", "post-votes"),
    # amazon
    ("rel-amazon", "user-ltv"),
    ("rel-amazon", "item-ltv"),
    # ratebeer
    ("rel-ratebeer", "user-count"),
    # arxiv
    ("rel-arxiv", "author-publication"),
]


def run_one(
    dataset_name: str,
    task_name: str,
    model_path: str = DEFAULT_MODEL_PATH,
    random_state: int | None = 42,
) -> dict:
    """Run regressor on a single (dataset, task). Returns dataset, task, mae, error."""
    dataset = RDBDataset.from_relbench(dataset_name)
    if task_name not in dataset.tasks:
        return {
            "dataset": dataset_name,
            "task": task_name,
            "mae": None,
            "error": f"Task not found. Available: {list(dataset.tasks.keys())}",
        }

    task = dataset.tasks[task_name]
    task_type = getattr(
        task.metadata.task_type, "value", str(task.metadata.task_type)
    )
    if str(task_type).lower() != "regression":
        return {
            "dataset": dataset_name,
            "task": task_name,
            "mae": None,
            "error": f"Not regression (task_type={task.metadata.task_type})",
        }

    config = dict(TABPFN_DEFAULT_CONFIG)
    config["model_path"] = model_path
    if random_state is not None:
        # NOTE: the results are not fully reproducible as fastdfs is not deterministic.
        config["random_state"] = random_state

    base_model = TabPFNRegressor(**config)
    reg = RDBLearnRegressor(base_estimator=base_model, random_state=random_state)

    X_train = task.train_df.drop(columns=[task.metadata.target_col])
    y_train = task.train_df[task.metadata.target_col]
    reg.fit(
        X=X_train,
        y=y_train,
        rdb=dataset.rdb,
        key_mappings=task.metadata.key_mappings,
        cutoff_time_column=task.metadata.time_col,
    )

    X_test = task.test_df.drop(columns=[task.metadata.target_col])
    y_test = task.test_df[task.metadata.target_col].astype(float)
    y_pred = reg.predict(X=X_test, output_type="median")
    mae = mean_absolute_error(y_test, y_pred)

    return {
        "dataset": dataset_name,
        "task": task_name,
        "mae": round(float(mae), 6),
        "error": None,
    }

def main(
    output_path: str = "regression_results.json",
    model_path: str = DEFAULT_MODEL_PATH,
    random_state: int | None = 42,
    task_filter: list[str] | None = None,
) -> None:
    """
    Run all predefined regression tasks and write results to output_path (JSON).
    task_filter: optional; only run (dataset, task) where name contains one.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for dataset_name, task_name in REGRESSION_TASKS:
        if task_filter and not any(
            f in dataset_name or f in task_name for f in task_filter
        ):
            continue
        logger.info(f"Running {dataset_name} / {task_name} ...")
        try:
            row = run_one(
                dataset_name=dataset_name,
                task_name=task_name,
                model_path=model_path,
                random_state=random_state,
            )
        except Exception as e:
            row = {
                "dataset": dataset_name,
                "task": task_name,
                "mae": None,
                "error": str(e),
            }

        results.append(row)
        if row.get("mae") is not None:
            logger.info(f"  -> MAE: {row['mae']:.6f}")
        else:
            logger.warning(f"  -> Skip/Error: {row.get('error', '')}")

    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote {len(results)} results to {path}")

    csv_path = path.with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write("dataset,task,mae,error\n")
        for r in results:
            mae = r.get("mae") if r.get("mae") is not None else ""
            err = (r.get("error") or "").replace(",", ";")
            f.write(f"{r['dataset']},{r['task']},{mae},{err}\n")
    logger.info(f"Wrote summary to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RDBLearn on RelBench regression tasks."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="regression_results.json",
        help="Output path for results (JSON). A .csv sibling is also written.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="TabPFN regressor checkpoint path.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        nargs="*",
        default=None,
        metavar="SUBSTR",
        help="Only run tasks whose dataset/task name contains one of these.",
    )
    args = parser.parse_args()
    main(
        output_path=args.output,
        model_path=args.model_path,
        random_state=args.random_state,
        task_filter=args.filter,
    )
