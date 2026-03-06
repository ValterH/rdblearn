#!/usr/bin/env python3
"""
Run RDBLearn classifier on RelBench classification tasks and store results.

Task mapping (dataset -> tasks):
  event:  ignore, repeat   -> user-ignore, user-repeat
  f1:     dnf, top3       -> driver-dnf, driver-top3
  avito:  clicks, visits  -> user-clicks, user-visits
  hm:     churn           -> user-churn
  stack:  badge, engagmt  -> user-badge, user-engagement
  trial:  out             -> study-outcome
  amazon: user, item      -> user-churn, item-churn
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

from rdblearn.constants import TABPFN_DEFAULT_CONFIG
from rdblearn.datasets import RDBDataset
from rdblearn.estimator import RDBLearnClassifier

logger.enable("rdblearn")

DEFAULT_MODEL_PATH = "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"

# (relbench_dataset_name, relbench_task_name) for classification only
CLASSIFICATION_TASKS = [
    # event
    ("rel-event", "user-ignore"),
    ("rel-event", "user-repeat"),
    # f1
    ("rel-f1", "driver-dnf"),
    ("rel-f1", "driver-top3"),
    # avito
    ("rel-avito", "user-clicks"),
    ("rel-avito", "user-visits"),
    # hm
    ("rel-hm", "user-churn"),
    # stack
    ("rel-stack", "user-badge"),
    ("rel-stack", "user-engagement"),
    # trial
    ("rel-trial", "study-outcome"),
    # amazon
    ("rel-amazon", "user-churn"),
    ("rel-amazon", "item-churn"),
]


def run_one(
    dataset_name: str,
    task_name: str,
    model_path: str = DEFAULT_MODEL_PATH,
    random_state: int | None = 42,
) -> dict:
    """Run classifier on a single (dataset, task). Returns dict with dataset, task, auc, error."""
    dataset = RDBDataset.from_relbench(dataset_name)
    if task_name not in dataset.tasks:
        return {
            "dataset": dataset_name,
            "task": task_name,
            "auc": None,
            "error": f"Task not found. Available: {list(dataset.tasks.keys())}",
        }

    task = dataset.tasks[task_name]
    # Skip non-classification (e.g. regression tasks that slipped in)
    tt = getattr(
        task.metadata.task_type, "value", str(task.metadata.task_type)
    )
    if tt != "classification":
        return {
            "dataset": dataset_name,
            "task": task_name,
            "auc": None,
            "error": f"Not classification (task_type={task.metadata.task_type})",
        }

    config = dict(TABPFN_DEFAULT_CONFIG)
    config["model_path"] = model_path
    if random_state is not None:
        config["random_state"] = random_state
    base_model = TabPFNClassifier(**config)
    clf = RDBLearnClassifier(
        base_estimator=base_model, random_state=random_state
    )

    X_train = task.train_df.drop(columns=[task.metadata.target_col])
    y_train = task.train_df[task.metadata.target_col]
    clf.fit(
        X=X_train,
        y=y_train,
        rdb=dataset.rdb,
        key_mappings=task.metadata.key_mappings,
        cutoff_time_column=task.metadata.time_col,
    )

    X_test = task.test_df.drop(columns=[task.metadata.target_col])
    y_test = task.test_df[task.metadata.target_col]
    y_pred_proba = clf.predict_proba(X=X_test)

    n_classes = y_pred_proba.shape[1]
    if n_classes == 2:
        auc = roc_auc_score(y_test, y_pred_proba[:, 1]) * 100
    else:
        auc = (
            roc_auc_score(
                y_test,
                y_pred_proba,
                multi_class="ovr",
                average="macro",
            )
            * 100
        )

    return {
        "dataset": dataset_name,
        "task": task_name,
        "auc": round(auc, 4),
        "error": None,
    }


def main(
    output_path: str = "classification_results.json",
    model_path: str = DEFAULT_MODEL_PATH,
    random_state: int | None = 42,
    task_filter: list[str] | None = None,
) -> None:
    """
    Run all classification tasks and write results to output_path (JSON).
    task_filter: optional; only run (dataset, task) where name contains one.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for dataset_name, task_name in CLASSIFICATION_TASKS:
        if task_filter:
            if not any(
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
                "auc": None,
                "error": str(e),
            }
        results.append(row)
        if row.get("auc") is not None:
            logger.info("  -> AUC: %.2f", row["auc"])
        else:
            logger.warning("  -> Skip/Error: %s", row.get("error", ""))

    # Write JSON (list of dicts)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote {len(results)} results to {path}")

    # Also write a compact CSV-like summary
    csv_path = path.with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write("dataset,task,auc,error\n")
        for r in results:
            auc = r.get("auc") if r.get("auc") is not None else ""
            err = (r.get("error") or "").replace(",", ";")
            f.write(f"{r['dataset']},{r['task']},{auc},{err}\n")
    logger.info(f"Wrote summary to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RDBLearn on RelBench classification tasks."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="classification_results.json",
        help="Output path for results (JSON). A .csv sibling is also written.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="TabPFN classifier checkpoint path.",
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
