from loguru import logger
from sklearn.metrics import mean_absolute_error
from tabpfn import TabPFNRegressor

from rdblearn.constants import TABPFN_DEFAULT_CONFIG
from rdblearn.datasets import RDBDataset
from rdblearn.estimator import RDBLearnRegressor

logger.enable("rdblearn")

DEFAULT_REGRESSOR_MODEL_PATH = "tabpfn-v2-regressor-v2_default.ckpt"


def main(
    dataset_name: str = "rel-avito",
    task_name: str = "ad-ctr",
    model_path: str = DEFAULT_REGRESSOR_MODEL_PATH,
    random_state: int | None = None,
):
    # 1. Load Dataset
    print(f"Loading '{dataset_name}' dataset...")
    dataset = RDBDataset.from_relbench(dataset_name)

    # Select task
    if task_name not in dataset.tasks:
        raise ValueError(
            f"Task '{task_name}' not found in dataset. Available tasks: {list(dataset.tasks.keys())}"
        )

    task = dataset.tasks[task_name]
    print(f"Loaded task: {task.name}")
    print(f"Train shape: {task.train_df.shape}")
    print(f"Test shape: {task.test_df.shape}")

    # 2. Initialize Model
    print("Initializing TabPFNRegressor...")
    config = dict(TABPFN_DEFAULT_CONFIG)
    config["model_path"] = model_path
    if random_state is not None:
        # NOTE: the results are not fully reproducible as fastdfs is not deterministic
        config["random_state"] = random_state
    base_model = TabPFNRegressor(**config)

    # Configure RDBLearn
    reg = RDBLearnRegressor(
        base_estimator=base_model,
        random_state=random_state,
    )

    # 3. Train
    print("Training model...")
    # Separate features and target
    X_train = task.train_df.drop(columns=[task.metadata.target_col])
    y_train = task.train_df[task.metadata.target_col]

    reg.fit(
        X=X_train,
        y=y_train,
        rdb=dataset.rdb,
        key_mappings=task.metadata.key_mappings,
        cutoff_time_column=task.metadata.time_col,
    )
    print("Training complete.")

    # 4. Predict
    print("Predicting on test set...")
    X_test = task.test_df.drop(columns=[task.metadata.target_col])
    y_test = task.test_df[task.metadata.target_col].astype(float)

    # Predict with output_type="median"
    y_pred = reg.predict(X=X_test, output_type="median")

    # 5. Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Test MAE: {mae:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RDBLearn RelBench regression example.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="rel-avito",
        help="RelBench dataset name (e.g. rel-avito, rel-f1, rel-event).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="ad-ctr",
        help="RelBench task name.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_REGRESSOR_MODEL_PATH,
        help=("Checkpoint path for TabPFNRegressor."),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility.",
    )
    args = parser.parse_args()

    main(
        dataset_name=args.dataset,
        task_name=args.task,
        model_path=args.model_path,
        random_state=args.random_state,
    )
