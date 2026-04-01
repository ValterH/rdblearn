from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

from rdblearn.constants import TABPFN_DEFAULT_CONFIG
from rdblearn.estimator import RDBLearnClassifier
from rdblearn.synthetic_conjunction_dataset import (
    make_synthetic_conjunction_dataset,
)

DEFAULT_CLASSIFIER_MODEL_PATH = "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"


def main(
    model_path: str = DEFAULT_CLASSIFIER_MODEL_PATH,
    random_state: int | None = 42,
) -> None:
    # 1. Load synthetic adversarial-style dataset
    print("Loading synthetic conjunction dataset...")
    dataset = make_synthetic_conjunction_dataset(
        n_timesteps=20,
        window_size=4,
        n_entities=10_000,
        random_state=random_state,
    )
    task = dataset.tasks["entity-conjunction"]
    print(f"Loaded task: {task.name}")
    print(f"Train shape: {task.train_df.shape}")
    print(f"Test shape: {task.test_df.shape}")

    # 2. Initialize Model
    print("Initializing TabPFNClassifier...")
    # Note: You might need to adjust 'device' in TABPFN_DEFAULT_CONFIG if you don't have a GPU
    config = dict(TABPFN_DEFAULT_CONFIG)
    config["model_path"] = model_path
    if random_state is not None:
        # NOTE: the results are not fully reproducible as fastdfs is not deterministic
        config["random_state"] = random_state
    base_model = TabPFNClassifier(**config)

    # Configure RDBLearn
    # Use the default configuration (automatically loaded if config is None)
    clf = RDBLearnClassifier(
        base_estimator=base_model,
        random_state=random_state,
    )

    # 3. Train
    print("Training model...")
    # Separate features and target
    X_train = task.train_df.drop(columns=[task.metadata.target_col])
    y_train = task.train_df[task.metadata.target_col]

    clf.fit(
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
    y_test = task.test_df[task.metadata.target_col]

    # Predict probabilities for AUC
    y_pred_proba = clf.predict_proba(X=X_test)

    # 5. Evaluate
    auc = roc_auc_score(y_test, y_pred_proba[:, 1]) * 100
    print(f"Test AUC: {auc:.2f}")


if __name__ == "__main__":
    main()
