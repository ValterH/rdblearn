import numpy as np
import pandas as pd

from fastdfs.api import create_rdb

from rdblearn.datasets import RDBDataset, Task, TaskMetadata


def make_synthetic_conjunction_dataset(
    n_entities: int = 2000,
    n_timesteps: int = 6,
    window_size: int = 4,
    train_fraction: float = 0.8,
    random_state: int = 42,
) -> RDBDataset:
    """
    Construct a synthetic relational dataset where:

      - primary table
        * one row per entity in table `entity` (column `entity_id`)

      - child table (`transactions`)
        * there are `n_timesteps` discrete timestamps (e.g. hours)
        * for every `(entity_id, timestamp)` pair there is exactly one row
        * the **label window** consists of the last `window_size` timestamps;
          only transactions in this window are used to compute the label
        * within this window, every entity has the same fixed number of ones
          for `feat_A` and for `feat_B` (same `k_A`, `k_B` across entities);
        * outside the window, `feat_A` and `feat_B` are IID Bernoulli(0.5)
          and do not affect the label

      - task table
        * one row per entity with columns:
              - `entity_id`
              - `cutoff_time` (the last timestamp)
              - `label`
        * label definition:
              y = 1  iff  there exists at least one transaction for that
                          entity in the window where
                              feat_A == 1 and feat_B == 1
              y = 0  otherwise
        * the label thus depends on whether the ones for `feat_A` and `feat_B`
          align on the **same** row within the window (row-level conjunction),
          not on the per-entity marginals of each feature separately

    Because the per-entity marginals of `feat_A` and `feat_B` in the label
    window are fixed and matched, standard per-entity aggregations
    (counts / means / max / min / std of each feature) are essentially
    uninformative about the label. A model that only sees flattened,
    per-feature aggregates should perform near chance, whereas a model that
    can reason about row-level interactions across time could, in principle,
    recover the conjunction signal.
    """
    rng = np.random.default_rng(random_state)

    # Primary table: one row per entity
    entity_ids = np.arange(n_entities, dtype=int)
    primary_df = pd.DataFrame({"entity_id": entity_ids})

    # Discrete timesteps shared by all entities: ensures at least
    # `n_entities` entities per timestamp.
    if n_timesteps < window_size + 1:
        raise ValueError("n_timesteps must be >= window_size + 1")

    base_time = pd.Timestamp("2024-01-01")
    time_indices = np.arange(n_timesteps)
    timestamps = base_time + pd.to_timedelta(time_indices, unit="h")

    tx_rows: list[dict] = []

    # Define window timesteps (last `window_size` steps)
    window_ts = timestamps[-window_size:]
    non_window_ts = timestamps[:-window_size]

    # For all entities, fix the number of ones per feature inside the window.
    # This keeps transaction marginals matched; labels are computed via conjunction.
    k_A = max(1, window_size // 2)
    k_B = max(1, window_size // 2)

    # Base patterns for A and B within the window (same for all entities),
    # which we will randomly shuffle per entity.
    a_base = np.zeros(window_size, dtype=int)
    b_base = np.zeros(window_size, dtype=int)
    a_base[:k_A] = 1
    b_base[:k_B] = 1

    for eid in entity_ids:
        # Non-window part: arbitrary IID Bernoulli(0.5) for both features
        for ts in non_window_ts:
            a = int(rng.integers(0, 2))
            b = int(rng.integers(0, 2))
            tx_rows.append(
                {
                    "entity_id": eid,
                    "timestamp": ts,
                    "feat_A": a,
                    "feat_B": b,
                }
            )

        # Window part: take the same marginal pattern and shuffle it per entity.
        # This makes the per-entity marginals identical across entities and
        # labels, while the label will be determined later by whether any
        # timestep happens to have A == 1 and B == 1 simultaneously.
        a_pattern = rng.permutation(a_base)
        b_pattern = rng.permutation(b_base)

        for local_t, ts in enumerate(window_ts):
            a = int(a_pattern[local_t])
            b = int(b_pattern[local_t])
            tx_rows.append(
                {
                    "entity_id": eid,
                    "timestamp": ts,
                    "feat_A": a,
                    "feat_B": b,
                }
            )

    transactions_df = pd.DataFrame(tx_rows)

    # Single global cutoff time for all entities (last timestamp)
    cutoff_time = timestamps[-1]

    # Compute labels from the generated features:
    # y = 1 iff there exists a row in the window with feat_A == 1 and feat_B == 1.
    labels: list[int] = []
    window_start = window_ts[0]
    for eid in entity_ids:
        tx = transactions_df[transactions_df["entity_id"] == eid]
        window_mask = (tx["timestamp"] >= window_start) & (
            tx["timestamp"] <= cutoff_time
        )
        tx_w = tx[window_mask]
        has_both = ((tx_w["feat_A"] == 1) & (tx_w["feat_B"] == 1)).any()
        labels.append(int(has_both))

    task_df = pd.DataFrame(
        {
            "entity_id": entity_ids,
            "cutoff_time": cutoff_time,
            "label": labels,
        }
    )

    # Simple random train / test split
    rng_idx = np.random.default_rng(random_state)
    idx = np.arange(len(task_df))
    rng_idx.shuffle(idx)
    split = int(train_fraction * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]

    train_df = task_df.iloc[train_idx].reset_index(drop=True)
    test_df = task_df.iloc[test_idx].reset_index(drop=True)

    # Build RDB with two tables:
    #   - entity: primary key entity_id
    #   - transactions: foreign key entity_id -> entity.entity_id, time column timestamp
    rdb = create_rdb(
        name="synthetic_conjunction",
        tables={
            "entity": primary_df,
            "transactions": transactions_df,
        },
        primary_keys={
            "entity": "entity_id",
        },
        foreign_keys=[
            ("transactions", "entity_id", "entity", "entity_id"),
        ],
        time_columns={
            "transactions": "timestamp",
        },
    )

    metadata = TaskMetadata(
        key_mappings={"entity_id": "entity.entity_id"},
        target_col="label",
        time_col="cutoff_time",
        task_type="classification",
        evaluation_metric="roc_auc_score",
    )

    task = Task(
        name="entity-conjunction",
        train_df=train_df,
        test_df=test_df,
        val_df=None,
        metadata=metadata,
    )

    return RDBDataset(rdb=rdb, tasks=[task])

