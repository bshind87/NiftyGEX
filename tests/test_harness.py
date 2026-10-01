from datetime import date, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from niftygex_eval.harness import assign_cycles, build_walks, check_split, split

IST = ZoneInfo("Asia/Kolkata")


def weekly_table(weeks: int = 30) -> tuple[pl.DataFrame, list[date]]:
    start = date(2024, 1, 1)  # Monday; expiries on Thursdays
    expiries = [start + timedelta(days=3 + 7 * w) for w in range(weeks)]
    days = [start + timedelta(days=d) for d in range(7 * weeks) if (start + timedelta(days=d)).weekday() < 5]
    frame = pl.DataFrame({"trade_date": days}).with_columns(
        (pl.col("trade_date").cast(pl.Datetime("us")) + pl.duration(hours=15, minutes=15)).dt.replace_time_zone("Asia/Kolkata").alias("snapshot_at"),
    ).with_columns(
        # T2-like target: known at the next day's close.
        (pl.col("snapshot_at") + pl.duration(days=1, minutes=15)).alias("target_available_at"),
    )
    return assign_cycles(frame, expiries), expiries


def test_cycles_follow_next_expiry_on_or_after_each_day() -> None:
    frame, expiries = weekly_table(2)
    by_day = dict(frame.select("trade_date", "cycle").iter_rows())
    assert by_day[date(2024, 1, 4)] == expiries[0]  # expiry day belongs to its own cycle
    assert by_day[date(2024, 1, 5)] == expiries[1]


def test_walks_are_consecutive_and_embargoed() -> None:
    frame, _ = weekly_table()
    walks = build_walks(frame["cycle"].unique().to_list(), test_cycles=4, embargo_cycles=1, min_train_cycles=10)
    assert len(walks) == 5
    for previous, current in zip(walks, walks[1:]):
        assert current.test_cycles[0] > previous.test_cycles[-1]
    for walk in walks:
        assert walk.train_cycles[-1] < walk.embargo_cycles[0] < walk.test_cycles[0]


def test_no_test_row_in_training_and_purge_applied() -> None:
    frame, _ = weekly_table()
    walks = build_walks(frame["cycle"].unique().to_list(), test_cycles=4, embargo_cycles=1, min_train_cycles=10)
    for walk in walks:
        train, test = split(frame, walk)
        check_split(train, test, walk)
        assert train.join(test, on="snapshot_at", how="semi").height == 0
        assert train["target_available_at"].max() < test["snapshot_at"].min()
        assert train["snapshot_at"].is_sorted() and test["snapshot_at"].is_sorted()


def test_purge_drops_training_rows_whose_label_resolves_in_test() -> None:
    frame, _ = weekly_table()
    walk = build_walks(frame["cycle"].unique().to_list(), test_cycles=4, embargo_cycles=1, min_train_cycles=10)[0]
    first_test = frame.filter(pl.col("cycle").is_in(walk.test_cycles))["snapshot_at"].min()
    # A long-horizon label on the last training day that resolves after the test starts must be purged.
    last_train_day = frame.filter(pl.col("cycle").is_in(walk.train_cycles))["snapshot_at"].max()
    frame = frame.with_columns(pl.when(pl.col("snapshot_at") == last_train_day).then(first_test + timedelta(days=1))
                               .otherwise(pl.col("target_available_at")).alias("target_available_at"))
    train, test = split(frame, walk)
    assert last_train_day not in train["snapshot_at"].to_list()
    check_split(train, test, walk)


def test_check_split_detects_leak() -> None:
    frame, _ = weekly_table()
    walk = build_walks(frame["cycle"].unique().to_list(), test_cycles=4, embargo_cycles=1, min_train_cycles=10)[0]
    train, test = split(frame, walk)
    with pytest.raises(AssertionError):
        check_split(pl.concat([train, test.head(1)]), test, walk)


def test_embargo_of_at_least_one_cycle_is_required() -> None:
    with pytest.raises(ValueError):
        build_walks([date(2024, 1, 4)] * 3, test_cycles=1, embargo_cycles=0, min_train_cycles=1)


def test_harness_on_real_t2_table_has_no_leakage() -> None:
    from niftygex_eval.config import load_eval_config
    from niftygex_eval.data import modeling_table

    cfg = load_eval_config()
    paths = cfg["data_config"]["paths"]
    if not (paths["features_root"] / "aggregate" / "aggregate.parquet").exists():
        pytest.skip("features not built")
    table, _, _ = modeling_table(cfg["data_config"], "t2")
    labeled = table.filter(pl.col("t2_label").is_not_null())
    walks = build_walks(labeled["cycle"].unique().to_list(), **cfg["eval"]["harness"])
    assert walks
    tested = set()
    for walk in walks:
        train, test = split(labeled, walk)
        check_split(train, test, walk)
        assert not (set(test["snapshot_at"].to_list()) & tested)  # each row is tested at most once
        tested |= set(test["snapshot_at"].to_list())


def test_inner_walk_is_time_ordered_inside_training_and_embargoed() -> None:
    from niftygex_eval.stage_b import inner_walk

    frame, _ = weekly_table()
    walk = build_walks(frame["cycle"].unique().to_list(), test_cycles=4, embargo_cycles=1, min_train_cycles=12)[0]
    inner = inner_walk(walk, validation_cycles=4, embargo_cycles=1)
    assert set(inner.train_cycles) | set(inner.embargo_cycles) | set(inner.test_cycles) == set(walk.train_cycles)
    assert inner.train_cycles[-1] < inner.embargo_cycles[0] < inner.test_cycles[0]
    assert inner.test_cycles[-1] < walk.embargo_cycles[0]
    train, _ = split(frame, walk)
    inner_train, validation = split(train, inner)
    check_split(inner_train, validation, inner)
    assert inner_walk(walk, validation_cycles=len(walk.train_cycles), embargo_cycles=1) is None
