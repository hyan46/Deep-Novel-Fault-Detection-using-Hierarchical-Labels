import pytest
from src.datasets.rolling import get_all_rolling_df, split


@pytest.fixture
def rolling_df():
    return get_all_rolling_df()


def test_split_dfs_add_up(rolling_df):
    in_dist_df, trn_df, val_df, tst_df, out_dist_df = split(
        all_rolling_df=rolling_df,
        split_ratios=[0.6, 0.2, 0.2],
        split_seed=0,
        left_out="A31",
    )
    assert len(in_dist_df) == len(trn_df) + len(val_df) + len(tst_df)
    assert len(rolling_df) == len(in_dist_df) + len(out_dist_df)


def test_split_dfs_non_overlapping(rolling_df):
    in_dist_df, trn_df, val_df, tst_df, out_dist_df = split(
        all_rolling_df=rolling_df,
        split_ratios=[0.6, 0.2, 0.2],
        split_seed=0,
        left_out="A31",
    )
    assert len(set(trn_df.index).intersection(set(val_df.index))) == 0
    assert len(set(tst_df.index).intersection(set(val_df.index))) == 0
    assert len(set(out_dist_df.index).intersection(set(val_df.index))) == 0
    assert len(set(trn_df.index).intersection(set(out_dist_df.index))) == 0
    assert len(set(tst_df.index).intersection(set(trn_df.index))) == 0
    assert set(trn_df.index).union(set(val_df.index)).union(
        set(tst_df.index)
    ).union(out_dist_df.index) == set(in_dist_df.index).union(
        set(out_dist_df.index)
    )


def test_split_reproducible():
    rolling_df = get_all_rolling_df()
    _, trn_df, val_df, tst_df, out_dist_df = split(
        all_rolling_df=rolling_df,
        split_ratios=[0.6, 0.2, 0.2],
        split_seed=0,
        left_out="A31",
    )
    rolling_df = get_all_rolling_df()
    _, trn_df_2, val_df_2, tst_df_2, out_dist_df_2 = split(
        all_rolling_df=rolling_df,
        split_ratios=[0.6, 0.2, 0.2],
        split_seed=0,
        left_out="A31",
    )

    assert set(trn_df.index) == set(trn_df_2.index)
    assert set(val_df.index) == set(val_df_2.index)
    assert set(tst_df.index) == set(tst_df_2.index)
    assert set(out_dist_df.index) == set(out_dist_df_2.index)
