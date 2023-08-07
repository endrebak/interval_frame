import interval_frame  # noqa: F401

from hypothesis import given, settings

from tests.property.generate_intervals import interval_df
from tests.property.helpers import compare_frames, to_pyranges
from tests.property.hypothesis_settings import MAX_EXAMPLES, PRINT_BLOB


@settings(max_examples=MAX_EXAMPLES, print_blob=PRINT_BLOB, deadline=None)
@given(df=interval_df(), df2=interval_df())
def test_join(df, df2):
    print(df)
    print(df2)
    res_pyranges = to_pyranges(df).join(to_pyranges(df2), suffix="_right", apply_strand_suffix=False).df
    print("PYRANGES", res_pyranges)
    res_interval_frame = df.interval.join(df2, on=("Start", "End"), by=["Chromosome"]).collect().to_pandas()
    print("PORANGES", res_interval_frame)
    compare_frames(pd_df=res_pyranges, pl_df=res_interval_frame)


@settings(max_examples=MAX_EXAMPLES, print_blob=PRINT_BLOB, deadline=None)
@given(df=interval_df(), df2=interval_df())
def test_missing_left(df, df2):
    print(df)
    print(df2)
    res_pyranges = to_pyranges(df).overlap(to_pyranges(df2), invert=True).df
    print("PYRANGES", res_pyranges)
    res_interval_frame = (
        df.interval.nonoverlapping(df2, on=("Start", "End"), by=["Chromosome"], how="left").collect().to_pandas()
    )
    print("PORANGES", res_interval_frame)
    compare_frames(pd_df=res_pyranges, pl_df=res_interval_frame, comparison_cols=("Start", "End", "Chromosome"))


@settings(max_examples=MAX_EXAMPLES, print_blob=PRINT_BLOB, deadline=None)
@given(df=interval_df(), df2=interval_df())
def test_missing_right(df, df2):
    print(df)
    print(df2)
    res_pyranges = to_pyranges(df2).overlap(to_pyranges(df), invert=True).df
    print("PYRANGES", res_pyranges)
    res_interval_frame = (
        df.interval.nonoverlapping(df2, on=("Start", "End"), by=["Chromosome"], how="right").collect().to_pandas()
    )
    print("PORANGES", res_interval_frame)
    compare_frames(pd_df=res_pyranges, pl_df=res_interval_frame, comparison_cols=("Start", "End", "Chromosome"))
