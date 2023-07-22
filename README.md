# IntervalFrame

IntervalFrame is an extension to the Polars library in Python that adds support for interval operations.

This library makes it easy to join and overlap dataframes based on intervals.

More operations are coming.

## Usage

```python
import polars as pl
import interval_frame  # registers the interval extension

df = pl.DataFrame({
    "chromosome": ["chr1", "chr1", "chr1", "chrX"],
    "start": [10, 20, 30, 10],
    "end": [23, 25, 35, 30],
})

other = pl.DataFrame({
    "chromosome": ["chr1", "chr1", "chr1", "chrX"],
    "start": [12, 22, 32, 29],
    "end": [17, 27, 37, 32],
})

# Join the DataFrames based on overlapping intervals, per chromosome
result = df.interval.join(other, on=("start", "end"), by="chromosome")

print(result.collect())
# shape: (5, 5)
# ┌────────────┬───────┬─────┬─────────────┬───────────┐
# │ chromosome ┆ start ┆ end ┆ start_right ┆ end_right │
# │ ---        ┆ ---   ┆ --- ┆ ---         ┆ ---       │
# │ str        ┆ i64   ┆ i64 ┆ i64         ┆ i64       │
# ╞════════════╪═══════╪═════╪═════════════╪═══════════╡
# │ chrX       ┆ 10    ┆ 30  ┆ 29          ┆ 32        │
# │ chr1       ┆ 10    ┆ 23  ┆ 12          ┆ 17        │
# │ chr1       ┆ 20    ┆ 25  ┆ 22          ┆ 27        │
# │ chr1       ┆ 10    ┆ 23  ┆ 22          ┆ 27        │
# │ chr1       ┆ 30    ┆ 35  ┆ 32          ┆ 37        │
# └────────────┴───────┴─────┴─────────────┴───────────┘
```
