from typing import Literal

import polars as pl


def add_length(
    starts: str,
    ends: str,
    alias: str,
) -> pl.Expr:
    return (
        pl.col(ends)
        .explode()
        .sub(
            pl.col(starts).explode(),
        )
        .alias(alias)
        .implode()
    )


def search(
    col1: str,
    col2: str,
    side: Literal["any", "left", "right"] = "left",
) -> pl.Expr:
    return (
        pl.col(col1)
        .explode()
        .search_sorted(
            pl.col(col2).explode(),
            side=side,
        )
    )
