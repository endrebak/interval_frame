from typing import TYPE_CHECKING, List, Literal

import polars as pl

from interval_frame.constants import (
    DUPLICATED_INDICES_PROPERTY,
    ENDS_1IN2_PROPERTY,
    ENDS_2IN1_PROPERTY,
    LENGTHS_1IN2_PROPERTY,
    LENGTHS_2IN1_PROPERTY,
    MASK_1IN2_PROPERTY,
    MASK_2IN1_PROPERTY,
    STARTS_1IN2_PROPERTY,
    STARTS_2IN1_PROPERTY,
)
from interval_frame.helper_ops import add_length, search

if TYPE_CHECKING:
    from interval_frame.groupby_join_result import GroupByJoinResult

"""Object that computes overlaps between two dataframes with intervals."""


class OverlappingIntervals:
    """The data needed to compute paired overlaps between two sets of ranges.
    This class is used to compute things like joins and overlaps between ranges.
    """

    def __init__(
        self,
        joined_result: "GroupByJoinResult",
        *,
        closed_intervals: bool = False,
    ) -> None:
        """Initialize the OverlappingIntervals object."""
        self.joined_result = joined_result
        self.closed_intervals = closed_intervals

        group_keys = joined_result.by

        self.data = (
            joined_result.joined.groupby(group_keys)
            .agg([pl.all().explode(), *self._find_starts_in_ends()])
            .groupby(group_keys)
            .agg([pl.all().explode(), *self._compute_masks()])
            .groupby(group_keys)
            .agg(
                [
                    pl.exclude(
                        STARTS_1IN2_PROPERTY,
                        ENDS_1IN2_PROPERTY,
                        STARTS_2IN1_PROPERTY,
                        ENDS_2IN1_PROPERTY,
                    ).explode(),
                    *self._apply_masks(),
                ],
            )
            .groupby(group_keys)
            .agg([pl.all(), *self._add_lengths()])
            .explode(pl.exclude(group_keys))
        )

    def _find_starts_in_ends(self) -> List[pl.Expr]:
        side: Literal["right", "left"] = "right" if self.closed_intervals else "left"

        return [
            search(self.joined_result.secondary_start_renamed, self.joined_result.main_start, side="left").alias(
                STARTS_2IN1_PROPERTY,
            ),
            search(self.joined_result.secondary_start_renamed, self.joined_result.main_end, side=side).alias(
                ENDS_2IN1_PROPERTY,
            ),
            search(self.joined_result.main_start, self.joined_result.secondary_start_renamed, side="right").alias(
                STARTS_1IN2_PROPERTY,
            ),
            search(self.joined_result.main_start, self.joined_result.secondary_end_renamed, side=side).alias(
                ENDS_1IN2_PROPERTY,
            ),
        ]

    @staticmethod
    def _compute_masks() -> List[pl.Expr]:
        return [
            pl.col(ENDS_2IN1_PROPERTY).explode().gt(pl.col(STARTS_2IN1_PROPERTY).explode()).alias(MASK_2IN1_PROPERTY),
            pl.col(ENDS_1IN2_PROPERTY).explode().gt(pl.col(STARTS_1IN2_PROPERTY).explode()).alias(MASK_1IN2_PROPERTY),
        ]

    @staticmethod
    def _apply_masks() -> List[pl.Expr]:
        return [
            pl.col([STARTS_2IN1_PROPERTY, ENDS_2IN1_PROPERTY]).explode().filter(pl.col(MASK_2IN1_PROPERTY).explode()),
            pl.col([STARTS_1IN2_PROPERTY, ENDS_1IN2_PROPERTY]).explode().filter(pl.col(MASK_1IN2_PROPERTY).explode()),
        ]

    @staticmethod
    def _add_lengths() -> List[pl.Expr]:
        return [
            add_length(STARTS_2IN1_PROPERTY, ENDS_2IN1_PROPERTY, LENGTHS_2IN1_PROPERTY),
            add_length(STARTS_1IN2_PROPERTY, ENDS_1IN2_PROPERTY, LENGTHS_1IN2_PROPERTY),
        ]

    # ... Rest of the methods with similar improvements ...
    @staticmethod
    def _repeat_frame(
        columns: List[str],
        startsin: str,
        endsin: str,
    ) -> pl.Expr:
        return pl.col(columns).explode().repeat_by(pl.col(endsin).explode() - pl.col(startsin).explode()).explode()

    @staticmethod
    def _mask_and_repeat_frame(
        *,
        columns: List[str],
        mask: str,
        startsin: str,
        endsin: str,
    ) -> pl.Expr:
        """Apply a mask to the values in specified columns, and then repeats the masked values."""
        return (
            pl.col(columns)
            .explode()
            .filter(pl.col(mask).explode())
            .repeat_by(
                pl.when(pl.col(mask).list.any())
                .then(pl.col(endsin).explode().drop_nulls() - pl.col(startsin).explode().drop_nulls())
                .otherwise(pl.lit(0)),
            )
            .explode()
        )

    @staticmethod
    def _repeat_other(
        *,
        columns: List[str],
        starts: pl.Expr,
        diffs: pl.Expr,
    ) -> pl.Expr:
        """Repeat the values in specified columns based on specified start and end values."""
        return (
            pl.col(columns)
            .explode()
            .take(
                pl.int_ranges(
                    start=starts,
                    end=starts.add(diffs),
                )
                .explode()
                .drop_nulls(),
            )
        )

    def overlapping_pairs(
        self,
        how: Literal["inner", "left", "right", "outer"] = "inner",
    ) -> "pl.LazyFrame":
        """Join two dataframes based on overlapping intervals.

        Returns
        -------
        - A data frame representing the overlapping pairs.
        """
        if self.joined_result.is_empty():
            return self.joined_result.joined

        self.joined_result.get_colnames_secondary_without_groupby()

        self.missing_overlaps_left()
        top_left = self._calculate_top_left()
        bottom_left = self._calculate_bottom_left()
        top_right = self._calculate_top_right()
        bottom_right = self._calculate_bottom_right()
        top_left.with_context(top_right).select(pl.all())
        bottom_left.with_context(bottom_right).select(pl.all())

        # Concatenate the parts to get the final result
        overlapping = (
            pl.concat(
                [top_left, bottom_left],
            )
            .with_context(
                pl.concat(
                    [top_right, bottom_right],
                ),
            )
            .select(
                pl.all(),
            )
        )

        if how == "inner":
            result = overlapping
        else:
            missing = self.missing_overlaps(
                how=how,
                include_missing_in_other=True,
            )

            with_nulls = [overlapping, missing]
            result = pl.concat(
                with_nulls,
                how="vertical_relaxed",
            )

        return result

    def _calculate_other_part(
            self,
            *,
            column_names: List[str],
            start_property: str,
            length_property: str,
            group_keys: List[str],
    ) -> pl.LazyFrame:
        return (
            self.data.groupby(group_keys)
            .agg(
                self._repeat_other(
                    columns=column_names,
                    starts=pl.col(start_property).explode(),
                    diffs=pl.col(length_property).explode(),
                ),
            )
            .explode(column_names)
            .drop_nulls()
        ).sort(group_keys)

    def _calculate_bottom_left(self) -> pl.LazyFrame:
        return self._calculate_other_part(
            column_names=self.joined_result.get_colnames_without_groupby(),
            start_property=STARTS_1IN2_PROPERTY,
            length_property=LENGTHS_1IN2_PROPERTY,
            group_keys=self.joined_result.by,
        )

    def _calculate_top_right(self) -> pl.LazyFrame:
        return self._calculate_other_part(
            column_names=self.joined_result.get_joined_colnames_secondary(),
            start_property=STARTS_2IN1_PROPERTY,
            length_property=LENGTHS_2IN1_PROPERTY,
            group_keys=self.joined_result.by,
        )

    def _calculate_self_part(
            self,
            *,
            column_names: List[str],
            mask_property: Literal["mask_1in2", "mask_2in1"],
            start_property: str,
            end_property: str,
    ) -> pl.LazyFrame:
        group_keys = self.joined_result.by
        return (
            self.data.filter(pl.col(mask_property).list.any())
            .groupby(group_keys)
            .agg(
                self._mask_and_repeat_frame(
                    columns=column_names,
                    mask=mask_property,
                    startsin=start_property,
                    endsin=end_property,
                ),
            )
            .explode(column_names)
            .drop_nulls()
        ).sort(group_keys)

    def _calculate_bottom_right(self) -> pl.LazyFrame:
        return self._calculate_self_part(
            column_names=self.joined_result.get_joined_colnames_secondary(),
            mask_property=MASK_1IN2_PROPERTY,
            start_property=STARTS_1IN2_PROPERTY,
            end_property=ENDS_1IN2_PROPERTY,
        )

    def _calculate_top_left(self) -> pl.LazyFrame:
        mask_columns = [
            c
            for c in self.joined_result.get_colnames_without_groupby()
            if c not in [MASK_2IN1_PROPERTY, STARTS_2IN1_PROPERTY, ENDS_2IN1_PROPERTY]
        ]
        return self._calculate_self_part(
            column_names=mask_columns,
            mask_property=MASK_2IN1_PROPERTY,
            start_property=STARTS_2IN1_PROPERTY,
            end_property=ENDS_2IN1_PROPERTY,
        )

    def overlaps(self) -> pl.LazyFrame:
        """Calculate the overlapping intervals between the two data frames.

        Returns
        -------
        - A data frame representing the overlapping intervals.
        """
        grouping_cols = self.joined_result.by
        cols_excluding_group_keys = self.joined_result.get_colnames_without_groupby()

        top_left = (
            self.data.groupby(grouping_cols)
            .agg(
                # Filter the columns by the mask
                pl.col(cols_excluding_group_keys)
                .explode()
                .filter(pl.col(MASK_2IN1_PROPERTY).explode()),
            )
            .explode(cols_excluding_group_keys)
            .drop_nulls()
        ).sort(grouping_cols)

        bottom_left = (
            self.data.groupby(grouping_cols)
            .agg(
                self._repeat_other(
                    columns=cols_excluding_group_keys,
                    starts=pl.col(STARTS_1IN2_PROPERTY).explode(),
                    diffs=pl.col(LENGTHS_1IN2_PROPERTY).explode(),
                ),
            )
            .explode(cols_excluding_group_keys)
            .drop_nulls()
        ).sort(grouping_cols)

        return pl.concat([top_left, bottom_left])

    def missing_overlaps(
        self,
        how: Literal["left", "right", "outer"],
        *,
        include_missing_in_other: bool = False,
    ) -> pl.LazyFrame:
        """Return the mising overlaps on the left and right sides of the join."""
        if self.joined_result.is_empty():
            if how == "left":
                missing = self.joined_result.main_frame
            elif how == "right":
                missing = self.joined_result.secondary_frame
            else:
                msg = "how not implemented for empty dataframes"
                raise NotImplementedError(msg)

            return missing

        if how not in ["left", "right", "outer"]:
            msg = f"how must be one of 'left', 'right', 'outer', but was {how}"
            raise ValueError(msg)

        missing = []
        if how in ["left", "outer"]:
            missing_top_left = self._calculate_missing_top_left()
            missing_bottom_left = self._calculate_missing_bottom_left()
            missing_left_within_groups = self._calculate_missing_overlaps(
                missing_top=missing_top_left,
                missing_bottom=missing_bottom_left,
            )

            if self.joined_result.groupby_args_given:
                missing_left = pl.concat(
                    [
                        missing_left_within_groups,
                        self.joined_result.groups_unique_to_left(),
                    ],
                )
            else:
                missing_left = missing_left_within_groups

            if include_missing_in_other:
                missing_left = missing_left.with_columns(
                    [pl.lit(None).alias(c) for c in self.joined_result.get_colnames_secondary_without_groupby()],
                )
            missing.append(missing_left)

        if how in ["right", "outer"]:
            missing_top_right = self._calculate_missing_top_right()
            missing_bottom_right = self._calculate_missing_bottom_right()
            missing_right_within_groups = self._calculate_missing_overlaps(
                missing_top=missing_top_right,
                missing_bottom=missing_bottom_right,
            )
            if self.joined_result.groupby_args_given:
                missing_right = pl.concat(
                    [
                        missing_right_within_groups,
                        self.joined_result.groups_unique_to_right(),
                    ],
                )
            else:
                missing_right = missing_right_within_groups

            if include_missing_in_other:
                missing_right = missing_right.select(
                    [
                        pl.lit(None).alias(c)
                        for c in self.joined_result.by + self.joined_result.get_colnames_without_groupby()
                    ]
                    + [pl.col(self.joined_result.get_colnames_secondary_without_groupby())],
                )
            elif how == "right":
                missing_right = missing_right.rename(
                    dict(
                        zip(
                            missing_right.columns,
                            self.joined_result.secondary_frame.columns,
                        ),
                    ),
                )
            missing.append(missing_right)
        return pl.concat(
            missing,
            how="vertical_relaxed",
        )

    @staticmethod
    def _calculate_missing_overlaps(
        missing_top: pl.LazyFrame,
        missing_bottom: pl.LazyFrame,
    ) -> pl.LazyFrame:
        return missing_bottom.join(missing_top, on=missing_top.columns, how="semi").select(
            pl.col(missing_top.columns),
        )

    def missing_overlaps_left(
        self,
        *,
        include_right: bool = False,
    ) -> pl.LazyFrame:
        """Return the missing overlaps on the left side of the join."""
        missing_top_left = self._calculate_missing_top_left()
        missing_bottom_left = self._calculate_missing_bottom_left()

        missing_left = self._calculate_missing_overlaps(
            missing_top=missing_top_left,
            missing_bottom=missing_bottom_left,
        )

        if include_right:
            missing_left = missing_left.with_columns(
                [pl.lit(None).alias(c) for c in self.joined_result.get_colnames_secondary_without_groupby()],
            )

        return missing_left

    def _calculate_missing_self(
            self,
            df_column_names_without_groupby_ks: List[str],
            group_keys: List[str],
            mask_property: Literal["mask_1in2", "mask_2in1"],
    ) -> pl.LazyFrame:
        return (
            self.data.groupby(group_keys)
            .agg(
                pl.col(df_column_names_without_groupby_ks)
                .explode()
                .filter(
                    ~pl.col(mask_property).explode(),
                ),
            )
            .explode(df_column_names_without_groupby_ks)
            .drop_nulls()
        ).sort(group_keys)

    def _calculate_missing_other(
            self,
            df_column_names_without_groupby_ks: List[str],
            group_keys: List[str],
            starts_property: str,
            lengths_property: str,
    ) -> pl.LazyFrame:
        return (
            (
                self.data.groupby(group_keys)
                .agg(
                    [
                        pl.all().explode(),
                        pl.concat(
                            [
                                pl.int_range(
                                    0,
                                    pl.col(df_column_names_without_groupby_ks[0]).explode().len(),
                                    dtype=pl.UInt32,
                                ).explode(),
                                pl.int_ranges(
                                    start=pl.col(starts_property).explode(),
                                    end=pl.col(starts_property)
                                    .explode()
                                    .add(
                                        pl.col(lengths_property).explode(),
                                    ),
                                    dtype=pl.UInt32,
                                )
                                .explode()
                                .unique(),
                            ],
                        )
                        .drop_nulls()
                        .alias(DUPLICATED_INDICES_PROPERTY),
                    ],
                )
                .groupby(group_keys)
                .agg(
                    pl.col(df_column_names_without_groupby_ks)
                    .explode()
                    .take(
                        pl.col(DUPLICATED_INDICES_PROPERTY)
                        .explode()
                        .filter(
                            ~pl.col(DUPLICATED_INDICES_PROPERTY).explode().is_duplicated(),
                        ),
                    )
                    .explode(),
                )
            )
            .sort(group_keys)
            .explode(
                pl.col(df_column_names_without_groupby_ks),
            )
        )

    def _calculate_missing_top_left(self) -> pl.LazyFrame:
        return self._calculate_missing_self(
            df_column_names_without_groupby_ks=self.joined_result.get_colnames_without_groupby(),
            group_keys=self.joined_result.by,
            mask_property=MASK_2IN1_PROPERTY,
        )

    def _calculate_missing_bottom_left(self) -> pl.LazyFrame:
        return self._calculate_missing_other(
            df_column_names_without_groupby_ks=self.joined_result.get_colnames_without_groupby(),
            group_keys=self.joined_result.by,
            starts_property=STARTS_1IN2_PROPERTY,
            lengths_property=LENGTHS_1IN2_PROPERTY,
        )

    def _calculate_missing_top_right(self) -> pl.LazyFrame:
        return self._calculate_missing_self(
            df_column_names_without_groupby_ks=self.joined_result.get_joined_colnames_secondary(),
            group_keys=self.joined_result.by,
            mask_property=MASK_1IN2_PROPERTY,
        )

    def _calculate_missing_bottom_right(self) -> pl.LazyFrame:
        return self._calculate_missing_other(
            df_column_names_without_groupby_ks=self.joined_result.get_joined_colnames_secondary(),
            group_keys=self.joined_result.by,
            starts_property=STARTS_2IN1_PROPERTY,
            lengths_property=LENGTHS_2IN1_PROPERTY,
        )

    def _missing_overlaps_right(self) -> pl.LazyFrame:
        return self.data.select(
            [pl.lit(None).alias(c) for c in self.joined_result.get_colnames_without_groupby()]
            + [
                pl.col(
                    self.joined_result.get_colnames_secondary_without_groupby(),
                )
                .explode()
                .filter(
                    ~pl.col(MASK_1IN2_PROPERTY).explode().inspect("mask {}"),
                ),
            ],
        )
