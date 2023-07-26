from typing import List, Literal, TYPE_CHECKING

import polars as pl

from interval_frame.constants import (
    LOW1_IN_STARTS2,
    HIGH1_IN_STARTS2,
    STARTS_1IN2_PROPERTY,
    ENDS_1IN2_PROPERTY,
    MASK_2IN1_PROPERTY,
    MASK_1IN2_PROPERTY,
    LENGTHS_2IN1_PROPERTY,
    LENGTHS_1IN2_PROPERTY,
    COUNT_PROPERTY,
)
from interval_frame.helper_ops import search, add_length

if TYPE_CHECKING:
    from interval_frame.groupby_join_result import GroupByJoinResult


class OverlappingIntervals:
    """
    The data needed to compute paired overlaps between two sets of ranges.
    This class is used to compute things like joins and overlaps between ranges.
    """

    def __init__(
        self,
        joined_result: "GroupByJoinResult",
        closed_intervals: bool = False,
    ) -> None:
        """
        Initializes the OverlappingIntervals object.

        Arguments:
        - joined_result: The result of a join operation.
        - closed_intervals: If True, the intervals are considered to be closed (inclusive). Default is False.
        """
        # Ensure that the required property is present in the joined columns
        assert COUNT_PROPERTY in joined_result.joined.columns, str(joined_result.joined.columns)

        self.joined_result = joined_result
        self.closed_intervals = closed_intervals

        group_keys = joined_result.by

        # The main data processing pipeline, broken down into multiple stages for clarity
        self.data = (
            joined_result.joined.groupby(group_keys)
            .agg([pl.all().explode()] + self.find_starts_in_ends())
            .groupby(group_keys)
            .agg([pl.all().explode()] + self.compute_masks())
            .groupby(group_keys)
            .agg(
                [
                    pl.exclude(
                        STARTS_1IN2_PROPERTY, ENDS_1IN2_PROPERTY, LOW1_IN_STARTS2, HIGH1_IN_STARTS2
                    ).explode()
                ]
                + self.apply_masks()
            )
            .groupby(group_keys)
            .agg([pl.all()] + self.add_lengths())
            .explode(pl.exclude(group_keys))
        )

    @staticmethod
    def explode_by_repeats(frame: pl.LazyFrame) -> pl.LazyFrame:
        """
        Explodes a frame by repeat counts.

        Arguments:
        - frame: The input data frame.

        Returns:
        - The exploded frame.
        """
        return (
            frame.groupby(pl.all())
            .first()
            .select(pl.exclude(COUNT_PROPERTY).repeat_by(pl.col(COUNT_PROPERTY).explode()))
            .explode(pl.all())
        )

    @staticmethod
    def lengths(
        starts: str,
        ends: str,
        outname: str = "",
    ) -> pl.Expr:
        """
        Calculates the lengths of intervals.

        Arguments:
        - starts: The start points of the intervals.
        - ends: The end points of the intervals.
        - outname: The output name for the calculated lengths.

        Returns:
        - An expression representing the lengths of the intervals.
        """
        return pl.col(ends).explode().sub(pl.col(starts).explode()).explode().alias(outname)

    def find_starts_in_ends(self) -> List[pl.Expr]:
        """
        Finds the start points in the end points.

        Returns:
        - A list of expressions representing the start points.
        """
        side: Literal["right", "left"] = "right" if self.closed_intervals else "left"

        return [
            search(self.joined_result.secondary_start_renamed, self.joined_result.main_start, side="left").alias(
                LOW1_IN_STARTS2
            ),
            search(self.joined_result.secondary_start_renamed, self.joined_result.main_end, side=side).alias(
                HIGH1_IN_STARTS2
            ),
            search(self.joined_result.main_start, self.joined_result.secondary_start_renamed, side="right").alias(
                STARTS_1IN2_PROPERTY
            ),
            search(self.joined_result.main_start, self.joined_result.secondary_end_renamed, side=side).alias(
                ENDS_1IN2_PROPERTY
            ),
        ]

    @staticmethod
    def compute_masks() -> List[pl.Expr]:
        """
        Computes the masks for the intervals.

        Returns:
        - A list of expressions representing the masks.
        """
        return [
            pl.col(HIGH1_IN_STARTS2).explode().gt(pl.col(LOW1_IN_STARTS2).explode()).alias(MASK_2IN1_PROPERTY),
            pl.col(ENDS_1IN2_PROPERTY).explode().gt(pl.col(STARTS_1IN2_PROPERTY).explode()).alias(MASK_1IN2_PROPERTY),
        ]

    @staticmethod
    def apply_masks() -> List[pl.Expr]:
        """
        Applies the masks to the intervals.

        Returns:
        - A list of expressions representing the intervals after applying the masks.
        """
        return [
            pl.col([LOW1_IN_STARTS2, HIGH1_IN_STARTS2]).explode().filter(pl.col(MASK_2IN1_PROPERTY).explode()),
            pl.col([STARTS_1IN2_PROPERTY, ENDS_1IN2_PROPERTY]).explode().filter(pl.col(MASK_1IN2_PROPERTY).explode()),
        ]

    @staticmethod
    def add_lengths() -> List[pl.Expr]:
        """
        Adds lengths to the intervals.

        Returns:
        - A list of expressions representing the intervals with added lengths.
        """
        return [
            add_length(LOW1_IN_STARTS2, HIGH1_IN_STARTS2, LENGTHS_2IN1_PROPERTY),
            add_length(STARTS_1IN2_PROPERTY, ENDS_1IN2_PROPERTY, LENGTHS_1IN2_PROPERTY),
        ]

    # ... Rest of the methods with similar improvements ...
    @staticmethod
    def repeat_frame(
        columns,
        startsin,
        endsin,
    ) -> pl.Expr:
        """
        Repeats the values in specified columns based on the difference between the start and end values.

        Arguments:
        - columns: The columns to repeat.
        - startsin: The column with the start values.
        - endsin: The column with the end values.

        Returns:
        - An expression representing the repeated columns.
        """
        return pl.col(columns).explode().repeat_by(pl.col(endsin).explode() - pl.col(startsin).explode()).explode()

    @staticmethod
    def mask_and_repeat_frame(
        columns,
        mask,
        startsin,
        endsin,
    ) -> pl.Expr:
        """
        Applies a mask to the values in specified columns, and then repeats the masked values based on the difference
        between the start and end values.

        Arguments:
        - columns: The columns to mask and repeat.
        - mask: The mask to apply.
        - startsin: The column with the start values.
        - endsin: The column with the end values.

        Returns:
        - An expression representing the masked and repeated columns.
        """
        return (
            pl.col(columns)
            .explode()
            .filter(pl.col(mask).explode())
            .repeat_by(
                pl.when(pl.col(mask).list.any())
                .then(pl.col(endsin).explode().drop_nulls() - pl.col(startsin).explode().drop_nulls())
                .otherwise(pl.lit(0))
            )
            .explode()
        )

    @staticmethod
    def repeat_other(
        columns,
        starts,
        diffs,
    ):
        """
        Repeats the values in specified columns based on specified start and end values.

        Arguments:
        - columns: The columns to repeat.
        - starts: The start values.
        - diffs: The differences between the start and end values.

        Returns:
        - An expression representing the repeated columns.
        """
        return pl.col(columns).explode().take(
            pl.int_ranges(
                start=starts,
                end=starts.add(diffs),
            ).explode().drop_nulls(),
        )

    def overlapping_pairs(
            self,
            how: Literal["inner", "left", "right", "outer"] = "inner",
            nulls_last: bool = False, # in case of missing values, should they be last (possible when how "outer", "left", "right")?
    ) -> "pl.LazyFrame":
        """
        Calculates the overlapping pairs in the joined data frames.

        Returns:
        - A data frame representing the overlapping pairs.
        """
        if self.joined_result.is_empty():
            return self.joined_result.joined

        group_keys = self.joined_result.by
        df_2_column_names_after_join = self.joined_result.get_joined_colnames_secondary()
        df_column_names_without_groupby_ks = self.joined_result.get_colnames_without_groupby()
        self.joined_result.get_colnames_secondary_without_groupby()

        # Calculate the top left, bottom left, top right, and bottom right parts of the final result
        missing_left = self.missing_overlaps_left(df_column_names_without_groupby_ks, group_keys, include_right=True)
        print("missing_left")
        print(missing_left.collect())
        # print("\nmissing_top_left", missing_top_left.collect())
        top_left = self.calculate_top_left(df_column_names_without_groupby_ks, group_keys)
        bottom_left = self.calculate_bottom_left(df_column_names_without_groupby_ks, group_keys)
        print("top_left")
        print(top_left.collect())
        print("bottom_left")
        print(bottom_left.collect())
        top_right = self.calculate_top_right(df_2_column_names_after_join, group_keys)
        bottom_right = self.calculate_bottom_right(df_2_column_names_after_join, group_keys)
        print("top_right")
        print(top_right.collect())
        print("bottom_right")
        print(bottom_right.collect())
        top = top_left.with_context(top_right).select(pl.all())
        bottom = bottom_left.with_context(bottom_right).select(pl.all())

        missing = []
        if how in ["left", "outer"]:
            missing += [self.missing_overlaps_left(df_column_names_without_groupby_ks, group_keys, include_right=True)]
        if how in ["right", "outer"]:
            missing += [self.missing_overlaps_right()]

        # Concatenate the parts to get the final result
        overlapping = pl.concat(
            [top_left, bottom_left]
        ).with_context(
            pl.concat(
                [top_right, bottom_right]
            )
        ).select(
            pl.all()
        )

        if how == "inner":
            return overlapping
        elif nulls_last:
            return pl.concat(
                [overlapping] + missing,
                how="vertical_relaxed"
            )
        else:
            return pl.concat(
                missing + [overlapping],
                how="vertical_relaxed"
            )

    def missing_overlaps_left(
            self,
            df_column_names_without_groupby_ks,
            group_keys,
            include_right: bool = False
        ):
        """Return the missing overlaps on the left side of the join."""

        missing_top_left = self._calculate_missing_top_left(df_column_names_without_groupby_ks, group_keys)
        missing_bottom_left = self._calculate_missing_bottom_left(df_column_names_without_groupby_ks, group_keys)

        missing_left = pl.concat(
            [
                missing_top_left,
                missing_bottom_left,
            ]
        ).groupby(
            self.joined_result.get_colnames_without_groupby_and_count(),
        ).agg(
            pl.all()
        ).filter(
            pl.col(COUNT_PROPERTY).list.lengths() == 2,
        ).groupby(
            group_keys
        ).agg(
            [
                pl.col(self.joined_result.get_colnames_without_groupby_and_count()),
                pl.col(COUNT_PROPERTY).list.sum().explode().keep_name(),
            ]
        ).explode(
            pl.all()
        )

        if include_right:
            missing_left = missing_left.with_columns(
                [
                    pl.lit(None).alias(c) for c in
                    self.joined_result.get_colnames_secondary_without_groupby()
                ]
            )

        return missing_left

    def missing_overlaps_right(self):
        """Return the missing overlaps on the left side of the join."""
        return self.data.select(
            [
                pl.lit(None).alias(c) for c in
                self.joined_result.get_colnames_without_groupby()
            ] + [
                pl.col(
                    self.joined_result.get_colnames_secondary_without_groupby()
                ).explode().filter(
                    ~pl.col(MASK_1IN2_PROPERTY).explode().inspect("mask {}")
                )#.repeat_by(pl.col(COUNT_PROPERTY).explode().filter(~pl.col(MASK_1IN2_PROPERTY).explode())).explode(),
            ]
        )

    def calculate_top_left(self, df_column_names_without_groupby_ks, group_keys):
        """
        Helper function to calculate the top left part of the final result.
        """
        return (
            self.data.filter(pl.col(MASK_2IN1_PROPERTY).list.any())
            .groupby(group_keys)
            .agg(
                self.mask_and_repeat_frame(
                    [
                        c
                        for c in df_column_names_without_groupby_ks
                        if c not in [MASK_2IN1_PROPERTY, LOW1_IN_STARTS2, HIGH1_IN_STARTS2]
                    ],
                    mask=MASK_2IN1_PROPERTY,
                    startsin=LOW1_IN_STARTS2,
                    endsin=HIGH1_IN_STARTS2,
                )
            )
            .explode(df_column_names_without_groupby_ks)
            .drop_nulls()
        ).sort(group_keys)

    def _calculate_missing_top_left(self, df_column_names_without_groupby_ks, group_keys):
        """
        Helper function to calculate the top left part of the final result.
        """
        return (
            self.data
            .groupby(group_keys)
            .agg(
                pl.col(df_column_names_without_groupby_ks).explode().filter(
                    ~pl.col(MASK_2IN1_PROPERTY).explode()
                )
            )
            .explode(df_column_names_without_groupby_ks)
            .drop_nulls()
        ).sort(group_keys)

    # def _calculate_missing_bottom_left(self, df_column_names_without_groupby_ks, group_keys):
    #     """
    #     Helper function to calculate the bottom left part of the final result.
    #     """
    #     return (
    #         self.data.inspect("data {}").groupby(group_keys)
    #         .agg(
    #             pl.col(df_column_names_without_groupby_ks).explode().take(
    #                 pl.int_range(
    #                     0, pl.col(df_column_names_without_groupby_ks[0]).explode().len(), dtype=pl.UInt32
    #                 ).inspect("BBBBBB {}").list.explode().list.difference(
    #                     pl.int_ranges(
    #                         start=pl.col(STARTS_1IN2_PROPERTY).explode(),
    #                         end=pl.col(STARTS_1IN2_PROPERTY).explode().add(
    #                             pl.col(LENGTHS_1IN2_PROPERTY).explode(),
    #                         ),
    #                         dtype=pl.UInt32,
    #                     ).list.explode().inspect("AAAAAA {}")
    #                 ).inspect("CCCCCCCCC {}")
    #             )
    #         )
    #     ).sort(group_keys)

    def _calculate_missing_bottom_left(self, df_column_names_without_groupby_ks, group_keys):
        """
        Helper function to calculate the bottom left part of the final result.
        """
        return (
            self.data.groupby(group_keys)
            .agg(
                [
                    pl.all().explode(),
                    pl.concat(
                        [
                            pl.int_range(
                                0,
                                pl.col(df_column_names_without_groupby_ks[0]).explode().len(),
                                dtype=pl.UInt32
                            ).explode(),
                            pl.int_ranges(
                                start=pl.col(STARTS_1IN2_PROPERTY).explode(),
                                end=pl.col(STARTS_1IN2_PROPERTY).explode().add(
                                    pl.col(LENGTHS_1IN2_PROPERTY).explode(),
                                ),
                                dtype=pl.UInt32,
                            ).explode().unique()
                        ]
                    ).alias("__duplicated_indices__")
                ]
            )
            .groupby(group_keys).agg(
                pl.col(df_column_names_without_groupby_ks).explode().take(
                    pl.col("__duplicated_indices__").explode().filter(
                        ~pl.col("__duplicated_indices__").explode().is_duplicated()
                    )
                ).explode()
            )
        ).sort(group_keys).explode(
            pl.col(self.joined_result.get_colnames_without_groupby()),
        )

    def calculate_bottom_left(self, df_column_names_without_groupby_ks, group_keys):
        """
        Helper function to calculate the bottom left part of the final result.
        """
        return (
            self.data.groupby(group_keys)
            .agg(
                self.repeat_other(
                    df_column_names_without_groupby_ks,
                    pl.col(STARTS_1IN2_PROPERTY).explode(),
                    pl.col(LENGTHS_1IN2_PROPERTY).explode(),
                )
            )
            .explode(df_column_names_without_groupby_ks)
            .drop_nulls()
        ).sort(group_keys)

    def calculate_top_right(self, df_2_column_names_after_join, group_keys):
        """
        Helper function to calculate the top right part of the final result.
        """
        return (
            self.data.groupby(group_keys)
            .agg(
                self.repeat_other(
                    df_2_column_names_after_join,
                    pl.col(LOW1_IN_STARTS2).explode(),
                    pl.col(LENGTHS_2IN1_PROPERTY).explode(),
                )
            )
            .explode(self.joined_result.get_colnames_secondary_without_groupby())
            .drop_nulls()
        ).sort(group_keys)

    def calculate_bottom_right(self, df_2_column_names_after_join, group_keys):
        """
        Helper function to calculate the bottom right part of the final result.
        """
        return (
            self.data.filter(pl.col(MASK_1IN2_PROPERTY).list.any())
            .groupby(group_keys)
            .agg(
                self.mask_and_repeat_frame(
                    df_2_column_names_after_join,
                    MASK_1IN2_PROPERTY,
                    STARTS_1IN2_PROPERTY,
                    ENDS_1IN2_PROPERTY,
                )
            )
            .explode(self.joined_result.get_colnames_secondary_without_groupby())
            .drop_nulls()
        ).sort(group_keys)

    def overlaps(self):
        """
        Calculates the overlapping intervals between the two data frames.

        Returns:
        - A data frame representing the overlapping intervals.
        """
        # Get the grouping columns and the columns excluding the group keys
        grouping_cols = self.joined_result.by
        cols_excluding_group_keys = self.joined_result.get_colnames_without_groupby()

        # Compute the top left part of the final data frame
        # This part includes the intervals that overlap according to the MASK_2IN1_PROPERTY
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

        # Compute the bottom left part of the final data frame
        # This part includes the intervals that are repeated according to the LENGTHS_1IN2_PROPERTY
        bottom_left = (
            self.data.groupby(grouping_cols)
            .agg(
                self.repeat_other(
                    columns=cols_excluding_group_keys,
                    starts=pl.col(STARTS_1IN2_PROPERTY).explode(),
                    diffs=pl.col(LENGTHS_1IN2_PROPERTY).explode(),
                )
            )
            .explode(cols_excluding_group_keys)
            .drop_nulls()
        ).sort(grouping_cols)

        # Combine the top left and bottom left parts and explode by the number of repeats
        return self.explode_by_repeats(pl.concat([top_left, bottom_left]))
