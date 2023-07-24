from typing import Optional, List, Literal

import polars as pl

from interval_frame.constants import ROW_NUMBER_PROPERTY, COUNT_PROPERTY


class GroupByJoinResult:
    def __init__(
        self,
        main_frame: pl.LazyFrame,
        secondary_frame: pl.LazyFrame,
        main_start: str,
        main_end: str,
        secondary_start: str,
        secondary_end: str,
        suffix: str,
        by: Optional[List[str]] = None,
        join_type: Literal["inner", "left", "outer", "semi", "anti", "cross"] = "inner",
        deduplicate_rows: bool = False,
    ):
        """
        Initializes the GroupByJoinResult object and performs a series of operations
        on the given data frames.

        Arguments:
        - main_frame: The main data frame.
        - secondary_frame: The secondary data frame.
        - main_start: The start of the main frame.
        - main_end: The end of the main frame.
        - secondary_start: The start of the secondary frame.
        - secondary_end: The end of the secondary frame.
        - suffix: The suffix to be added.
        - by: The columns to be grouped by. Default is None.
        - join_type: The type of join to be performed. Default is 'inner'.
        - deduplicate_rows: Whether to remove duplicate rows. Default is False.
        """
        self.secondary_start = secondary_start
        self.secondary_end = secondary_end
        self.secondary_start_renamed = self._rename_if_same(
            main_start,
            secondary_start,
            suffix,
        )
        self.secondary_end_renamed = self._rename_if_same(
            main_end,
            secondary_end,
            suffix,
        )
        self.columns = main_frame.columns
        self.main_frame = self._deduplicate_if_needed(main_frame, deduplicate_rows)
        self.secondary_frame = secondary_frame
        self.main_start = main_start
        self.main_end = main_end
        self.suffix = suffix
        self.groupby_args_given = by is not None
        self.by = by if self.groupby_args_given else [ROW_NUMBER_PROPERTY]
        self.joined = self._perform_join(join_type)

    @staticmethod
    def _rename_if_same(
        name1: str,
        name2: str,
        suffix: str,
    ) -> str:
        return name1 + suffix if name1 == name2 else name2

    @staticmethod
    def _deduplicate_if_needed(
        frame: pl.LazyFrame,
        deduplicate: bool,
    ) -> pl.LazyFrame:
        if deduplicate:
            return frame.groupby(frame.columns).agg([pl.all(), pl.count().alias(COUNT_PROPERTY)])

        return frame

    def _perform_join(
        self,
        join_type: Literal["inner", "left", "outer", "semi", "anti", "cross"] = "inner",
    ) -> pl.LazyFrame:
        sorted_main = self.main_frame.sort(self.main_start, self.main_end)
        sorted_secondary = self.secondary_frame.sort(
            self.secondary_start,
            self.secondary_end,
        )

        if not self.groupby_args_given:
            joined_frame = sorted_main.select(pl.all().implode()).join(
                sorted_secondary.select(pl.all().implode()),
                how="cross",
                suffix=self.suffix,
            )
            return joined_frame.with_row_count(ROW_NUMBER_PROPERTY)

        return (
            sorted_main.groupby(self.by)
            .all()
            .join(
                sorted_secondary.groupby(self.by).all(),
                left_on=self.by,
                right_on=self.by,
                suffix=self.suffix,
                how=join_type,
            )
        )

    def is_empty(self) -> bool:
        """
        Checks if either of the data frames or the joined frame is empty.

        Returns:
        - True if at least one of the frames is empty. False otherwise.
        """
        main_frame_empty = self._is_frame_empty(self.main_frame)
        secondary_frame_empty = self._is_frame_empty(self.secondary_frame)
        joined_frame_empty = self._is_frame_empty(self.joined)

        return main_frame_empty or secondary_frame_empty or joined_frame_empty

    @staticmethod
    def _is_frame_empty(frame: pl.LazyFrame) -> bool:
        return frame.first().collect().shape[0] == 0

    def get_joined_colnames_secondary(self) -> List[str]:
        """
        Returns the column names of the secondary frame after the join operation.

        Columns that are also present in the main frame will have the suffix added.
        """
        possibly_duplicated_cols = self._get_cols_excluding(
            self.secondary_frame.columns,
            self.by,
        )
        return self._add_suffix_if_in(
            possibly_duplicated_cols,
            self.main_frame.columns,
        )

    @staticmethod
    def _get_cols_excluding(
        cols: List[str],
        exclude_cols: List[str],
    ) -> List[str]:
        return [col for col in cols if col not in exclude_cols]

    def _add_suffix_if_in(
        self,
        cols: List[str],
        target_cols: List[str],
    ) -> List[str]:
        return [col + self.suffix if col in set(target_cols) else col for col in cols]

    def get_colnames_without_groupby(
        self,
    ) -> List[str]:
        """
        Returns the column names of the main frame excluding the 'groupby' columns.
        """
        return self._get_cols_excluding(
            self.main_frame.columns,
            self.by,
        )

    def get_colnames_secondary_without_groupby(
        self,
    ) -> List[str]:
        """
        Returns the column names of the secondary frame excluding the 'groupby' columns.
        """
        joined_colnames_secondary = self.get_joined_colnames_secondary()
        return self._get_cols_excluding(
            joined_colnames_secondary,
            self.by,
        )
