"""Compute trends to have some comparison basis."""
import datetime as dt
import math

import numpy as np
import pandas as pd


class ExistingColumnNameError(Exception):
    """Wrong column name."""

    def __init__(self, column_name: str, *args: object) -> None:
        super().__init__(
            f"{column_name} already exists and would have been erased.",
            *args,
        )


class AverageTrendEvaluation:
    """Average Evaluation / Relevance."""

    nb_years_out_ref: int = 5
    min_trends_year_nb: int = 3
    evaluation_thresholds: dict[str, dict[str, float]] = {
        "bad": {"min": 3, "max": 5},
        "correct": {"min": 5, "max": 10},
        "good": {"min": 10, "max": 15},
        "very good": {"min": 15, "max": 25},
        "excellent": {"min": 25, "max": np.nan},
    }

    def __init__(
        self,
        measure_start: dt.datetime,
        measure_end: dt.datetime,
        point_number: int,
    ) -> None:
        self.measure_start = measure_start
        self.measure_end = measure_end
        self.point_number = point_number

    @property
    def estimated_coverage(self) -> float:
        """Mean estimated coverage.

        Returns
        -------
        float
            Point of measure / days in measure period
        """
        measure_span = self.measure_end - self.measure_start
        max_measuring_points = measure_span.days
        return self.point_number / max_measuring_points

    @property
    def has_enough_data(self) -> bool:
        """Indication if the stations has enough history to compute trend.

        Returns
        -------
        bool
            True if there's enough data to compute a trend.
        """
        measure_period = (self.measure_end - self.measure_start).days
        measure_years = measure_period / 365.25
        minimum_years_nb = self.nb_years_out_ref + self.min_trends_year_nb
        return measure_years >= minimum_years_nb

    def get_trend_boundaries(
        self,
    ) -> tuple[dt.datetime | None, dt.datetime | None]:
        """Compute time boundaries for the trend data.

        Returns
        -------
        tuple[dt.datetime | None, dt.datetime | None]
            First date for trend, last date for trend.
        """
        if not self.has_enough_data:
            return None, None

        days_out_ref = math.ceil(365.25 * self.nb_years_out_ref)
        ref_start_date = self.measure_start
        ref_end_date = self.measure_end - dt.timedelta(days=days_out_ref)
        return ref_start_date, ref_end_date


class AverageTrend:
    """Transform data to add historic averaged values as reference."""

    day_of_year_column: str = "day_of_year"
    mean_values_column: str = "mean_value"

    def __init__(self) -> None:
        pass

    def add_days_of_year_column(
        self,
        dates_df: pd.DataFrame,
        dates_column: str,
        remove: bool = False,
    ) -> pd.DataFrame:
        """Add a 'day of year number' column to a DataFrame.

        Parameters
        ----------
        dates_df : pd.DataFrame
            DataFrame with regular dates in at least one column.
        dates_column : str
            Name of the column with dates values.
        remove : bool, optional
            Whether to remove the original date column or not.
            , by default False

        Returns
        -------
        pd.DataFrame
            Copy of dates_df with an additional column (named
            self.day_of_year_column) with the day of the year number.
            If 'remove' is True, the original dates column is removed.

        Raises
        ------
        ExistingColumnNameError
            If self.day_of_year_column is already the name of an existing
            column.
        """
        # get dates column
        if remove:
            dates = pd.to_datetime(dates_df.pop(dates_column))
        else:
            dates = pd.to_datetime(dates_df[dates_column])
        # transform to day of year (=> number between 1 and 366)
        days_of_year = dates.dt.day_of_year
        # add day of year column
        if self.day_of_year_column in dates_df.columns:
            raise ExistingColumnNameError(self.day_of_year_column)
        dates_df[self.day_of_year_column] = days_of_year
        return dates_df

    def compute_reference_values(
        self,
        history_days_of_year: pd.DataFrame,
        values_column: str,
    ) -> pd.DataFrame:
        """Compute the average value over the historic data.

        Parameters
        ----------
        history_days_of_year : pd.DataFrame
            DataFrame with the column self.day_of_year_column containing
            the day of year number.
        values_column : str
            Name of the column with the values to average.

        Returns
        -------
        pd.DataFrame
            DataFrame with the values averaged for each day of year.
        """
        # rename values_column
        values_column = history_days_of_year.pop(values_column)
        history_days_of_year[self.mean_values_column] = values_column
        # group by day
        day_group = history_days_of_year.groupby(self.day_of_year_column)
        # compute average value over the years
        return day_group.mean().reset_index()

    def transform(
        self,
        historical_df: pd.DataFrame,
        present_df: pd.DataFrame,
        dates_column: str,
        values_column: str,
    ) -> pd.DataFrame:
        """Add to present data the day-by-day average of 'value_column'.

        The average is computed over 'historical_df''s data. The joint is
        a left joint, performed over 'present_df'. Therefore, no data will be
        removed from 'present_df', but if 'historical_df' does not cover
        all days of a year, the average value will be 'np.nan'.

        Parameters
        ----------
        historical_df : pd.DataFrame
            DataFrame with historical values.
        present_df : pd.DataFrame
            DataFrame with present values.
        dates_column : str
            Name of the column containing date informations. Both
            'historical_df' and 'present_df' are supposed to have this column.
        values_column : str
            Name of the column containing the value to average. Both
            'historical_df' and 'present_df' are supposed to have this column.

        Returns
        -------
        pd.DataFrame
            Joint between present data and historical data average.
        """
        # Copy dataframes to avoid prevent modifications
        history_copy = historical_df[[dates_column, values_column]].copy()
        present_copy = present_df.copy()
        # replace date column by day of year
        history_days_of_year = self.add_days_of_year_column(
            dates_df=history_copy,
            dates_column=dates_column,
            remove=True,
        )
        present_day_of_year = self.add_days_of_year_column(
            dates_df=present_copy,
            dates_column=dates_column,
            remove=False,
        )
        # Compute reference values for historical data
        historical_reference = self.compute_reference_values(
            history_days_of_year=history_days_of_year,
            values_column=values_column,
        )
        # Joined averaged historical data to present data
        return present_day_of_year.merge(
            how="left",
            right=historical_reference,
            left_on=self.day_of_year_column,
            right_on=self.day_of_year_column,
        )
