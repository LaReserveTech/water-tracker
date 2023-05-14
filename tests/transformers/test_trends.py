"""Tests for Trends transformers."""

from unittest.mock import Mock

import numpy as np
import pytest
from water_tracker.transformers.trends import (
    TrendProperties,
    TrendThreshold,
)


@pytest.mark.parametrize(
    ("minimum_value", "value", "expected"),
    [
        (1, 0, False),
        (2, 3, True),
        (np.nan, 3, True),
    ],
)
def test_threshold_minimum(
    minimum_value: float,
    value: float,
    expected: bool,
) -> None:
    """Test for the minimal bound assertion of the TrendThreshold.

    Parameters
    ----------
    minimum_value : float
        Minimum value to verify.
    value : float
        Value to test.
    expected : bool
        Expected value.
    """
    threshold = TrendThreshold(
        return_value="test",
        minimum_value=minimum_value,
        maximum_value=np.nan,
    )
    assert threshold.verifies_minimum(value) == expected


@pytest.mark.parametrize(
    ("maximum_value", "value", "expected"),
    [
        (1, 0, True),
        (2, 3, False),
        (np.nan, 3, True),
    ],
)
def test_threshold_maximum(
    maximum_value: float,
    value: float,
    expected: bool,
) -> None:
    """Test for the maximum bound assertion of the TrendThreshold.

    Parameters
    ----------
    maximum_value : float
        Maximum value to verify.
    value : float
        Value to test.
    expected : bool
        Expected value.
    """
    threshold = TrendThreshold(
        return_value="test",
        minimum_value=np.nan,
        maximum_value=maximum_value,
    )
    assert threshold.verifies_maximum(value) == expected


@pytest.mark.parametrize(
    ("nb_years_history", "minimum_value", "maximum_value", "expected"),
    [
        (1.2, 1, 3, True),
        (1, 3, 5, False),
        (7, 2, 6, False),
        (-3, -5, 3, True),
        (-3, -2, 3, False),
        (-5.8, -6, -5, True),
        (-3, -6, -5, False),
        (3.9, np.nan, 7, True),
        (9, np.nan, 7, False),
        (3.9, 2, np.nan, True),
        (1, 2, np.nan, False),
    ],
)
def test_is_in_threshold(
    nb_years_history: float,
    minimum_value: float,
    maximum_value: float,
    expected: bool,
) -> None:
    """Test the is_in_threshold method of TrendThreshold.

    Parameters
    ----------
    nb_years_history : float
        Number of years of history for the Mock TrendProperty.
    minimum_value : float
        Threshold lower bound.
    maximum_value : float
        Threshold upper bound.
    expected : bool
        Expected value.
    """
    trend_prop_mock = Mock(TrendProperties)
    trend_prop_mock.nb_years_history = nb_years_history
    threshold = TrendThreshold(
        return_value="test",
        minimum_value=minimum_value,
        maximum_value=maximum_value,
    )
    assert threshold.is_in_threshold(trend_prop_mock) == expected
