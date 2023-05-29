"""Test for Inputs."""

from unittest.mock import Mock

import pandas as pd
import pytest
from water_tracker.display.inputs import DepartmentInput, StationInput


@pytest.fixture()
def dep_input() -> DepartmentInput:
    """Create department input object.

    Returns
    -------
    Mock
        Department Input.
    """
    dep_default = Mock()
    dep_default.value = "01"
    return DepartmentInput(label="test", default_input=dep_default)


@pytest.fixture()
def station_input() -> StationInput:
    """Create Station input object.

    Returns
    -------
    StationInput
        Station Input.
    """
    station_default = Mock()
    station_default.value = "code0"
    stations_df = pd.DataFrame(
        {
            "bss": ["code0", "code1"],
            "city": ["city0", "city1"],
        },
    )
    return StationInput(
        label="test",
        stations_df=stations_df,
        default_input=station_default,
        bss_field_name="bss",
        city_field_name="city",
    )


def test_dep_options(dep_input: DepartmentInput) -> None:
    """Test DefaultDepartment.options.

    Only checks that 96 departments are in options and
    that "2A" and "2B" are among them.

    Parameters
    ----------
    default_dep : DefaultDepartement
        Defqult department Mock.
    """
    depts_len = 96
    assert len(dep_input.options) == depts_len
    assert "2A" in dep_input.options
    assert "2B" in dep_input.options


def test_station_options(station_input: StationInput) -> None:
    """Test SatationInput.options.

    Parameters
    ----------
    station_input : StationInput
        Station Input.
    """
    assert station_input.options == station_input.stations.index.to_list()


def test_station_format(station_input: StationInput) -> None:
    """Test StationInput.format_func.

    Parameters
    ----------
    station_input : StationInput
        Station Input.
    """
    assert station_input.format_func(0) == "code0 (city0)"
