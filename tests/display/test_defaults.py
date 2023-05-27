"""Test for the defaults input values objects."""

import geopandas as gpd
import pytest
from pytest_mock import MockerFixture
from shapely import Point, Polygon
from water_tracker.display.defaults import DefaultDepartement


@pytest.fixture()
def depts_gdf() -> gpd.GeoDataFrame:
    """Departments GeoDataframe."""
    return gpd.GeoDataFrame(
        {
            "code": ["02", "03"],
        },
        geometry=[
            Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
            Polygon([[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]),
        ],
    )


@pytest.mark.parametrize(
    ("query_params", "expected"),
    [
        ({"lat": ["1"], "lon": ["1"]}, True),
        ({"lt": ["1"], "lon": ["1"]}, False),
        ({}, False),
    ],
)
def test_dep_check_params(
    query_params: dict[str, list[str]],
    expected: bool,
    mocker: MockerFixture,
) -> None:
    """Test DefaultDepartement.check_params.

    Parameters
    ----------
    query_params : dict[str, list[str]]
        Query parameters.
    expected : bool
        Expected check output.
    mocker : MockerFixture
        Mocker fixture.
    """
    mocker.patch("geopandas.read_file", return_value=gpd.GeoDataFrame())
    dept_default = DefaultDepartement(
        query_params=query_params,
        longitude_query_param="lon",
        latitude_query_param="lat",
    )
    assert dept_default.check_params() == expected


def test_dep_read_query_params(mocker: MockerFixture) -> None:
    """Test DefaultDepartment.read_query_params.

    Parameters
    ----------
    mocker : MockerFixture
        Mocker Fixture.
    """
    mocker.patch("geopandas.read_file", return_value=gpd.GeoDataFrame())
    query_params = {"lat": ["5"], "lon": ["1"]}
    dept_default = DefaultDepartement(
        query_params=query_params,
        longitude_query_param="lon",
        latitude_query_param="lat",
    )
    assert dept_default.read_query_params() == Point(1, 5)


@pytest.mark.parametrize(
    ("point", "expected_code"),
    [
        (Point(0.5, 0.5), "02"),
        (Point(1.5, 1.5), "03"),
    ],
)
def test_dep_get_point_department(
    point: Point,
    expected_code: str,
    mocker: MockerFixture,
    depts_gdf: gpd.GeoDataFrame,
) -> None:
    """Test DefaultDepartment.get_point_department.

    Parameters
    ----------
    point : Point
        Point to find the department of.
    expected_code : str
        Expected code department.
    mocker : MockerFixture
        Mocker Fixture.
    depts_gdf: gpd.GeoDataFrame
        Departments GeoDataFrame.
    """
    mocker.patch("geopandas.read_file", return_value=depts_gdf)
    query_params = {"lat": ["5"], "lon": ["1"]}
    dept_default = DefaultDepartement(
        query_params=query_params,
        longitude_query_param="lon",
        latitude_query_param="lat",
        geojson_code_field="code",
    )
    assert dept_default.get_point_department(point) == expected_code


@pytest.mark.parametrize(
    ("point"),
    [
        (Point(0, 0)),
        (Point(-1.5, 1.5)),
    ],
)
def test_dep_get_point_department_default(
    point: Point,
    mocker: MockerFixture,
    depts_gdf: gpd.GeoDataFrame,
) -> None:
    """Test DefaultDepartment.get_point_department for default value.

    Parameters
    ----------
    point : Point
        Point to find the department of.
    mocker : MockerFixture
        Mocker Fixture.
    depts_gdf : gpd.GeoDataFrame
        Departments GeoDataFrame.
    """
    mocker.patch("geopandas.read_file", return_value=depts_gdf)
    query_params = {"lat": ["5"], "lon": ["1"]}
    dept_default = DefaultDepartement(
        query_params=query_params,
        longitude_query_param="lon",
        latitude_query_param="lat",
        geojson_code_field="code",
    )
    expected = dept_default.default_value
    assert dept_default.get_point_department(point) == expected


@pytest.mark.parametrize(
    ("query_params"),
    [
        ({"lt": ["1"], "lon": ["1"]}),
        ({}),
    ],
)
def test_dep_value_default(
    query_params: dict,
    mocker: MockerFixture,
) -> None:
    """Test DefaultDepartment.value for default value.

    Parameters
    ----------
    query_params : dict
        Query parameters.
    mocker : MockerFixture
        Mocker Fixture.
    """
    mocker.patch("geopandas.read_file", return_value=gpd.GeoDataFrame())
    dept_default = DefaultDepartement(
        query_params=query_params,
        longitude_query_param="lon",
        latitude_query_param="lat",
    )
    assert dept_default.value == dept_default.default_value


@pytest.mark.parametrize(
    ("query_params", "expected_code"),
    [
        ({"lat": ["0.5"], "lon": ["0.5"]}, "02"),
        ({"lat": ["1.5"], "lon": ["1.5"]}, "03"),
    ],
)
def test_dep_value(
    query_params: dict,
    expected_code: str,
    mocker: MockerFixture,
    depts_gdf: gpd.geopandas,
) -> None:
    """Test DefaultDepartment.value.

    Parameters
    ----------
    query_params : dict
        Query parameters.
    expected_code : str
        Expected department code.
    mocker : MockerFixture
        MOcker Fixture.
    depts_gdf : gpd.geopandas
        Departments GeoDataFrame.
    """
    mocker.patch("geopandas.read_file", return_value=depts_gdf)
    dept_default = DefaultDepartement(
        query_params=query_params,
        longitude_query_param="lon",
        latitude_query_param="lat",
        geojson_code_field="code",
    )
    assert dept_default.value == expected_code
