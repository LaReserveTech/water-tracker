"""Tools to compute defaults values for user inputs."""

from abc import ABC, abstractproperty
from functools import cached_property
from pathlib import Path
from typing import Generic, TypeVar

import geopandas as gpd
from shapely import Point

from water_tracker import BASE_DIR

depts_geojson_path_relative = Path("resources/departments_geojson.json")
depts_geojson_path = BASE_DIR.joinpath(depts_geojson_path_relative)

DefaultValueT = TypeVar("DefaultValueT")


class DefaultInput(ABC, Generic[DefaultValueT]):
    """Base class for default user inputs."""

    @abstractproperty
    def value(self) -> DefaultValueT:
        """Value to use as default."""


class DefaultDepartement(DefaultInput[str]):
    """Default input for department selection from query_params.

    Parameters
    ----------
    query_params : dict[str, list[str]]
        Query parameters shown in the browser's url.
    longitude_query_param : str
        Name of the latitude field.
    latitude_query_param : str
        Name of the longitude field.
    depts_geojson_path : Path, optional
        Path to the geojson with departments boundaries.
        , by default depts_geojson_path
    geojson_code_field : str, optional
        Geojson field with departments code., by default "code"
    """

    _default_dept_nb: str = "01"

    def __init__(
        self,
        query_params: dict[str, list[str]],
        longitude_query_param: str,
        latitude_query_param: str,
        depts_geojson_path: Path = depts_geojson_path,
        geojson_code_field: str = "code",
    ) -> None:
        self.lat_param = latitude_query_param
        self.lon_param = longitude_query_param
        self._depts_path = depts_geojson_path
        self.geojson_code_field = geojson_code_field
        self.query_params = query_params

    @cached_property
    def departments_geojson(self) -> gpd.GeoDataFrame:
        """Geodataframe with departments polygons."""
        return gpd.read_file(self._depts_path)

    @property
    def query_params(self) -> dict:
        """Query parameters."""
        return self._query

    @query_params.setter
    def query_params(self, query_params: dict[str, list[str]]) -> None:
        self._query = query_params
        self._valid_query_params = self.check_params()

    @property
    def value(self) -> str:
        """Value to use a default for department."""
        if self._valid_query_params:
            point = self.read_query_params()
            return self.get_point_department(point)
        return self._default_dept_nb

    def check_params(self) -> bool:
        """Check if the query parameters have latitude and longitude fields.

        Returns
        -------
        bool
            True if the query_parameters has fields for longitude and latitude.
        """
        is_lat_in_key = self.lat_param in self._query
        is_lon_in_key = self.lon_param in self._query
        return is_lat_in_key and is_lon_in_key

    def read_query_params(self) -> Point:
        """Read the query parameters and returns the corresponding Point.

        Returns
        -------
        Point
            Point corresponding to the query parameters.
        """
        lat = float(self._query[self.lat_param][0])
        lon = float(self._query[self.lon_param][0])
        return Point(lon, lat)

    def get_point_department(self, point: Point) -> str:
        """Find department to which the point belongs to.

        Parameters
        ----------
        point : Point
            Point to find the departement of.

        Returns
        -------
        str
            Departement's code.
        """
        contains_point = self.departments_geojson.contains(point)
        if not contains_point.any():
            return self._default_dept_nb
        first_containing = self.departments_geojson[contains_point].iloc[0]
        return first_containing[self.geojson_code_field]
