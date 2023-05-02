"""Meteo France Connectors."""

from abc import ABC
from pathlib import Path

import pandas as pd

from water_tracker import BASE_DIR
from water_tracker.connectors.base import BaseConnector
from water_tracker.utils import load_dep_name_to_nb

resources_dept_path_relative = Path("resources/departments_regions.json")
resources_dept_path = BASE_DIR.joinpath(resources_dept_path_relative)


class MeteoFranceConnector(BaseConnector, ABC):
    """Connector Class to retrieve data from Meteo France."""

    def retrieve(self, params: dict) -> pd.DataFrame:
        """Retrieve data from Meteo France.

        Parameters
        ----------
        params : dict
            Parameters for the request.

        Returns
        -------
        pd.DataFrame
            Formatted output.
        """
        raw_df = pd.read_csv(**params)
        return self.format_output(raw_df)


class SSWIMFConnector(MeteoFranceConnector):
    """Connector class to retrieve SSWI data from Meteo France."""

    date_format: str = "%Y%m%d"

    columns_to_keep: dict[str, str] = {
        "LON": "longitude",
        "LAT": "latitude",
        "DATE": "date",
        "DECADE": "decade",
        "SSWI_DECAD": "sswi",
    }
    date_columns: list[str] = [
        "DATE",
    ]

    def retrieve(self, params: dict) -> pd.DataFrame:
        """Retrieve data from Meteo France.

        Parameters
        ----------
        params : dict
            Parameters for the request.

        Returns
        -------
        pd.DataFrame
            Formatted output.
        """
        return super().retrieve(params)


class PrecipitationsMFConnector(MeteoFranceConnector):
    """Connector class to retrieve precipitations data from Meteo France."""

    zone_column: str = "zone"
    columns_to_keep: dict[str, str] = {
        "Zone": zone_column,
        "RRSm Ag (mm)": "precip",
        "Nor RRSm Ag (mm)": "precip_norm",
        "Rap RRSm Ag": "ratio_precip",
    }
    date_columns: list[str] = []
    date_format: str | None = None

    def __init__(self) -> None:
        super().__init__()
        self.dept_mapping = load_dep_name_to_nb(resources_dept_path)

    def format_output(
        self,
        raw_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Format the output of the request function.

        Parameters
        ----------
        raw_df : pd.DataFrame
            Output of the request.

        Returns
        -------
        pd.DataFrame
            Formatted dataframe.
        """
        formatted = super().format_output(raw_df=raw_df)
        zone_col = formatted.pop(self.zone_column)
        formatted[self.zone_column] = zone_col.replace(self.dept_mapping)
        return formatted

    def retrieve(self, params: dict) -> pd.DataFrame:
        """Retrieve data from Meteo France.

        Parameters
        ----------
        params : dict
            Parameters for the request.

        Returns
        -------
        pd.DataFrame
            Formatted output.
        """
        return super().retrieve(params)
