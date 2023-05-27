"""Inputs objects."""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import pandas as pd

from water_tracker.display.defaults import (
    DefaultDepartement,
    DefaultInput,
    DefaultStation,
)

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator


DefaultInputT = TypeVar("DefaultInputT", bound="DefaultInput")


class BaseInput(ABC, Generic[DefaultInputT]):
    """Base class for inputs.

    Parameters
    ----------
    label : str
        Label for the input object.
    default_input : DefaultInputT
        Default Input object.
    """

    def __init__(
        self,
        label: str,
        default_input: DefaultInputT,
    ) -> None:
        self.label = label
        self._default = default_input

    @abstractmethod
    def build(self, container: "DeltaGenerator") -> Any | None:
        """Build the input.

        Parameters
        ----------
        container : DeltaGenerator
            Container to build the input object to.

        Returns
        -------
        Any | None
            Input value.
        """


class DepartmentInput(BaseInput[DefaultDepartement]):
    """Departments inputs.

    Parameters
    ----------
    label : str
        Label for the input object.
    default_input : DefaultInputT
        Default Input object.
    """

    @cached_property
    def options(self) -> list[str]:
        """Inputs options."""
        return [
            *map(DepartmentInput.format_dept, range(1, 20)),
            *["2A", "2B"],
            *map(DepartmentInput.format_dept, range(21, 96)),
        ]

    @staticmethod
    def format_dept(dept_nb: int) -> str:
        """Format departements numbers.

        Parameters
        ----------
        dept_nb : int
            Departement number.

        Returns
        -------
        str
            Formatted department number (1 -> '01', 10 -> '10', '2A' -> '2A').
        """
        return str(dept_nb).zfill(2)

    def build(self, container: "DeltaGenerator") -> str | None:
        """Build the input in a given container.

        Parameters
        ----------
        container : DeltaGenerator
            Container to build the input object to.

        Returns
        -------
        str | None
            Input value.
        """
        return container.selectbox(
            label=self.label,
            options=self.options,
            index=self.options.index(self._default.value),
        )


class StationInput(BaseInput[DefaultStation]):
    """Stations Input.

    Parameters
    ----------
    label : str
        Label for the input object.
    stations_df : pd.DataFrame
        Stations DataFrame.
    default_input : DefaultInputT
        Default Input object.
    """

    def __init__(
        self,
        label: str,
        stations_df: pd.DataFrame,
        default_input: DefaultStation,
    ) -> None:
        super().__init__(label, default_input)
        self._stations = stations_df

    @property
    def options(self) -> list[int]:
        """Input Options."""
        return self._stations.index.to_list()

    def format_func(self, row_index: int) -> str:
        """Format function to apply to stations dataframe index.

        Parameters
        ----------
        row_index : int
            Index of a row.

        Returns
        -------
        str
            Formatted string to display.
        """
        bss_code = self._stations.loc[row_index, "code_bss"]
        city_name = self._stations.loc[row_index, "nom_commune"]
        return f"{bss_code} ({city_name})"

    def build(self, container: "DeltaGenerator") -> int | None:
        """Build the input in a given container.

        Parameters
        ----------
        container : DeltaGenerator
            Container to build the input object to.

        Returns
        -------
        int | None
            Input value.
        """

        def format_func(row_index: int) -> str:
            return self.format_func(row_index=row_index)

        return container.selectbox(
            label=self.label,
            options=self.options,
            index=self.options.index(self._default.value),
            format_func=format_func,
        )
