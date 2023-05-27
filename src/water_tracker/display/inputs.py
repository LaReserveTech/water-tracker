"""Inputs objects."""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from water_tracker.display.defaults import DefaultDepartement, DefaultInput

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
    container : DeltaGenerator
        Container to build the input to.
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
        Any | None
            Input value.
        """
        return container.selectbox(
            label=self.label,
            options=self.options,
            index=self.options.index(self._default.value),
        )
