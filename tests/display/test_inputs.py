"""Test for Inputs."""

from unittest.mock import Mock

import pytest
from water_tracker.display.defaults import DefaultDepartement
from water_tracker.display.inputs import DepartmentInput


@pytest.fixture()
def default_dep() -> Mock:
    """Create default department object Mock.

    Returns
    -------
    Mock
        Default department Mock.
    """
    dep_default = Mock()
    dep_default.value = "01"
    return dep_default


def test_options(default_dep: DefaultDepartement) -> None:
    """Test DefaultDepartment.options.

    Only checks that 96 departments are in options and
    that "2A" and "2B" are among them.

    Parameters
    ----------
    default_dep : DefaultDepartement
        Defqult department Mock.
    """
    dep_input = DepartmentInput(label="test", default_input=default_dep)
    depts_len = 96
    assert len(dep_input.options) == depts_len
    assert "2A" in dep_input.options
    assert "2B" in dep_input.options
