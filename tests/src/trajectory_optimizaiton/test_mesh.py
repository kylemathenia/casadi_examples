"""Test the mesh.py module."""

import numpy as np
import pytest
from ....src.trajectory_optimization.mesh import OcpMesh


class TestOcpMesh:
    """Tests for the OcpMesh class."""

    @pytest.mark.parametrize(
        "col_pt_tau, expected",
        [
            (
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
            ),
            ([0.0, 0.3, 0.6, 1.0], [0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0]),
        ],
    )
    def test_grid_tau(self, col_pt_tau, expected) -> None:
        """Test grid_tau calculation with expected midpoints."""
        mesh = OcpMesh(np.array(col_pt_tau))
        np.testing.assert_almost_equal(mesh.grid_tau, np.array(expected))

    def test_col_pt_delta_tau(self) -> None:
        """Test col_pt_delta_tau calculates differences correctly."""
        col_pt_tau = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        expected_deltas = np.array([0.25, 0.25, 0.25, 0.25])
        mesh = OcpMesh(col_pt_tau)
        np.testing.assert_almost_equal(mesh.col_pt_delta_tau, expected_deltas)

    def test_grid_delta_tau(self) -> None:
        """Test grid_delta_tau (should be same as col_pt_delta_tau for now)."""
        col_pt_tau = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        expected_deltas = np.array([0.25, 0.25, 0.25, 0.25])
        mesh = OcpMesh(col_pt_tau)
        np.testing.assert_almost_equal(mesh.grid_delta_tau, expected_deltas)

    @pytest.mark.parametrize(
        "invalid_col_pt_tau, error_message",
        [
            ([0.1, 0.25, 0.5, 0.75, 1.0], "col_pt_tau lowest value must be zero"),
            ([0.0, 0.25, 0.5, 0.75, 0.9], "col_pt_tau highest value must be one"),
        ],
    )
    def test_invalid_col_pt_tau(self, invalid_col_pt_tau, error_message):
        """Test validation of col_pt_tau boundary values."""
        with pytest.raises(ValueError, match=error_message):
            OcpMesh(np.array(invalid_col_pt_tau))

    def test_col_pt_sorted(self) -> None:
        mesh: OcpMesh = OcpMesh(np.array([0.0, 0.5, 0.25, 1.0]))
        assert np.allclose(mesh.col_pt_tau, [0.0, 0.25, 0.5, 1.0])

    def test_col_pt_tau_setter(self) -> None:
        """Test col_pt_tau setter sorts input and enforces rules."""
        mesh = OcpMesh(np.array([0.0, 0.5, 1.0]))  # Valid input
        mesh.col_pt_tau = np.array([1.0, 0.0, 0.25, 0.75, 0.5])  # Should be sorted
        expected_sorted = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_equal(mesh.col_pt_tau, expected_sorted)
