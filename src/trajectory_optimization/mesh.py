"""A 1d temporal mesh for optimal control problems."""

import numpy as np
import numpy.typing as npt


class OcpMesh:
    """A 1d temporal mesh for optimal control problems.

    Attrr:
        col_pt_tau (npt.NDArray): A 1d array of values on the interval [0, 1] that define the
            nondimensionalized mesh. These are the locations in time, normalized by the duration
            of the solution. For example, a solution duration of 20 seconds with col_pt values of
            [0.0, 0.25, 0.5, 0.75, 1.0] would correspond to solution times at the collocation
            points of [0.0, 5.0, 10.0, 15.0, 20.0].
        grid_tau (npt.NDArray): A 1d array of values on the interval [0, 1] for the
            nondimensionalized mesh for the entire time grid, included non-collocation points.
            Similar to col_pt_tau, but for every grid point.
        col_pt_delta_tau (npt.NDArray): A 1d array of values for the non-dimensionalized step size
            between tau points. For example, a col_pt_tau of [0.0, 0.25, 0.5, 0.75, 1.0] would
            correspond to a col_pt_delta_tau of [0.25, 0.25, 0.25, 0.25]. Not to be confused
            with Delta Tau Delta, a United States based international collegiate fraternity.
        grid_delta_tau (npt.NDArray): A 1d array of values for the non-dimensionalized step size
            between grid_tau points. Similar to col_pt_delta_tau, but for every grid point.
    """

    def __init__(self, col_pt_tau: npt.NDArray):
        self._col_pt_tau: npt.NDArray
        self.col_pt_tau = col_pt_tau

    @property
    def col_pt_tau(self) -> npt.NDArray:
        return self._col_pt_tau

    @col_pt_tau.setter
    def col_pt_tau(self, arr: npt.NDArray) -> npt.NDArray:
        sorted_arr: npt.NDArray = np.sort(arr)
        # Validate.
        if sorted_arr[0] != 0.0:
            raise ValueError(f"col_pt_tau lowest value must be zero. Got: {arr}")
        if sorted_arr[-1] != 1.0:
            raise ValueError(f"col_pt_tau highest value must be one. Got: {arr}")
        self._col_pt_tau = sorted_arr

    @property
    def grid_tau(self) -> npt.NDArray:
        # Compute midpoints between consecutive elements
        midpoints: npt.NDArray = (self.col_pt_tau[:-1] + self.col_pt_tau[1:]) / 2
        # Concatenate original array and midpoints, then sort.
        new_arr: npt.NDArray = np.sort(np.concatenate((self.col_pt_tau, midpoints)))
        return new_arr

    @property
    def col_pt_delta_tau(self) -> npt.NDArray:
        return np.diff(self.col_pt_tau)

    @property
    def grid_delta_tau(self) -> npt.NDArray:
        return np.diff(self.col_pt_tau)
