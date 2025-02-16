"""Optimal control problem solution related structures."""

from __future__ import annotations
import copy
import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d


class OcpTrajectory:
    def __init__(self) -> None:
        self.time: npt.NDArray = np.array([])
        self.state: dict[str, npt.NDArray] = {}
        self.control: dict[str, npt.NDArray] = {}
        self.parameter: dict[str, float] = {}

    def interpolate(self, times: npt.NDArray, kind: str = "cubic") -> OcpTrajectory:
        new_traj: OcpTrajectory = copy.deepcopy(self)
        new_traj.time = times

        # Check to make sure the times can be interpolated.
        _, counts = np.unique(self.time, return_counts=True)
        has_duplicate_times = np.any(counts > 1)
        if has_duplicate_times:
            msg = "The trajectory has multiple of the same time values. Cannot interpolate. "
            msg += f"Times: {self.time}"
            raise ValueError(msg)
        # Interpolate.
        for name, arr in self.state.items():
            interp = interp1d(self.time, arr, fill_value="extrapolation", kind=kind)
            new_traj.state[name] = interp(times)
        for name, arr in self.control.items():
            interp = interp1d(self.time, arr, fill_value="extrapolation", kind=kind)
            new_traj.control[name] = interp(times)

        return new_traj


# TODO: Make the actual solution interpolant that corresponds with the discretization method.
class OcpPhaseSolution:
    def __init__(self) -> None:
        raise NotImplementedError

    def eval(self) -> OcpTrajectory:
        raise NotImplementedError
