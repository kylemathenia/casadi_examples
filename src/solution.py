"""Optimal control problem solution related structures."""

import numpy as np
import numpy.typing as npt


class OcpTrajectory:
    def __init__(self) -> None:
        self.time: npt.NDArray = np.array([])
        self.state: dict[str, npt.NDArray] = {}
        self.control: dict[str, npt.NDArray] = {}
        self.parameter: dict[str, float] = {}


# TODO
class OcpPhaseSolution:
    def __init__(self) -> None:
        raise NotImplementedError

    def eval(self) -> OcpTrajectory:
        raise NotImplementedError
