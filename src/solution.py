import numpy as np
import numpy.typing as npt


class Limits:
    def __init__(self, low: float = -np.inf, upp: float = np.inf) -> None:
        self.low: float = low
        self.upp: float = upp


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
