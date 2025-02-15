"""This module manages the solving process for the optimal control problem."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ocp import OcpPhase
    from solution import OcpPhaseSolution, OcpTrajectory


def solve_ocp(ocp: OcpPhase) -> OcpTrajectory:
    raise NotImplementedError
