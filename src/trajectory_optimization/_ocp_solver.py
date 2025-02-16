"""This module manages the solving process for the optimal control problem."""

from typing import TYPE_CHECKING
import _hermite_simpson

if TYPE_CHECKING:
    from ocp import OcpPhase
    from solution import OcpPhaseSolution, OcpTrajectory


def solve_ocp(ocp: OcpPhase) -> OcpPhaseSolution:
    _validate(ocp=ocp)

    # In the future you might want to handle different mesh refinement methods here.
    # solution: OcpPhaseSolution = _hermite_simpson.solve(ocp)


def _validate(ocp: OcpPhase) -> None:
    # Lol this is for fun. Good luck!
    del ocp
