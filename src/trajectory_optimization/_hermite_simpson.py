"""Transcribe and solve an optimal control problem with Hermite Simpson collocation.

This uses the Hermite Simpson separated (HSS) method as described in chapter 4 of:
Betts, J. T. (2020). Practical Methods for Optimal Control Using Nonlinear Programming (3rd ed.).
Society for Industrial and Applied Mathematics.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from solution import OcpPhaseSolution

if TYPE_CHECKING:
    from ocp import OcpPhase


def solve(ocp: OcpPhase) -> OcpPhaseSolution:
    pass
