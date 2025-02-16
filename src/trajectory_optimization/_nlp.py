"""Nonlinear programming structures that are independent of the transcription method."""

import casadi as ca

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ocp import OcpPhase


class NlpPhase:
    def __init__(self, ocp: OcpPhase):
        self.ocp: OcpPhase = ocp
        self.opti: ca.Opti = ca.Opti()
        self.mesh: Mesh = ocp.options.mesh.mesh
