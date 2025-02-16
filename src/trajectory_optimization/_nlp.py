"""Nonlinear programming structures that are independent of the transcription method."""

from __future__ import annotations
import casadi as ca

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ocp import OcpPhase
    from mesh import OcpMesh


class NlpPhase:
    def __init__(self, ocp: OcpPhase):
        self.ocp: OcpPhase = ocp
        self.opti: ca.Opti = ca.Opti()
        self.mesh: OcpMesh = ocp.options.mesh.mesh
