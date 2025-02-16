"""Options for solving an optimal control problem."""

from mesh import OcpMesh


class OcpOptions:
    def __init__(self) -> None:
        self.mesh: MeshOptions = MeshOptions()
        self.nlp: MeshOptions = MeshOptions()


class MeshOptions:
    def __init__(self):
        self.mesh: OcpMesh = OcpMesh.from_evenly_distributed(n_col_pts=20)


class NlpOptions:
    def __init__(self):
        # These are options to be passed to the solver.
        # For ipopt see: https://coin-or.github.io/Ipopt/OPTIONS.html
        self.solver_options: dict = {}
        # These are options for casadi for solving nlp's.
        self.plugin_options: dict = {}
