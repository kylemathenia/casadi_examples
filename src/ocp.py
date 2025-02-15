"""Build an optimal control problem in continuous time."""

from typing import Optional
import casadi as ca
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


class OcpVectorSymbol:
    def __init__(
        self,
        name: str,
        bounds_c: Optional[Limits],
        bounds_i: Optional[Limits],
        bounds_f: Optional[Limits],
        guess: Optional[npt.NDArray] = None,
        scale: float = 1.0,
        units: Optional[str] = None,
    ) -> None:
        self.name: str = name
        self.sym_c: ca.MX = ca.MX.sym(name)
        self.sym_i: ca.MX = ca.MX.sym(name + "_i")
        self.sym_f: ca.MX = ca.MX.sym(name + "_f")
        self.bounds_c: Limits = Limits() if bounds_c is None else bounds_c
        self.bounds_i: Limits = Limits() if bounds_i is None else bounds_i
        self.bounds_f: Limits = Limits() if bounds_f is None else bounds_f
        self.guess: npt.NDArray = np.array([0, 0]) if guess is None else guess
        self.scale: float = scale
        self.units: Optional[str] = units


class OcpVectorExpr:
    def __init__(
        self,
        name: str,
        expr: ca.MX,
        bounds_c: Optional[Limits],
        bounds_i: Optional[Limits],
        bounds_f: Optional[Limits],
        scale: Optional[float] = None,
        units: Optional[str] = None,
    ) -> None:
        self.name: str = name
        self.expr: ca.MX = expr
        self.bounds_c: Limits = Limits() if bounds_c is None else bounds_c
        self.bounds_i: Limits = Limits() if bounds_i is None else bounds_i
        self.bounds_f: Limits = Limits() if bounds_f is None else bounds_f
        self.scale: Optional[float] = scale
        self.units: Optional[str] = units


class OcpScalarSymbol:
    def __init__(
        self,
        name: str,
        bounds: Optional[Limits],
        guess: Optional[float] = None,
        scale: float = 1.0,
        units: Optional[str] = None,
    ) -> None:
        self.name: str = name
        self.sym_: ca.MX = ca.MX.sym(name)
        self.bounds: Limits = Limits() if bounds is None else bounds
        self.guess: float = 0.0 if guess is None else guess
        self.scale: float = scale
        self.units: Optional[str] = units


class OcpPhase:
    def __init__(self, name: str) -> None:
        self.name: str = "ocp" if not name else name
        self.state: dict[str, OcpVectorSymbol] = {}
        self.control: dict[str, OcpVectorSymbol] = {}
        self.param: dict[str, OcpScalarSymbol] = {}
        self.ode: dict[str, OcpVectorExpr] = {}
        self.path: dict[str, OcpVectorExpr] = {}

    def add_state(
        self,
        name: str,
        bounds_c: Optional[Limits],
        bounds_i: Optional[Limits],
        bounds_f: Optional[Limits],
        guess: Optional[npt.NDArray] = None,
        scale: float = 1.0,
        units: Optional[str] = None,
    ) -> OcpVectorSymbol:
        if name in self.state:
            raise ValueError(f"State '{name}' is already defined.")
        sym: OcpVectorSymbol = OcpVectorSymbol(
            name=name,
            bounds_c=bounds_c,
            bounds_i=bounds_i,
            bounds_f=bounds_f,
            guess=guess,
            scale=scale,
            units=units,
        )
        self.state[name] = sym
        return sym

    def add_control(
        self,
        name: str,
        bounds_c: Optional[Limits],
        bounds_i: Optional[Limits],
        bounds_f: Optional[Limits],
        guess: Optional[npt.NDArray] = None,
        scale: float = 1.0,
        units: Optional[str] = None,
    ) -> OcpVectorSymbol:
        if name in self.control:
            raise ValueError(f"Control '{name}' is already defined.")
        sym: OcpVectorSymbol = OcpVectorSymbol(
            name=name,
            bounds_c=bounds_c,
            bounds_i=bounds_i,
            bounds_f=bounds_f,
            guess=guess,
            scale=scale,
            units=units,
        )
        self.control[name] = sym
        return sym

    def add_param(
        self,
        name: str,
        bounds: Optional[Limits],
        guess: Optional[float] = None,
        scale: float = 1.0,
        units: Optional[str] = None,
    ) -> OcpScalarSymbol:
        if name in self.param:
            raise ValueError(f"Param '{name}' is already defined.")
        sym: OcpScalarSymbol = OcpScalarSymbol(
            name=name, bounds=bounds, guess=guess, scale=scale, units=units
        )
        self.param[name] = sym
        return sym

    def add_ode(
        self,
        state: OcpVectorSymbol,
        expr: ca.MX,
        bounds_c: Optional[Limits],
        bounds_i: Optional[Limits],
        bounds_f: Optional[Limits],
        scale: float = 1.0,
        units: Optional[str] = None,
    ) -> OcpVectorExpr:
        if state.name in self.ode:
            raise ValueError(f"Ode already defined for state '{state.name}'")
        vec_expr: OcpVectorExpr = OcpVectorExpr(
            name=state.name,
            expr=expr,
            bounds_c=bounds_c,
            bounds_i=bounds_i,
            bounds_f=bounds_f,
            scale=scale,
            units=units,
        )
        self.ode[state.name] = expr
        return vec_expr

    def add_path(
        self,
        name: str,
        expr: ca.MX,
        bounds_c: Optional[Limits],
        bounds_i: Optional[Limits],
        bounds_f: Optional[Limits],
        scale: float = 1.0,
        units: Optional[str] = None,
    ) -> OcpVectorExpr:
        if name in self.path:
            raise ValueError(f"Path constraint '{name}' is already defined.")
        vec_expr: OcpVectorExpr = OcpVectorExpr(
            name=name,
            expr=expr,
            bounds_c=bounds_c,
            bounds_i=bounds_i,
            bounds_f=bounds_f,
            scale=scale,
            units=units,
        )
        self.path[name] = expr
        return vec_expr

    @property
    def guess(self) -> OcpTrajectory:
        # TODO: Combine all the guesses in the problem elements into a single trajectory.
        raise NotImplementedError

    @guess.setter
    def guess(self, traj: OcpTrajectory) -> None:
        # TODO: Want to be able to set the guess with a trajectory, perhaps from a prev solution.
        raise NotImplementedError
