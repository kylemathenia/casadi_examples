"""Examples of som casadi basics."""

import casadi as ca
import numpy as np
import numpy.typing as npt


def send_it() -> None:
    """Casadi basics."""
    # Create some symbolic variables.
    pressure: ca.MX = ca.MX.sym("pressure")
    area: ca.MX = ca.MX.sym("area")

    # Do math with symbols.
    force: ca.MX = pressure * area

    # Build larger expressions.
    mass: ca.MX = ca.MX.sym("mass")
    accel: ca.MX = force / mass
    print(f"'accel' depends on symbolic vars: {ca.symvar(accel)}")

    # Create a casadi function.
    symbolic_var_inputs: list[ca.MX] = [pressure, area, mass]
    input_names: list[str] = [ca_mx.name() for ca_mx in symbolic_var_inputs]
    output_exprs: list[ca.MX] = [accel, force]
    output_names: list[str] = ["accel", "force"]
    func: ca.Function
    func = ca.Function(
        "func", symbolic_var_inputs, output_exprs, input_names, output_names
    )
    print(f"The function inputs are: {func.name_in()}")
    print(f"The function outputs are: {func.name_out()}")

    # Evaluate the function based on arg order.
    print(func(1, 2, 3))
    # Evaluate the function with key-value pairs.
    print(func.call({"area": 2, "mass": 3, "pressure": 1}))

    # Function output expressions don't need to use all the inputs.
    smell: ca.MX = ca.MX.sym("smell")
    symbolic_var_inputs = [pressure, area, mass, smell]
    input_names = [ca_mx.name() for ca_mx in symbolic_var_inputs]
    func = ca.Function(
        "func", symbolic_var_inputs, output_exprs, input_names, output_names
    )
    print(func.call({"area": 2, "mass": 3, "pressure": 1, "smell": 4}))

    # Creating functions from data is very helpful.
    force_data: npt.NDArray = np.linspace(0, 5, 6)
    displacement_data: npt.NDArray = np.array([0, 2, 5, 10, 50, 500])
    displacement_interp: ca.Function = ca.interpolant(
        "displacement_interp", "bspline", [force_data], displacement_data
    )
    print(f"Displacement from a force of 4: {displacement_interp(4)}")

    # You can evaluate interpolants symbolically.
    # Now we have a symbolic expression for displacement as a function of pressure and area.
    displacement_expr: ca.MX = displacement_interp.call([force])[0]
    print(f"'displacement_expr': {displacement_expr}")
    print(
        f"'displacement_expr' expression depends on symbolic vars: {ca.symvar(displacement_expr)}"
    )

    # Substituting symbolics for numbers or other symbolics is quite useful.
    displacement_expr = ca.substitute(displacement_expr, area, 2)
    print(f"'displacement_expr': {displacement_expr}")
    print(
        f"'displacement_expr' expression now depends on symbolic vars: {ca.symvar(displacement_expr)}"
    )

    # Automatic differentiation.
    print(f"Force: {force}")
    d_force_d_area = ca.jacobian(force, area)
    print(f"Partial derivative of force with respect to area is: '{d_force_d_area}'")
    d_force_d_pressure = ca.jacobian(force, pressure)
    print(
        f"Partial derivative of force with respect to pressure is: '{d_force_d_pressure}'"
    )


if __name__ == "__main__":
    send_it()
