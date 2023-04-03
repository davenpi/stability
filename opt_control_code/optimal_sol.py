"""
Contains methods to solve the optimal control problem.
"""
import numpy as np
import casadi as ca
from solvers import collocation_solver


def optimal_snake(
    N: int,
    use_upsampled_prior: bool,
    y_f: float,
    x_f: float,
    opt_guess=None,
):
    """
    Implement the snake dynamics and define the control problem.

    The reason for many of the extra boolean arguments is because I need to
    solve a different version of the control problem to create an initial guess
    for the time optimal problem. The boolean flags are used to indicate which
    problem I am solving and whether or not I want to use prior solutions as an
    initial guess.
    Also x_f is determined during the initial guess building phase. x_f is
    given by the maximum x that is reached when solving the first optimization
    problem.

    Parameters
    ----------
    N : int
        Number of control intervals.
    use_upsampled_prior : bool
        Passed directly to the collocation solver. This argument is True if
        I want to use a prior solution I constructed as an initial guess. In
        particular these prior solutions are the upsampled prior solutions from
        previous runs.
    y_f : float
        The final y position we are aiming for.
    x_f : float
        The final x position we are aiming for. Set to 0 if we are not
        including a final x target.
    energy_optimal: bool
        True if we want to solve the energy optimal problem.
    """
    alpha = 1  # rho*g/B

    # Degree of interpolating polynomial
    d = 3

    # Control discretization
    N = N  # number of control intervals

    # Time horizon. This is a trick to make the final time a parameter for the
    # time optimal problem.
    T = 1.0

    # declare model variables
    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    theta = ca.SX.sym("theta")
    theta_prime = ca.SX.sym("theta_prime")
    s = ca.SX.sym("s")
    state = ca.vertcat(s, x, y, theta, theta_prime)
    u = ca.SX.sym("u")

    # model equations
    ds_ds = 1
    dx_ds = ca.cos(theta)
    dy_ds = ca.sin(theta)
    dtheta_ds = theta_prime
    d2theta_ds = alpha * (s - 1)*ca.cos(theta) - u
    xdot = ca.vertcat(
        ds_ds,
        dx_ds,
        dy_ds,
        dtheta_ds,
        d2theta_ds,
    )

    # Objective term. The thing to be minimized by the controller.
    L = u**2

    # Define the casadi function we will pass to the solver.
    f = ca.Function(
        "f", [state, u], [xdot, L], ["state", "u"], ["xdot", "L"]
    )

    # initial state
    x0 = [0, 0, 0, 0, 0]

    # Final state
    eq1 = y - y_f
    eq2 = x - x_f
    eq3 = theta - np.pi / 2
    eq = ca.vertcat(eq1, eq2, eq3)

    xf_eq = ca.Function("xf_eq", [state], [eq], ["state"], ["eq"])

    # State Constraints
    x_lb = [0, -np.inf, -np.inf, -np.inf, -np.inf]
    x_ub = [1, np.inf, np.inf, np.inf, np.inf]

    # Control bounds
    u_lb = -1.0
    u_ub = 1.0

    # Parameter bounds and initial guess
    # tf_guess = 17.0
    # p_lb = [1.0]
    # p_ub = [180.0]
    # p0 = [tf_guess]

    x_opt, u_opt, sol_x, sol = collocation_solver(
        f,
        x0,
        x_lb,
        x_ub,
        N,
        T,
        xf_eq=xf_eq,
        u_lb=u_lb,
        u_ub=u_ub,
        # p0=p0,
        # p_lb=p_lb,
        # p_ub=p_ub,
        opt_guess=opt_guess,
        use_upsampled_prior=use_upsampled_prior,
        d=d,
    )
    return x_opt, u_opt, sol_x, sol


if __name__ == "__main__":
    optimal_snake()
