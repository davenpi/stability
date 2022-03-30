import jax
from jax import grad
import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt


class StabilityEvalEnv:
    """
    Environment where we will solve the optimization problem.

    Attributes
    ----------

    Methods
    -------
    """

    def __init__(
        self,
        ma: callable = None,
        l: float = 1,
        num_mesh_points: int = 50,
        B: float = 1,
        rho: float = 1,
    ):
        """

        Parameters
        ----------
        ma : callable
            Active moment function.
        l : float
            Initial guess for l.
        num_mesh_points : int
            Number of mesh points to use to find solution.
        B : float
            Bending stiffness.
        rho : float
            Density.

        """
        self.ma = ma
        self.grad_ma = jax.vmap(grad(self.ma))
        self.l = l
        self.num_mesh_points = num_mesh_points
        self.x = np.linspace(0, 1, self.num_mesh_points)
        self.y = np.zeros((2, self.num_mesh_points))
        self.make_initial_guesses()
        self.B = B
        self.rho = rho
        self.g = 9.8

    def make_initial_guesses(self):
        """
        Make initial guesses for y values.

        Just filling in the known boundary condition.
        """
        self.y[0, 0] = 0
        self.y[0, -1] = 0
        self.y[1, 0] = -self.l * self.ma(0)

    def fun(self, x: np.ndarray, y: np.ndarray, params: list) -> np.ndarray:
        """
        Returns dy/dx.

        Input x, y, and parameters and return the right hand side of the dynamical
        equations.

        Parameters
        ----------
        x : np.ndarray
            Domain where we are solving the equation. Shape (m,)
        y : np.ndarray
            Array of variables we are solving for. Shape (n,m) where n is the
            number of equations in our system.
        params : list
            List containing the parameters. For us the list is length one and only
            contains our initial guess for l.

        Returns
        -------
        dy/dx : np.ndarray
            Array containing the dynamical equations. Shape (n,m)
        """
        l = params[0]
        y_1_prime = y[1]
        dma_ds = np.array(self.grad_ma(x))
        y_2_prime = -(l / self.B) * dma_ds + (
            l ** 3 / self.B
        ) * x * self.rho * self.g * np.cos(y[0])
        dy_dx = np.vstack((y_1_prime, y_2_prime))
        return dy_dx

    def bc(self, ya: np.ndarray, yb: np.ndarray, params: list) -> np.ndarray:
        """
        Evaluate and return boundary condition residuals.

        Parameters
        ----------
        ya : np.ndarray
            Array containing the value of each y at x = a (the left side of
            the domain).
        yb : np.ndarray
            Array containing the value of each y at x = b (the right side of
            the domain).
        params : list
            List containing the parameters. For us the list is length one and only
            contains our initial guess for l.
        """
        l = params[0]
        res1 = ya[0]
        res2 = ya[1] + l * self.ma(0.0).item()
        res3 = yb[0]
        residuals = np.array([res1, res2, res3])
        return residuals

    def solve(self):
        self.sol = si.solve_bvp(self.fun, self.bc, self.x, self.y, p=[self.l])
        self.x_pos = self.sol.p[0] * si.cumtrapz(np.cos(self.sol.y[0]), self.sol.x)
        self.x_pos -= self.x_pos[-1]
        self.y_pos = self.sol.p[0] * si.cumtrapz(np.sin(self.sol.y[0]), self.sol.x)
        self.y_pos -= self.y_pos[-1]
        print(f"Solution says l = {round(self.sol.p[0], 3)}")

    def get_cost(self, alpha=0.5):
        """
        Compute the cost function we are trying to minimize.

        Note we have the computed values as a function of r so we should
        integrate from 0 to 1 instead of 0 to l.
        """
        r = self.sol.x
        dtheta_dr = self.sol.y[1]
        active_moment = self.ma(r)
        work = 0.5 * si.trapz(active_moment * dtheta_dr, r)
        height = self.y_pos[0]
        cost = (1 - alpha) * work - alpha * height
        return cost

    def plot_snake(self):
        plt.plot(self.x_pos, self.y_pos)
        plt.title("Snake layout")
        plt.show()

    def plot_solutions(self):
        plt.plot(self.sol.x, self.sol.y[0], label=r"$\theta$")
        plt.plot(self.sol.x, self.sol.y[1], label=r"$\frac{d\theta}{d\tilde{s}}$")
        plt.xlabel(r"$\tilde{s}$")
        plt.ylabel(r"Solutions")
        plt.title("Solutions to ODE")
        plt.legend()
        plt.show()
