#dicke.py
import numpy as np
from scipy.integrate import solve_ivp


def dicke_decay_rate(n, num_atoms, gammac):
    """
    #dicke-decay-rate-calc
    Computes collective decay rate from |n> to |n-1>.
    """

    return gammac * n * (num_atoms - n + 1)


def dicke_rate_equation_rhs(t, p, num_atoms, gammac):
    """
    #dicke-rate-equation-calc
    Computes dp_n/dt for n = 0, 1, ..., N.
    """

    dp_dt = np.zeros_like(p)

    for n in range(num_atoms + 1):

        if n + 1 <= num_atoms:
            rate_from_np1_to_n = dicke_decay_rate(n + 1, num_atoms, gammac)
            dp_dt[n] += rate_from_np1_to_n * p[n + 1]

        if n >= 1:
            rate_from_n_to_nm1 = dicke_decay_rate(n, num_atoms, gammac)
            dp_dt[n] -= rate_from_n_to_nm1 * p[n]

    return dp_dt


def solve_dicke_rate_equation(num_atoms, gammac, t_grid):
    """
    #solve-dicke-rate-equation-calc
    Solves the Dicke-basis rate equation for N atoms.
    """

    p0 = np.zeros(num_atoms + 1)
    p0[num_atoms] = 1.0

    solution = solve_ivp(
        fun=lambda t, p: dicke_rate_equation_rhs(t, p, num_atoms, gammac),
        t_span=(t_grid[0], t_grid[-1]),
        y0=p0,
        t_eval=t_grid,
        method="RK45",
        rtol=1e-9,
        atol=1e-11,
    )

    if not solution.success:
        print("Solver failed for N =", num_atoms)
        print(solution.message)

    return solution.y.T


def compute_dicke_observables(probabilities, num_atoms, gammac):
    """
    #dicke-observables-calc
    Computes total excited population P_e(t) and photon emission rate R(t).
    """

    n_values = np.arange(num_atoms + 1)

    total_excited_population = probabilities @ n_values

    transition_rates = np.array([
        dicke_decay_rate(n, num_atoms, gammac)
        for n in n_values
    ])

    emission_rate = probabilities @ transition_rates

    return total_excited_population, emission_rate