#master_equation.py
import numpy as np
import qutip as qt

from .operators import build_many_atom_operators, build_collapse_operators


def solve_many_atom_master_equation(num_atoms, gamma0, gammac, t_grid, verbose=True):
    """
    #many-atom-master-equation-solver
    Solves the full Hilbert-space Lindblad master equation for N atoms.

    #output
    Returns total excited-state population P_e(t).
    """

    operators = build_many_atom_operators(num_atoms)

    H = 0 * qt.qeye([2 for _ in range(num_atoms)])

    collapse_operators = build_collapse_operators(num_atoms, gamma0, gammac)

    e_ops = [operators["total_excited_population_operator"]]

    if verbose:
        print(f"Solving N = {num_atoms}")
        print(f"Hilbert dimension d = {2 ** num_atoms}")
        print(f"Density matrix entries d^2 = {(2 ** num_atoms) ** 2}")
        print(f"Number of collapse operators = {len(collapse_operators)}")

    result = qt.mesolve(
        H,
        operators["rho0"],
        t_grid,
        collapse_operators,
        e_ops=e_ops,
    )

    total_excited_population = np.real(result.expect[0])

    return total_excited_population