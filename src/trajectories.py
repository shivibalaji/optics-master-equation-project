#trajectories.py
import numpy as np
import scipy.sparse as sp_sparse

from .operators import build_many_atom_operators, build_collapse_operators


def qobj_to_csr(qobj):
    """
    #qobj-to-csr-calc
    Converts a QuTiP Qobj into a scipy CSR sparse matrix.
    """

    try:
        return qobj.data.as_scipy().tocsr()
    except Exception:
        return sp_sparse.csr_matrix(qobj.full())


def solve_manual_quantum_trajectories(
    num_atoms,
    gamma0,
    gammac,
    t_grid,
    num_trajectories,
    max_internal_dt=5e-4,
    seed=1234,
):
    """
    #manual-quantum-trajectories-calc
    Fixed-time-step Monte Carlo wavefunction solver.

    #method
    At each small internal time step:
    p_j = dt * <psi|L_j^dagger L_j|psi>.
    """

    rng = np.random.default_rng(seed)

    operators = build_many_atom_operators(num_atoms)

    collapse_operators_qobj = build_collapse_operators(num_atoms, gamma0, gammac)
    collapse_operators = [qobj_to_csr(L) for L in collapse_operators_qobj]

    K = None
    for L in collapse_operators:
        LdaggerL = L.getH() @ L
        K = LdaggerL if K is None else K + LdaggerL

    P_op = qobj_to_csr(operators["total_excited_population_operator"])
    psi0 = np.asarray(operators["psi0"].full()).reshape(-1).astype(complex)

    population_sum = np.zeros(len(t_grid), dtype=float)
    max_jump_probability_seen = 0.0

    for traj_index in range(num_trajectories):

        psi = psi0.copy()
        population_traj = np.zeros(len(t_grid), dtype=float)

        for time_index in range(len(t_grid)):

            population_traj[time_index] = np.real(np.vdot(psi, P_op @ psi))

            if time_index == len(t_grid) - 1:
                break

            dt_output = t_grid[time_index + 1] - t_grid[time_index]
            num_substeps = int(np.ceil(dt_output / max_internal_dt))
            dt = dt_output / num_substeps

            for _ in range(num_substeps):

                rates = np.empty(len(collapse_operators), dtype=float)
                Lpsi_list = []

                for channel_index, L in enumerate(collapse_operators):
                    Lpsi = L @ psi
                    Lpsi_list.append(Lpsi)
                    rates[channel_index] = np.real(np.vdot(Lpsi, Lpsi))

                total_rate = np.sum(rates)
                jump_probability = dt * total_rate

                max_jump_probability_seen = max(max_jump_probability_seen, jump_probability)

                if jump_probability > 0.2:
                    raise RuntimeError(
                        f"Jump probability too large: {jump_probability:.3f}. "
                        "Decrease max_internal_dt."
                    )

                if rng.random() < jump_probability and total_rate > 0:

                    channel_probabilities = rates / total_rate
                    chosen_channel = rng.choice(
                        len(collapse_operators),
                        p=channel_probabilities,
                    )

                    psi = Lpsi_list[chosen_channel]
                    psi = psi / np.linalg.norm(psi)

                else:

                    psi = psi - 0.5 * dt * (K @ psi)
                    psi = psi / np.linalg.norm(psi)

        population_sum += population_traj

        if (traj_index + 1) % max(1, num_trajectories // 10) == 0:
            print(f"Completed {traj_index + 1}/{num_trajectories} trajectories")

    average_population = population_sum / num_trajectories

    print("Maximum internal jump probability seen:", max_jump_probability_seen)

    return average_population