#operators.py
import numpy as np
import qutip as qt


#single-atom-basis-and-operators
ket_e_single = qt.basis(2, 0)
ket_g_single = qt.basis(2, 1)

sigma_minus_single = ket_g_single * ket_e_single.dag()
sigma_plus_single = sigma_minus_single.dag()
sigma_ee_single = ket_e_single * ket_e_single.dag()
identity_single = qt.qeye(2)


def local_operator(single_atom_operator, atom_index, num_atoms):
    """
    #local-operator-calc
    Places a single-atom operator on atom_index and identities on all other atoms.
    """

    operator_list = []

    for index in range(num_atoms):
        if index == atom_index:
            operator_list.append(single_atom_operator)
        else:
            operator_list.append(identity_single)

    return qt.tensor(operator_list)


def build_many_atom_operators(num_atoms):
    """
    #many-atom-operators-calc
    Builds local lowering operators, excited-state projectors, collective lowering operator,
    total excited population operator, and the all-excited initial state.
    """

    sigma_minus_ops = [
        local_operator(sigma_minus_single, atom_index, num_atoms)
        for atom_index in range(num_atoms)
    ]

    sigma_ee_ops = [
        local_operator(sigma_ee_single, atom_index, num_atoms)
        for atom_index in range(num_atoms)
    ]

    J_minus = sum(sigma_minus_ops)
    total_excited_population_operator = sum(sigma_ee_ops)

    psi0 = qt.tensor([ket_e_single for _ in range(num_atoms)])
    rho0 = psi0 * psi0.dag()

    return {
        "sigma_minus_ops": sigma_minus_ops,
        "sigma_ee_ops": sigma_ee_ops,
        "J_minus": J_minus,
        "total_excited_population_operator": total_excited_population_operator,
        "psi0": psi0,
        "rho0": rho0,
    }


def build_collapse_operators(num_atoms, gamma0, gammac):
    """
    #collapse-operators-calc
    Builds collapse operators for collective cavity decay and independent free-space decay.

    #collective-cavity-decay
    L_c = sqrt(gammac) J_minus

    #independent-free-space-decay
    L_i = sqrt(gamma0) sigma_minus_i
    """

    operators = build_many_atom_operators(num_atoms)
    collapse_operators = []

    if gammac > 0:
        L_cavity = np.sqrt(gammac) * operators["J_minus"]
        collapse_operators.append(L_cavity)

    if gamma0 > 0:
        for sigma_minus_i in operators["sigma_minus_ops"]:
            L_free_i = np.sqrt(gamma0) * sigma_minus_i
            collapse_operators.append(L_free_i)

    return collapse_operators