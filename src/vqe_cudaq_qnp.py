"""
    Contains the class with the VQE using the quantum-number-preserving ansatz
"""
import os
import numpy as np
import cudaq
from cudaq import spin as spin_op
import time
from scipy.optimize import minimize
from datetime import datetime


class VQE(object):
    """
        Implements the quantum-number-preserving ansatz from Anselmetti et al. NJP 23 (2021)
    """

    def __init__(self,
                 n_qubits,
                 num_active_electrons,
                 spin,
                 options):
        self.n_qubits = n_qubits
        self.n_layers = options.get('n_vqe_layers', 1)
        self.number_of_Q_blocks = n_qubits // 2 - 1
        self.num_params = 2 * self.number_of_Q_blocks * self.n_layers
        self.options = options
        num_active_orbitals = n_qubits // 2

        # number of alpha and beta electrons in the active space
        num_active_electrons_alpha = (num_active_electrons + spin) // 2
        num_active_electrons_beta = (num_active_electrons - spin) // 2

        # Define the initial state for the VQE as a list
        # [n_1, n_2, ....]
        # where n_j=(0,1,2) is the occupation of j-th the orbital

        n_alpha_vec = [1] * num_active_electrons_alpha + [0] * (num_active_orbitals - num_active_electrons_alpha)
        n_beta_vec = [1] * num_active_electrons_beta + [0] * (num_active_orbitals - num_active_electrons_beta)
        init_mo_occ = [n_a + n_b for n_a, n_b in zip(n_alpha_vec, n_beta_vec)]

        self.init_mo_occ = init_mo_occ
        self.final_state_vector_best = None
        self.best_vqe_params = None
        self.best_vqe_energy = None
        self.target = options.get("target", "nvidia")
        self.target_option = options.get("target_option", "mgpu")
        self.num_qpus = 0
        self.initial_x_gates_pos = self.prepare_initial_circuit()
        self.str_date = datetime.today().strftime('%Y%m%d_%H%M%S')

    def prepare_initial_circuit(self):
        """
        Creates a list with the position of the X gates that should be applied to the initial |00...0>
        state to set the number of electrons and the spin correctly
        """
        x_gates_pos_list = []
        if self.init_mo_occ is not None:
            for idx_occ, occ in enumerate(self.init_mo_occ):
                if int(occ) == 2:
                    x_gates_pos_list.extend([2 * idx_occ, 2 * idx_occ + 1])
                elif int(occ) == 1:
                    x_gates_pos_list.append(2 * idx_occ)

        return x_gates_pos_list

    def layers(self):
        """
            Generates the QNP ansatz circuit and returns the  kernel and the optimization parameters thetas

        :return: tuple
            A tuple containing the constructed quantum kernel and the list of theta parameters.

        """

        n_qubits = self.n_qubits
        n_layers = self.n_layers
        number_of_blocks = self.number_of_Q_blocks

        kernel, thetas = cudaq.make_kernel(list)
        qubits = kernel.qalloc(n_qubits)

        for init_gate_position in self.initial_x_gates_pos:
            kernel.x(qubits[init_gate_position])

        count_params = 0
        for idx_layer in range(n_layers):
            for starting_block_num in [0, 1]:
                for idx_block in range(starting_block_num, number_of_blocks, 2):
                    qubit_list = [qubits[2 * idx_block + j] for j in range(4)]

                    # PX gates decomposed in terms of standard gates
                    # and NO controlled Y rotations.
                    # See Appendix E1 of Anselmetti et al New J. Phys. 23 (2021) 113010

                    a, b, c, d = qubit_list
                    kernel.cx(d, b)
                    kernel.cx(d, a)
                    kernel.rz(parameter=-np.pi / 2, target=a)
                    kernel.s(b)
                    kernel.h(d)
                    kernel.cx(d, c)
                    kernel.cx(b, a)
                    kernel.ry(parameter=(1 / 8) * thetas[count_params], target=c)
                    kernel.ry(parameter=(-1 / 8) * thetas[count_params], target=d)
                    kernel.rz(parameter=+np.pi / 2, target=a)
                    kernel.cz(a, d)
                    kernel.cx(a, c)
                    kernel.ry(parameter=(-1 / 8) * thetas[count_params], target=d)
                    kernel.ry(parameter=(+1 / 8) * thetas[count_params], target=c)
                    kernel.cx(b, c)
                    kernel.cx(b, d)
                    kernel.rz(parameter=+np.pi / 2, target=b)
                    kernel.ry(parameter=(-1 / 8) * thetas[count_params], target=c)
                    kernel.ry(parameter=(+1 / 8) * thetas[count_params], target=d)
                    kernel.cx(a, c)
                    kernel.cz(a, d)
                    kernel.ry(parameter=(-1 / 8) * thetas[count_params], target=c)
                    kernel.ry(parameter=(1 / 8) * thetas[count_params], target=d)
                    kernel.cx(d, c)
                    kernel.h(d)
                    kernel.cx(d, b)
                    kernel.s(d)
                    kernel.rz(parameter=-np.pi / 2, target=b)
                    kernel.cx(b, a)
                    count_params += 1

                    # Orbital rotation
                    kernel.fermionic_swap(np.pi, b, c)
                    kernel.givens_rotation((-1 / 2) * thetas[count_params], a, b)
                    kernel.givens_rotation((-1 / 2) * thetas[count_params], c, d)
                    kernel.fermionic_swap(np.pi, b, c)
                    count_params += 1

        return kernel, thetas

    def get_state_vector(self, param_list):
        """
        Returns the state vector generated by the ansatz with parameters given by param_list
        """
        kernel, thetas = self.layers()
        state = convert_state_big_endian(np.array(cudaq.get_state(kernel, param_list), dtype=complex))
        return state

    def execute(self, hamiltonian):
        """
        Run VQE
        """
        start_t = time.time()
        options = self.options
        maxiter = options.get('maxiter', 100)
        method_optimizer = options.get("optimizer", "COBYLA")
        mpi_support = any([options.get(key, False) for key in ["mpi", "mpi_support"]])
        return_final_state_vec = options.get("return_final_state_vec", False)
        initial_parameters = options.get('initial_parameters', None)
        if self.target == "nvidia":
            # ("mgpu", "tensornet", "nvidia-mgpu"):
            cudaq.set_target("nvidia", option=self.target_option)
            target = cudaq.get_target()
            if mpi_support:
                cudaq.mpi.initialize()
                num_ranks = cudaq.mpi.num_ranks()
                rank = cudaq.mpi.rank()
                print('# rank', rank, 'num_ranks', num_ranks)
                self.num_qpus = target.num_qpus()
                if rank == 0:
                    print(f"# Set target nvidia with options {self.target_option}")
                    print('# mpi is initialized? ', cudaq.mpi.is_initialized())
                    print("# num gpus=", target.num_qpus())
            else:
                print(f"# Set target nvidia with options {self.target_option}")
                print("# num gpus=", target.num_qpus())

        elif self.target in ('tensornet', "tensornet-mps"):
            cudaq.set_target(self.target)
            print(f"# Set target {self.target}")
            target = cudaq.get_target()
            self.num_qpus = target.num_qpus()
            if mpi_support:
                cudaq.mpi.initialize()
                num_ranks = cudaq.mpi.num_ranks()
                rank = cudaq.mpi.rank()
                print('# rank', rank, 'num_ranks', num_ranks)
                self.num_qpus = target.num_qpus()
                if rank == 0:
                    print(f"# Set target nvidia with options {self.target_option}")
                    print('# mpi is initialized? ', cudaq.mpi.is_initialized())
                    print("# num gpus=", target.num_qpus())

        elif self.target == 'qpp-cpu':
            print(f"# Set target qpp-cpu")
            cudaq.set_target('qpp-cpu')
            self.num_qpus = 0

        else:
            print(f"# Target not defined")
            exit()

        if initial_parameters is not None:
            initial_parameters = np.pad(initial_parameters, (0, self.num_params - len(initial_parameters)),
                                        constant_values=0.01)
        else:
            initial_parameters = np.random.uniform(low=-np.pi, high=np.pi, size=self.num_params)
        # print("# Init params")
        # print(initial_parameters)
        kernel, thetas = self.layers()
        callback_energies = []

        def cost(theta):
            """
            Compute the energy by using different execution types and cudaq.observe
            """
            if self.num_qpus:
                if self.target == "nvidia":
                    if mpi_support:
                        exp_val = cudaq.observe(kernel,
                                                hamiltonian,
                                                theta,
                                                execution=cudaq.parallel.mpi).expectation()
                    else:
                        exp_val = cudaq.observe(kernel,
                                                hamiltonian,
                                                theta,
                                                execution=cudaq.parallel.thread).expectation()
                elif self.target in ("tensornet", "tensornet-mps"):
                    exp_val = cudaq.observe(kernel,
                                            hamiltonian,
                                            theta,
                                            ).expectation()
                else:
                    print(f"Target {self.target} not supported")
                    exit()
            else:
                exp_val = cudaq.observe(kernel,
                                        hamiltonian,
                                        theta).expectation()

            callback_energies.append(exp_val)
            print('exp_val', exp_val)
            return exp_val

        energy_core = options.get('energy_core', 0.)
        initial_energy = cost(initial_parameters)
        print("# Initial energy: ", initial_energy + energy_core)
        print("# Start VQE minimization")
            
        result_optimizer = minimize(cost,
                                    initial_parameters,
                                    method=method_optimizer,
                                    options={'maxiter': maxiter})

        best_parameters = result_optimizer['x']
        energy_optimized = result_optimizer['fun']

        total_opt_energy = energy_optimized + energy_core
        # We add here the energy core
        callback_energies = [en + energy_core for en in callback_energies]
        end_t = time.time()
        print("# Num params:", self.num_params)
        print("# Qubits:", self.n_qubits)
        print("# N_layers:", self.n_layers)
        print("# Energy after the VQE:", total_opt_energy)
        print("# Time for VQE [min]:", (end_t - start_t) / 60.)

        result = {"energy_optimized": total_opt_energy,
                  "best_parameters": best_parameters,
                  "callback_energies": callback_energies,
                  "time_vqe": end_t - start_t,
                  "initial_energy": initial_energy + energy_core}

        cwd = os.getcwd()
        save_data_dir = os.path.join(cwd, "data")
        os.makedirs(save_data_dir, exist_ok=True)

        fname = f"callback_energies_fenta_" \
                f"cas_{self.n_qubits}q_" \
                f"layer_{self.n_layers}_opt_{method_optimizer}.dat"

        np.savetxt(os.path.join(save_data_dir, fname),
                   callback_energies)

        fname = f"best_params_fenta_" \
                f"cas_{self.n_qubits}q_" \
                f"layer_{self.n_layers}_opt_{method_optimizer}.dat"

        np.savetxt(os.path.join(save_data_dir, fname),
                   best_parameters)

        if return_final_state_vec:
            result["state_vec"] = self.get_state_vector(best_parameters)
            fname = f"state_vec_fenta_" \
                    f"cas_{self.n_qubits}q_" \
                    f"layer_{self.n_layers}_opt_{method_optimizer}.dat"

            np.save(os.path.join(save_data_dir, fname),
                    result["state_vec"])

        return result


def convert_state_big_endian(state_little_endian):
    """
    Converts a state vector from little-endian format (cudaq) to big-endian format (ipie).
    """
    state_big_endian = 0. * state_little_endian

    n_qubits = int(np.log2(state_big_endian.size))
    for j, val in enumerate(state_little_endian):
        little_endian_pos = np.binary_repr(j, n_qubits)
        big_endian_pos = little_endian_pos[::-1]
        int_big_endian_pos = int(big_endian_pos, 2)
        state_big_endian[int_big_endian_pos] = state_little_endian[j]

    return state_big_endian


def from_string_to_cudaq_spin(pauli_string, qubit):
    """
    Convert a Pauli string in string format to a spin operator in CUDAq format.

    :param pauli_string: The Pauli string in string format ('I', 'X', 'Y', or 'Z').
    :type pauli_string: str

    :param qubit: The qubit index.
    :type qubit: int

    :return: The spin operator in CUDAq format.
    :rtype: spin

    """
    if pauli_string.lower() in ('id', 'i'):
        return 1
    elif pauli_string.lower() == 'x':
        return spin_op.x(qubit)
    elif pauli_string.lower() == 'y':
        return spin_op.y(qubit)
    elif pauli_string.lower() == 'z':
        return spin_op.z(qubit)


def get_cudaq_hamiltonian(jw_hamiltonian):
    """
    Converts a Jordan-Wigner Hamiltonian to a CUDA Quantum Hamiltonian.

    This function processes a given Jordan-Wigner Hamiltonian and converts it
    into a format suitable for CUDA quantum computations. The input Hamiltonian
    is a list of terms, where each term is a dictionary containing operators and
    their corresponding coefficients. The function iterates through each term,
    constructs the corresponding CUDA quantum operator, and sums up the terms to
    produce the final CUDA quantum Hamiltonian. Additionally, it extracts any
    constant energy offset present in the Hamiltonian.

    :param jw_hamiltonian: List of Hamiltonian terms in Jordan-Wigner form. Each term
                           is a dictionary with operators as keys and coefficients as values.
    :type jw_hamiltonian: list of dict
    :return: A tuple containing the CUDA quantum Hamiltonian and the core energy.
    :rtype: tuple (float, float)
    """

    hamiltonian_cudaq = 0.0
    energy_core = 0.0
    for ham_term in jw_hamiltonian:
        [(operators, ham_coeff)] = ham_term.terms.items()
        if len(operators):
            cuda_operator = 1.0
            for qubit_index, pauli_op in operators:
                cuda_operator *= from_string_to_cudaq_spin(pauli_op, qubit_index)
        else:
            cuda_operator = 0.0
            energy_core = ham_coeff
        cuda_operator = ham_coeff * cuda_operator
        hamiltonian_cudaq += cuda_operator

    return hamiltonian_cudaq, energy_core
